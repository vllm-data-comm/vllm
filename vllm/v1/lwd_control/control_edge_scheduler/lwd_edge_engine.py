# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Edge engine core: the control plane to the same-index cloud DP rank, the
check of the cloud's scheduling limits and the request entry (edge admission +
announcement to the cloud; communication stays in the engine).

Stepping is the native step_with_batch_queue, adapted at construction: depth
2, is_pooling_model=True (no native sample_tokens) and an executor proxy (a
normally completed batch returning None becomes a placeholder).

Startup blocks until the cloud accepts the handshake and sends its scheduling
limits, within one handshake timeout; any failure stops the engine. There is
no self-healing: restart the whole group.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from vllm.lwd_logger import lwd_logger
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_executor_proxy import (
    LwdEdgeExecutorProxy,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
    LwdEdgeScheduler,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_engine import (
    LwdBaseEngineCore,
)
from vllm.v1.lwd_control.lwd_comm.lwd_control_plane import LwdEdgeControlPlane
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import LwdPeer
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LWD_SAMPLING_DEFAULTS,
    LwdC2eNotify,
    LwdCloudParamsNotify,
    LwdEdgeToCloudMessage,
    LwdMessage,
    LwdRequestNotify,
    lwd_decode_cloud_to_edge,
)
from vllm.v1.request import Request

# Native batch queue depth: dispatched (embed + unembed) batches not yet
# reaped. 2 keeps the pipeline aligned; the reference branch's 4 is
# conservative, validate performance with 2 first.
LWD_EDGE_BATCH_QUEUE_DEPTH = 2

# A full outbound queue is retried with growing intervals (about 3 s in
# total) before failing: by then the cloud stopped reading or is gone.
_LWD_SEND_RETRY_STEPS = 5
_LWD_SEND_RETRY_INTERVAL_S = 0.2


class LwdEdgeEngineCore(LwdBaseEngineCore):
    """Edge PO engine: control plane + cloud limits check + native stepping."""

    lwd_scheduler_cls = LwdEdgeScheduler
    scheduler: LwdEdgeScheduler

    def _lwd_assemble(self) -> None:
        self._lwd_cloud_params: LwdCloudParamsNotify | None = None
        self._lwd_cloud_params_received = threading.Event()
        plane = self._lwd_start_control_plane()
        assert isinstance(plane, LwdEdgeControlPlane)
        # One handshake_timeout_s covers the accept and the limits: it bounds
        # the wait for a cloud ready to serve.
        handshake_timeout_s = plane.spec.config.handshake_timeout_s
        deadline = time.monotonic() + handshake_timeout_s
        self._lwd_cloud: LwdPeer = plane.wait_until_ready(handshake_timeout_s)
        if not self._lwd_cloud_params_received.wait(
            max(0.0, deadline - time.monotonic())
        ):
            raise RuntimeError(
                f"[LWD] edge engine init failed: cloud {self._lwd_cloud} at "
                f"{plane.endpoint} accepted the handshake but sent no scheduling "
                f"limits within {handshake_timeout_s}s"
            )
        self._lwd_check_cloud_params()
        self.scheduler.lwd_send_to_cloud = self._lwd_send_to_cloud
        # Native batch queue stepping (step/step_with_batch_queue untouched):
        # 1. depth 2, rebuilding the deque so its maxlen matches and
        #    appendleft on a full queue cannot silently evict a batch in
        #    flight;
        # 2. the edge samples inside execute, so is_pooling_model stops the
        #    native sample_tokens call;
        # 3. the executor proxy.
        self.is_pooling_model = True
        self.batch_queue_size = LWD_EDGE_BATCH_QUEUE_DEPTH
        self.batch_queue = deque(maxlen=self.batch_queue_size)
        self.step_fn = self.step_with_batch_queue
        self.model_executor = LwdEdgeExecutorProxy(self.model_executor)
        lwd_logger.info(
            "[edge] engine assembled: cloud %s at %s, native batch-queue "
            "stepping depth=%d",
            self._lwd_cloud,
            plane.endpoint,
            self.batch_queue_size,
        )

    def _lwd_decode(self, payload: bytes) -> LwdMessage:
        return lwd_decode_cloud_to_edge(payload)

    def _lwd_dispatch_message(self, peer: LwdPeer, msg: LwdMessage) -> bool:
        """Cloud limits -> wake the startup wait; c2e -> scheduler notify
        queue + WAKEUP; anything else is unknown."""
        if isinstance(msg, LwdCloudParamsNotify):
            self._lwd_cloud_params = msg
            self._lwd_cloud_params_received.set()
            return True
        if isinstance(msg, LwdC2eNotify):
            lwd_logger.info(
                "[edge] c2e received reqs=%d hidden_numel=%d",
                len(msg.req_ids),
                msg.hidden_num_elements,
            )
            self.scheduler.unembed_notify_queue.append(msg)
            self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))
            return True
        return False

    def _lwd_check_cloud_params(self) -> None:
        """(max_num_batched_tokens, max_num_seqs) must match the cloud: both
        feed the cap/preflight formulas, and a mismatch makes the cloud hold
        prefills behind decodes forever without any error. The speculative
        factor and drafting slots are adopted from the cloud and liveness
        re-checked with them."""
        params = self._lwd_cloud_params
        assert params is not None
        sched = self.vllm_config.scheduler_config
        local = (sched.max_num_batched_tokens, sched.max_num_seqs)
        remote = (params.max_num_batched_tokens, params.max_num_seqs)
        if local != remote:
            raise RuntimeError(
                f"[LWD] edge/cloud scheduling config mismatch: "
                f"(max_num_batched_tokens, max_num_seqs) edge={local} != "
                f"cloud={remote}; a mismatch makes the cloud hold prefills "
                f"behind decodes forever, so start both sides with the same values"
            )
        self._lwd_adopt_cloud_spec(params)
        lwd_logger.info("[cloud-params] scheduling limits verified: %s", local)
        self._lwd_check_engram_fingerprint(params.engram_fingerprint)

    def _lwd_adopt_cloud_spec(self, params: LwdCloudParamsNotify) -> None:
        """Adopt the cloud's spec budget terms: they reserve draft budget of
        cloud steps, and the edge has no legitimate local source (a spec
        config on the edge would start the draft machinery)."""
        scheduler = self.scheduler
        local = (scheduler._lwd_spec_factor, scheduler._lwd_draft_slots)
        remote = (params.num_speculative_tokens + 1, params.num_draft_slots)
        if local == remote:
            return
        lwd_logger.warning(
            "[cloud-params] adopt cloud (spec_factor, draft_slots)=%s over "
            "edge local %s",
            remote,
            local,
        )
        scheduler._lwd_spec_factor, scheduler._lwd_draft_slots = remote
        # The scheduler checked liveness with the local terms.
        scheduler._lwd_check_deadlock_config()

    def _lwd_check_engram_fingerprint(self, remote: str) -> None:
        """The edge's Engram bucket indices must address the cloud's tables:
        both sides derive the hash parameters from their own tokenizer and
        config, and a mismatch would hash every prompt into wrong rows
        without any error."""
        local = self.scheduler.lwd_model_family.engram_fingerprint()
        if local != remote:
            raise RuntimeError(
                f"[LWD] edge/cloud Engram hash parameters mismatch: edge "
                f"fingerprint {local or '<no Engram>'} != cloud "
                f"{remote or '<no Engram>'}; serve the same checkpoint and "
                f"tokenizer on both sides"
            )
        if local:
            lwd_logger.info("[LWD] Engram hash fingerprint verified: %s", local)

    def _lwd_send_to_cloud(self, message: LwdEdgeToCloudMessage) -> None:
        """Send to the cloud, retrying a full outbound queue; raises once the
        retries run out instead of losing the message."""
        for attempt in range(_LWD_SEND_RETRY_STEPS):
            if self._lwd_send(self._lwd_cloud, message):
                return
            time.sleep(_LWD_SEND_RETRY_INTERVAL_S * (attempt + 1))
        raise RuntimeError(
            f"[LWD] {type(message).__name__} to cloud {self._lwd_cloud} not "
            f"sent: outbound queue still full after {_LWD_SEND_RETRY_STEPS} "
            f"retries (cloud stalled or gone?)"
        )

    # ------------------------------------------------------------------ #
    # Engine interface overrides                                         #
    # ------------------------------------------------------------------ #
    def add_request(self, request: Request, _request_wave: int = 0) -> None:
        """Edge admission (validation + model-family metadata, scheduler-local
        ledger)
        -> cloud request notify (communication stays in the engine)
        -> native engine entry (native validation + enqueue +
        abort_immediately handled natively)."""
        lwd_logger.info(
            "[edge] request received req=%s tokens=%d (admit + announce)",
            request.request_id,
            request.num_prompt_tokens,
        )
        self.scheduler.lwd_admit_request(request)
        self._lwd_notify_request(request)
        super().add_request(request)

    def abort_requests(self, request_ids: list[str]) -> None:
        """Pending-ledger removal (leaving entries in would push the
        prefill gate up); finishing goes through native; the cloud
        abort-notify feature is not landed yet and will be completed
        together with that feature."""
        self.scheduler.lwd_edge_abort(request_ids)
        super().abort_requests(request_ids)

    def _lwd_notify_request(self, request: Request) -> None:
        """Announce the request to the cloud, packing the sampling fields from
        the shared table and the model family's whole-request metadata."""
        sampling_params = request.sampling_params
        assert sampling_params is not None, "admission rejects pooling requests"
        values: dict[str, Any] = {
            f: getattr(sampling_params, f) for f in LWD_SAMPLING_DEFAULTS
        }
        if values["max_tokens"] is None:
            values["max_tokens"] = LWD_SAMPLING_DEFAULTS["max_tokens"]
        values["stop_token_ids"] = list(values["stop_token_ids"] or [])
        metadata = self.scheduler.lwd_model_family.request_metadata(request)
        self._lwd_send_to_cloud(
            LwdRequestNotify(
                request_id=request.request_id,
                num_prompt_tokens=request.num_prompt_tokens,
                block_hashes=list(request.block_hashes),
                eos_token_id=sampling_params.eos_token_id,
                engram_tail_compressed_ids=metadata.engram_tail_compressed_ids,
                engram_tail_dead=metadata.engram_tail_dead,
                image_placeholder_spans=metadata.image_placeholder_spans,
                **values,
            )
        )
