# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Edge engine core: one control plane per cloud DP rank, the check of the
clouds' scheduling limits and the request entry (edge admission +
announcement to the owning cloud; communication stays in the engine).

The edge is a single process whatever the cloud DP count is, so it holds one
link per cloud DP rather than being replicated: each link is its own
`LwdEdgeControlPlane` (its own ZMQ context, IO thread and DEALER socket), and
the scheduler routes every request to one of them for its whole life.

Stepping is the native step_with_batch_queue, adapted at construction: depth
scales with the cloud DP count, is_pooling_model=True (no native
sample_tokens) and an executor proxy (a normally completed batch returning
None becomes a placeholder).

Startup blocks until every cloud accepts the handshake and sends its
scheduling limits, within one handshake timeout; any failure stops the
engine. There is no self-healing: restart the whole group.
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

# Native batch queue depth per cloud DP: dispatched (embed + unembed)
# batches not yet reaped. 2 keeps one link's pipeline aligned; the reference
# branch's 4 is conservative, validate performance with 2 first. The depth
# scales with the DP count so that N links can each hold a batch in flight
# instead of sharing two slots.
LWD_EDGE_BATCH_QUEUE_DEPTH_PER_DP = 2

# A full outbound queue is retried with growing intervals (about 3 s in
# total) before failing: by then the cloud stopped reading or is gone.
_LWD_SEND_RETRY_STEPS = 5
_LWD_SEND_RETRY_INTERVAL_S = 0.2


class LwdEdgeEngineCore(LwdBaseEngineCore):
    """Edge PO engine: control plane + cloud limits check + native stepping."""

    lwd_scheduler_cls = LwdEdgeScheduler
    scheduler: LwdEdgeScheduler

    def _lwd_assemble(self) -> None:
        # One entry per cloud DP, indexed by its dp_idx: the params notify
        # of that cloud, the event that marks its arrival, and the peer the
        # handshake resolved.
        dp_size = self.lwd_config.cloud_dp_size
        self._lwd_cloud_params: list[LwdCloudParamsNotify | None] = [None] * dp_size
        self._lwd_cloud_params_received = [threading.Event() for _ in range(dp_size)]
        self._lwd_clouds: list[LwdPeer | None] = [None] * dp_size
        planes = self._lwd_start_control_planes()
        # One handshake_timeout_s covers the accept and the limits of every
        # link: they are established concurrently (each plane has its own IO
        # thread), so the budget bounds the wait for the whole fleet to be
        # ready rather than any single cloud.
        handshake_timeout_s = planes[0].spec.config.handshake_timeout_s
        deadline = time.monotonic() + handshake_timeout_s
        for dp_idx, plane in enumerate(planes):
            remaining = max(0.0, deadline - time.monotonic())
            self._lwd_clouds[dp_idx] = plane.wait_until_ready(remaining)
            lwd_logger.info(
                "[edge] cloud dp=%d registered: %s at %s",
                dp_idx,
                self._lwd_clouds[dp_idx],
                plane.endpoint,
            )
        for dp_idx, plane in enumerate(planes):
            if not self._lwd_cloud_params_received[dp_idx].wait(
                max(0.0, deadline - time.monotonic())
            ):
                raise RuntimeError(
                    f"[LWD] edge engine init failed: cloud dp{dp_idx} "
                    f"{self._lwd_clouds[dp_idx]} at {plane.endpoint} accepted "
                    f"the handshake but sent no scheduling limits within "
                    f"{handshake_timeout_s}s"
                )
        self._lwd_check_cloud_params()
        self.scheduler.lwd_send_to_cloud = self._lwd_send_to_cloud
        # Native batch queue stepping (step/step_with_batch_queue untouched):
        # 1. depth scaled by the cloud DP count, rebuilding the deque so its
        #    maxlen matches and appendleft on a full queue cannot silently
        #    evict a batch in flight;
        # 2. the edge samples inside execute, so is_pooling_model stops the
        #    native sample_tokens call;
        # 3. the executor proxy.
        self.is_pooling_model = True
        self.batch_queue_size = LWD_EDGE_BATCH_QUEUE_DEPTH_PER_DP * dp_size
        self.batch_queue = deque(maxlen=self.batch_queue_size)
        self.step_fn = self.step_with_batch_queue
        self.model_executor = LwdEdgeExecutorProxy(self.model_executor)
        lwd_logger.info(
            "[edge] engine assembled: %d cloud DP link(s) %s, native "
            "batch-queue stepping depth=%d",
            dp_size,
            [str(peer) for peer in self._lwd_clouds],
            self.batch_queue_size,
        )

    def _lwd_decode(self, payload: bytes) -> LwdMessage:
        return lwd_decode_cloud_to_edge(payload)

    def _lwd_dispatch_message(self, peer: LwdPeer, msg: LwdMessage) -> bool:
        """Cloud limits -> wake the startup wait; c2e -> scheduler notify
        queue + WAKEUP; anything else is unknown.

        `peer.dp_rank` names the cloud DP that sent it: every cloud DP keeps
        its own `down_seqno` counter and its own DOWN channel, so a notify
        that lost its origin could not be matched to either.
        """
        dp_idx = peer.dp_rank
        if isinstance(msg, LwdCloudParamsNotify):
            self._lwd_cloud_params[dp_idx] = msg
            self._lwd_cloud_params_received[dp_idx].set()
            return True
        if isinstance(msg, LwdC2eNotify):
            lwd_logger.info(
                "[edge] c2e received dp=%d down_seqno=%d reqs=%d hidden_numel=%d",
                dp_idx,
                msg.down_seqno,
                len(msg.req_ids),
                msg.hidden_num_elements,
            )
            self.scheduler.unembed_notify_queue.append((dp_idx, msg))
            self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))
            return True
        return False

    def _lwd_check_cloud_params(self) -> None:
        """(max_num_batched_tokens, max_num_seqs) must match every cloud DP:
        both feed the cap/preflight formulas, and a mismatch makes the cloud
        hold prefills behind decodes forever without any error. The
        speculative factor and drafting slots are adopted from the cloud and
        liveness re-checked with them.

        The DPs are also checked against each other: the edge keeps one set
        of budget terms for the whole fleet, so DPs started with different
        limits cannot be represented and must fail here rather than silently
        taking whichever notify arrived last.
        """
        sched = self.vllm_config.scheduler_config
        local = (sched.max_num_batched_tokens, sched.max_num_seqs)
        first = self._lwd_cloud_params[0]
        assert first is not None
        for dp_idx, params in enumerate(self._lwd_cloud_params):
            assert params is not None
            remote = (params.max_num_batched_tokens, params.max_num_seqs)
            if local != remote:
                raise RuntimeError(
                    f"[LWD] edge/cloud scheduling config mismatch: "
                    f"(max_num_batched_tokens, max_num_seqs) edge={local} != "
                    f"cloud dp{dp_idx}={remote}; a mismatch makes the cloud "
                    f"hold prefills behind decodes forever, so start both "
                    f"sides with the same values"
                )
            if params != first:
                raise RuntimeError(
                    f"[LWD] cloud DPs disagree on their scheduling limits: "
                    f"dp0={first} != dp{dp_idx}={params}; the edge keeps one "
                    f"set of budget terms for every link, so start every "
                    f"cloud DP with the same configuration"
                )
        self._lwd_adopt_cloud_spec(first)
        lwd_logger.info(
            "[cloud-params] scheduling limits verified across %d cloud DP(s): %s",
            len(self._lwd_cloud_params),
            local,
        )
        self._lwd_check_engram_fingerprint(first.engram_fingerprint)

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

    def _lwd_send_to_cloud(
        self, dp_idx: int, message: LwdEdgeToCloudMessage
    ) -> None:
        """Send to one cloud DP, retrying a full outbound queue; raises once
        the retries run out instead of losing the message. Each link has its
        own queue, so a stalled cloud DP cannot block the others."""
        cloud = self._lwd_clouds[dp_idx]
        assert cloud is not None, f"cloud dp{dp_idx} not registered"
        for attempt in range(_LWD_SEND_RETRY_STEPS):
            if self._lwd_send(cloud, message, link=dp_idx):
                return
            time.sleep(_LWD_SEND_RETRY_INTERVAL_S * (attempt + 1))
        raise RuntimeError(
            f"[LWD] {type(message).__name__} to cloud dp{dp_idx} {cloud} not "
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
        """Announce the request to its routed cloud DP, packing the sampling
        fields from the shared table and the model family's whole-request
        metadata. The scheduler pinned the route at admission; every later
        message about this request takes the same link."""
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
            self.scheduler.lwd_request_dp(request.request_id),
            LwdRequestNotify(
                request_id=request.request_id,
                num_prompt_tokens=request.num_prompt_tokens,
                block_hashes=list(request.block_hashes),
                eos_token_id=sampling_params.eos_token_id,
                image_placeholder_spans=metadata.image_placeholder_spans,
                **values,
            )
        )
