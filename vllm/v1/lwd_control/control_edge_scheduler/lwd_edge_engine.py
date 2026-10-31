# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Edge engine core: the control plane to every cloud DP rank, the check of
each cloud's scheduling limits and the request entry (edge admission +
announcement to the routed cloud DP rank; communication stays in the engine).

Stepping is the native step_with_batch_queue, adapted at construction: depth
2, is_pooling_model=True (no native sample_tokens) and an executor proxy (a
normally completed batch returning None becomes a placeholder).

Startup blocks until every cloud DP rank accepted the handshake and sent its
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
from vllm.v1.lwd_control.control_edge_scheduler.lwd_coord_client import (
    LwdCoordClient,
)
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
    """Edge PO engine: control plane to every cloud DP rank, cloud limits
    check and native stepping."""

    lwd_scheduler_cls = LwdEdgeScheduler
    scheduler: LwdEdgeScheduler

    def _lwd_assemble(self) -> None:
        self._lwd_cloud_params: dict[int, LwdCloudParamsNotify] = {}
        self._lwd_cloud_params_received = threading.Event()
        plane = self._lwd_start_control_plane()
        assert isinstance(plane, LwdEdgeControlPlane)
        # One handshake_timeout_s covers every link: accept and limits for
        # all M cloud DP ranks, bounding the wait for a full cloud.
        handshake_timeout_s = plane.spec.config.handshake_timeout_s
        deadline = time.monotonic() + handshake_timeout_s
        self._lwd_clouds: dict[int, LwdPeer] = plane.wait_until_ready(
            handshake_timeout_s
        )
        expected = len(self._lwd_clouds)
        while len(self._lwd_cloud_params) < expected:
            if not self._lwd_cloud_params_received.wait(
                max(0.0, deadline - time.monotonic())
            ):
                missing = expected - len(self._lwd_cloud_params)
                raise RuntimeError(
                    f"[LWD] edge engine init failed: {missing} of "
                    f"{expected} cloud DP ranks accepted the handshake "
                    f"but sent no scheduling limits within "
                    f"{handshake_timeout_s}s"
                )
            self._lwd_cloud_params_received.clear()
        for dp_idx, params in self._lwd_cloud_params.items():
            self._lwd_check_cloud_params(dp_idx, params)
        # A MoE multi-DP cloud pauses its engines between waves; the edge
        # is its front-end stand-in, so it wakes the wave of the routed
        # engine on the first request. Links without a coordinator (single
        # DP, non-MoE) stay without a client.
        self._lwd_coord_clients: dict[int, LwdCoordClient] = {
            dp_idx: LwdCoordClient(dp_idx, params.dp_coordinator_addr)
            for dp_idx, params in self._lwd_cloud_params.items()
            if params.dp_coordinator_addr
        }
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
            "[edge] engine assembled: %d cloud DP ranks, native batch-queue "
            "stepping depth=%d",
            len(self._lwd_clouds),
            self.batch_queue_size,
        )

    def shutdown(self) -> None:
        # Reached after a failed assembly too, before the clients exist.
        for client in getattr(self, "_lwd_coord_clients", {}).values():
            client.shutdown()
        super().shutdown()

    def _lwd_decode(self, payload: bytes) -> LwdMessage:
        return lwd_decode_cloud_to_edge(payload)

    def _lwd_dispatch_message(self, peer: LwdPeer, msg: LwdMessage) -> bool:
        """Cloud limits -> record and wake the startup wait; c2e -> scheduler
        notify queue + WAKEUP; anything else is unknown."""
        if isinstance(msg, LwdCloudParamsNotify):
            self._lwd_cloud_params[peer.dp_rank] = msg
            self._lwd_cloud_params_received.set()
            return True
        if isinstance(msg, LwdC2eNotify):
            lwd_logger.info(
                "[edge] c2e received from dp%d reqs=%d hidden_numel=%d",
                peer.dp_rank,
                len(msg.req_ids),
                msg.hidden_num_elements,
            )
            self.scheduler.unembed_notify_queue.append((peer.dp_rank, msg))
            self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))
            return True
        return False

    def _lwd_check_cloud_params(
        self, dp_idx: int, params: LwdCloudParamsNotify
    ) -> None:
        """(max_num_batched_tokens, max_num_seqs) must match every cloud DP
        rank: a mismatch makes the cloud hold prefills behind decodes forever
        without any error."""
        sched = self.vllm_config.scheduler_config
        local = (sched.max_num_batched_tokens, sched.max_num_seqs)
        remote = (params.max_num_batched_tokens, params.max_num_seqs)
        if local != remote:
            raise RuntimeError(
                f"[LWD] edge/cloud scheduling config mismatch on dp{dp_idx}: "
                f"(max_num_batched_tokens, max_num_seqs) edge={local} != "
                f"cloud={remote}; a mismatch makes the cloud hold prefills "
                "behind decodes forever, so start both sides with the same values"
            )
        lwd_logger.info("[cloud-params] dp%d limits verified: %s", dp_idx, local)

    def _lwd_send_to_cloud(self, dp_idx: int, message: LwdEdgeToCloudMessage) -> None:
        """Send to one cloud DP rank, retrying a full outbound queue; raises
        once the retries run out instead of losing the message."""
        cloud = self._lwd_clouds[dp_idx]
        for attempt in range(_LWD_SEND_RETRY_STEPS):
            if self._lwd_send(cloud, message):
                return
            time.sleep(_LWD_SEND_RETRY_INTERVAL_S * (attempt + 1))
        raise RuntimeError(
            f"[LWD] {type(message).__name__} to cloud dp{dp_idx} {cloud} not "
            f"sent: outbound queue still full after {_LWD_SEND_RETRY_STEPS} "
            "retries (cloud stalled or gone?)"
        )

    # ------------------------------------------------------------------ #
    # Engine interface overrides                                         #
    # ------------------------------------------------------------------ #
    def add_request(self, request: Request, _request_wave: int = 0) -> None:
        """Edge admission (validation + mrope, scheduler-local ledger and
        routing) -> cloud request notify of the routed DP rank
        (communication stays in the engine) -> native engine entry (native
        validation + enqueue + abort_immediately handled natively)."""
        lwd_logger.info(
            "[edge] request received req=%s tokens=%d (admit + announce)",
            request.request_id,
            request.num_prompt_tokens,
        )
        dp_idx = self.scheduler.lwd_admit_request(request)
        client = self._lwd_coord_clients.get(dp_idx)
        if client is not None:
            client.send_first_req()
        self._lwd_notify_request(dp_idx, request)
        super().add_request(request)

    def abort_requests(self, request_ids: list[str]) -> None:
        """Settle local scheduler state, then finish natively."""
        self.scheduler.lwd_edge_abort(request_ids)
        super().abort_requests(request_ids)

    def _lwd_notify_request(self, dp_idx: int, request: Request) -> None:
        """Announce the request to one cloud DP rank, packing the sampling
        fields from the shared table."""
        sampling_params = request.sampling_params
        assert sampling_params is not None, "admission rejects pooling requests"
        values: dict[str, Any] = {
            f: getattr(sampling_params, f) for f in LWD_SAMPLING_DEFAULTS
        }
        if values["max_tokens"] is None:
            values["max_tokens"] = LWD_SAMPLING_DEFAULTS["max_tokens"]
        values["stop_token_ids"] = list(values["stop_token_ids"] or [])
        self._lwd_send_to_cloud(
            dp_idx,
            LwdRequestNotify(
                request_id=request.request_id,
                num_prompt_tokens=request.num_prompt_tokens,
                block_hashes=list(request.block_hashes),
                eos_token_id=sampling_params.eos_token_id,
                mrope_positions=self.scheduler.lwd_mrope_positions(request.request_id),
                **values,
            ),
        )
