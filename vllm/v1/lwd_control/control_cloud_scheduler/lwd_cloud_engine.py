# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cloud engine cores for one edge connection per DP rank.

Range notifies post extended early-receive hints, request notifies become
native ADDs with prompt-length and M-RoPE metadata, and abort notifies use the
native ABORT path. Step metadata is flushed to the edge in post_step.

The LWD seams live in LwdCloudEngineCoreMixin and are orthogonal to the
engine base: LwdCloudEngineCore (EngineCoreProc) serves one DP rank of a
cloud instance and non-MoE multi-DP deployments; LwdCloudDPEngineCore
(DPEngineCoreProc) serves MoE multi-DP deployments, inheriting the native
DP semantics (wave state machine, dummy batches, START_DP_WAVE, stats
publishing) unchanged.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterable
from typing import Any

import torch

from vllm.lwd_logger import lwd_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc, EngineShutdownState
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_scheduler import (
    LwdCloudScheduler,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_engine import (
    LwdBaseEngineCore,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    lwd_spec_factor,
)
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import LwdPeer
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LWD_SAMPLING_DEFAULTS,
    LwdAbortNotify,
    LwdCloudParamsNotify,
    LwdCloudToEdgeMessage,
    LwdEdgeToCloudMessage,
    LwdMessage,
    LwdRangeNotify,
    LwdRequestNotify,
    lwd_decode_edge_to_cloud,
)
from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag
from vllm.v1.request import Request

# Sleep between retries of a full outbound queue: step metadata must not be
# lost, the edge posts its receive of the exact size from it.
_LWD_SEND_RETRY_SLEEP_S = 0.05
_LWD_INGEST_JOIN_TIMEOUT_S = 5.0


_LwdIngestItem = tuple[LwdPeer, LwdEdgeToCloudMessage]


class LwdCloudEngineCoreMixin(LwdBaseEngineCore):
    """The LWD seams of a cloud engine, shared by both engine bases."""

    lwd_scheduler_cls = LwdCloudScheduler
    scheduler: LwdCloudScheduler

    def __init__(self, *args: Any, lwd_dp_size: int, **kwargs: Any) -> None:
        vllm_config = kwargs["vllm_config"]
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("[LWD] cloud prefix caching is unsupported")
        # Pre-recv hint MQ must exist before super().__init__ spawns the
        # executor workers (same process): create_hint_mq() publishes the
        # handle through the environment, and only processes spawned after
        # that see it. The sideband MQ exists because rpc_broadcast_mq is
        # drained by the worker's single-threaded busy loop, which sits
        # inside a several-hundred-ms prefill forward - hints queued behind
        # it would lose the whole point of pre-posting the receive.
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import create_hint_mq

        self._lwd_hint_mq = create_hint_mq()
        super().__init__(*args, lwd_dp_size=lwd_dp_size, **kwargs)
        # The cloud endpoint worker created its done-report MQ during
        # init_device; now that the executor is up, fetch its handle over
        # the executor's collective RPC (the executor itself stays free of
        # LWD plumbing). Exactly one worker - the endpoint rank - must
        # report a handle; anything else is a wiring error.
        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

        handles = [
            handle
            for handle in self.model_executor.collective_rpc(
                "lwd_early_recv_done_handle"
            )
            if handle is not None
        ]
        if len(handles) != 1:
            raise RuntimeError(
                "[LWD] early receive expects one endpoint worker reporting "
                f"completions, got {len(handles)}"
            )
        self._lwd_done_mq = MessageQueue.create_from_handle(handles[0], 0)
        lwd_logger.info("[pre-recv] hint MQ created, done MQ attached via RPC")

    def _lwd_assemble(self) -> None:
        self._lwd_ingest_thread: threading.Thread | None = None
        self._lwd_edge: LwdPeer | None = None
        self._lwd_prompt_lens: dict[str, int] = {}
        self._lwd_retired_prompt_lens: dict[str, int] = {}
        self._lwd_retired_req_ids: set[str] = set()
        self._lwd_pending_seqnos: dict[int, set[str]] = {}
        self._lwd_request_seqnos: dict[str, set[int]] = {}
        self._lwd_latest_seqno = -1
        sched_config = self.vllm_config.scheduler_config
        # Scheduling limits are checked by the edge. The coordinator address
        # lets the edge
        # trigger the first DP wave; its published host may not be routable
        # from the edge, so it is rewritten to this DP's topology address.
        self._lwd_cloud_params = LwdCloudParamsNotify(
            max_num_batched_tokens=sched_config.max_num_batched_tokens,
            max_num_seqs=sched_config.max_num_seqs,
            num_speculative_tokens=lwd_spec_factor(self.vllm_config) - 1,
            dp_coordinator_addr=self._lwd_dp_coordinator_addr(),
        )
        self._lwd_register_recv_hint()
        # Edge messages are handled off the IO thread: building a request
        # allocates its placeholder embeds and the pre-recv hint may block,
        # which must not delay step metadata going out.
        self._lwd_ingest_queue: queue.SimpleQueue[_LwdIngestItem | None] = (
            queue.SimpleQueue()
        )
        self._lwd_ingest_thread = threading.Thread(
            target=self._lwd_ingest_loop, name="lwd-cloud-ingest", daemon=True
        )
        self._lwd_ingest_thread.start()
        plane = self._lwd_start_control_plane()
        lwd_logger.info("[cloud] engine assembled: control plane on %s", plane.endpoint)

    def _lwd_dp_coordinator_addr(self) -> str:
        """Routable coordinator stats endpoint, or empty without one."""
        address = getattr(self, "frontend_stats_publish_address", None)
        if not self.has_coordinator or not address:
            return ""
        if not str(address).startswith("tcp://"):
            return ""
        _, _, port = str(address).rpartition(":")
        dp_rank = self.vllm_config.parallel_config.data_parallel_index
        local = self.lwd_config.instance.dp[dp_rank]
        return f"tcp://{local.addr}:{port}"

    def _lwd_decode(self, payload: bytes) -> LwdMessage:
        return lwd_decode_edge_to_cloud(payload)

    def _lwd_handle_peer_registered(self, peer: LwdPeer) -> None:
        """Engine thread: remember the edge and send it the scheduling
        limits before anything else."""
        self._lwd_edge = peer
        self._lwd_send_until_queued(peer, self._lwd_cloud_params)
        lwd_logger.info(
            "[cloud] edge %s registered; scheduling limits sent: %s",
            peer,
            self._lwd_cloud_params,
        )

    def _lwd_send_until_queued(
        self, peer: LwdPeer, message: LwdCloudToEdgeMessage
    ) -> None:
        """Retry a full outbound queue until the message is queued; gives up
        with an error once shutdown is requested, so a stalled edge cannot
        block it."""
        while not self._lwd_send(peer, message):
            if self.shutdown_state != EngineShutdownState.RUNNING:
                lwd_logger.error(
                    "[cloud] %s to edge %s dropped: shutting down with the "
                    "outbound queue full",
                    type(message).__name__,
                    peer,
                )
                return
            time.sleep(_LWD_SEND_RETRY_SLEEP_S)

    def _lwd_register_recv_hint(self) -> None:
        """[LWD][pre-recv] Early receive: the hint delivery channel =
        engine -> cloud endpoint worker comm thread. It goes through
        the sideband MQ this engine creates in __init__ (not
        rpc_broadcast_mq: that one is consumed single-threaded by the
        busy_loop and gets blocked by a several-hundred-ms prefill
        forward, so the hint would no longer be "early")."""
        hint_mq = self._lwd_hint_mq
        if hint_mq is not None:
            from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
                register_hint_sender,
            )

            # Blocking enqueue: once the gate is on, hints are a
            # correctness dependency — drop one and nobody pre-posts
            # that recv, the watermark stops advancing, and batches
            # behind the gate are never dispatched.
            register_hint_sender(lambda hint, mq=hint_mq: mq.enqueue(hint))
            lwd_logger.info("[pre-recv] hint sender registered (PD_mix -> hint MQ)")
        else:
            # PD_mix is the only scheduling mode: a missing hint MQ =
            # hints never go out and the gate never opens; this must be
            # explicitly visible
            lwd_logger.warning(
                "[pre-recv] hint MQ missing: pre-recv gate will never open"
            )

    def _lwd_dispatch_message(self, peer: LwdPeer, msg: LwdMessage) -> bool:
        """IO thread: hand all edge business notifies to the ingest thread."""
        if not isinstance(msg, (LwdRangeNotify, LwdRequestNotify, LwdAbortNotify)):
            return False
        self._lwd_ingest_queue.put((peer, msg))
        return True

    def _lwd_ingest_loop(self) -> None:
        while (item := self._lwd_ingest_queue.get()) is not None:
            try:
                self._lwd_handle_edge_message(*item)
            except Exception as e:
                lwd_logger.exception(
                    "[cloud] ingest of %s failed", type(item[1]).__name__
                )
                self._lwd_fail(f"ingest of {type(item[1]).__name__}: {e!r}")
                return

    def _lwd_handle_edge_message(
        self, peer: LwdPeer, msg: LwdEdgeToCloudMessage
    ) -> None:
        lwd_logger.info(
            "[cloud] ingest msg=%s seqno=%s",
            type(msg).__name__,
            getattr(msg, "seqno", "-"),
        )
        # TODO(lwd): several edges per cloud DP rank need requests tracked by
        # the peer that owns them, issue #8.
        if isinstance(msg, LwdRangeNotify):
            self._lwd_handle_range_notify(msg)
        elif isinstance(msg, LwdRequestNotify):
            self._lwd_handle_request_notify(msg)
        elif isinstance(msg, LwdAbortNotify):
            self._lwd_handle_abort_notify(msg)

    def _lwd_handle_range_notify(self, notify: LwdRangeNotify) -> None:
        """Post each new extended hint exactly once, including mixed batches."""
        if notify.seqno <= self._lwd_latest_seqno:
            lwd_logger.info(
                "[cloud] ignore duplicate RangeNotify seqno=%d", notify.seqno
            )
            return
        items = notify.items
        request_ids = [item.request_id for item in items]
        prompt_lens = [
            self._lwd_prompt_lens.get(
                request_id, self._lwd_retired_prompt_lens.get(request_id, 0)
            )
            for request_id in request_ids
        ]
        if any(prompt_len <= 0 for prompt_len in prompt_lens):
            raise RuntimeError(
                f"[LWD] RangeNotify seqno={notify.seqno} arrived before "
                "RequestNotify prompt lengths were recorded"
            )
        self._lwd_track_range(notify.seqno, request_ids)
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
            make_recv_hint,
            post_irecv_hint,
        )

        num_tokens = sum(item.num_tokens for item in items)
        lwd_logger.info(
            "[Lwd][pre-recv] HINT seqno=%d num_tokens=%d ts=%.3f",
            notify.seqno,
            num_tokens,
            time.monotonic(),
        )
        post_irecv_hint(
            make_recv_hint(
                notify.seqno,
                num_tokens,
                0,
                req_ids=request_ids,
                rows=[item.num_tokens for item in items],
                prompt_offsets=[item.offset for item in items],
                prompt_lens=prompt_lens,
            )
        )

    def _lwd_track_range(self, seqno: int, request_ids: list[str]) -> None:
        self._lwd_latest_seqno = seqno
        self._lwd_pending_seqnos[seqno] = set(request_ids)
        for request_id in request_ids:
            self._lwd_request_seqnos.setdefault(request_id, set()).add(seqno)

    def _lwd_complete_seqno(self, seqno: int) -> None:
        request_ids = self._lwd_pending_seqnos.pop(seqno, None)
        if request_ids is None:
            return
        for request_id in request_ids:
            pending = self._lwd_request_seqnos.get(request_id)
            if pending is not None:
                pending.discard(seqno)
            if pending:
                continue
            self._lwd_request_seqnos.pop(request_id, None)
            if request_id in self._lwd_retired_req_ids:
                self._lwd_clear_retired(request_id)

    def _lwd_handle_request_notify(self, notify: LwdRequestNotify) -> None:
        if self._lwd_request_seqnos.get(notify.request_id):
            raise RuntimeError(
                f"[LWD] request_id {notify.request_id} reused while the "
                "previous generation still has pending receives"
            )
        self._lwd_prompt_lens[notify.request_id] = notify.num_prompt_tokens
        request = self._lwd_build_request(notify)
        self.input_queue.put_nowait((EngineCoreRequestType.ADD, (request, 0)))
        lwd_logger.info(
            "[cloud] request added req=%s tokens=%d",
            notify.request_id,
            notify.num_prompt_tokens,
        )
        # The local request proves work exists even if the coordinator's
        # FIRST_REQ wake-up could not reach this DP engine.
        if not self.engines_running:
            self.engines_running = True
            lwd_logger.info(
                "[cloud] first request wakes a paused wave (local fallback)"
            )

    def _lwd_handle_abort_notify(self, notify: LwdAbortNotify) -> None:
        request_id = notify.request_id
        self._lwd_retire_request(request_id, preserve_prompt=True)
        lwd_logger.info("[cloud] abort req=%s", request_id)
        self.aborts_queue.put_nowait([request_id])
        self.input_queue.put_nowait((EngineCoreRequestType.ABORT, [request_id]))

    def _lwd_retire_request(
        self, request_id: str, preserve_prompt: bool = False
    ) -> None:
        self.scheduler.lwd_recv_rows.pop(request_id, None)
        if not self._lwd_request_seqnos.get(request_id):
            self._lwd_clear_retired(request_id)
            return
        self._lwd_retired_req_ids.add(request_id)
        if preserve_prompt:
            prompt_len = self._lwd_prompt_lens.get(request_id)
            if prompt_len is not None:
                self._lwd_retired_prompt_lens[request_id] = prompt_len

    def _lwd_clear_retired(self, request_id: str) -> None:
        self._lwd_retired_req_ids.discard(request_id)
        self._lwd_retired_prompt_lens.pop(request_id, None)
        self._lwd_request_seqnos.pop(request_id, None)
        self._lwd_prompt_lens.pop(request_id, None)

    def shutdown(self) -> None:
        # Before the executor goes: a pre-recv hint blocks on its queue.
        if self._lwd_ingest_thread is not None:
            self._lwd_ingest_queue.put(None)
            self._lwd_ingest_thread.join(_LWD_INGEST_JOIN_TIMEOUT_S)
        super().shutdown()

    def _lwd_build_request(self, req_notify: LwdRequestNotify) -> Request:
        """Request building (the sole request-creation point): sampling
        params rebuilt from the request notify + placeholder meta
        embeds (shape only, no storage; prompt length is the row
        count; the worker rebuilds a local zeroed buffer from is_meta,
        and the UP inject writes straight into that buffer window).
        Prefix caching is rejected before engine startup."""
        sampling_params = self._lwd_build_sampling_params(req_notify)
        # Meta embeds carry only the prompt shape. Full-prompt M-RoPE
        # positions travel separately to scheduled_new_reqs.
        prompt_embeds = torch.empty(
            req_notify.num_prompt_tokens,
            self.vllm_config.model_config.get_hidden_size(),
            dtype=self.vllm_config.model_config.dtype,
            device="meta",
        )
        request = Request(
            request_id=req_notify.request_id,
            prompt_token_ids=None,
            prompt_embeds=prompt_embeds,
            sampling_params=sampling_params,
            pooling_params=None,
        )
        request.lwd_mrope_positions = req_notify.mrope_positions
        return request

    @staticmethod
    def _lwd_build_sampling_params(req_notify: LwdRequestNotify) -> SamplingParams:
        """Sampling params rebuilt from the request notify (field-table
        driven); eos_token_id goes through the native backfill entry
        (sets _eos_token_id and adds it to _all_stop_token_ids for the
        min_tokens check; the cloud has no client generation_config,
        so pass empty)."""
        values = {f: getattr(req_notify, f) for f in LWD_SAMPLING_DEFAULTS}
        values["stop_token_ids"] = list(values["stop_token_ids"])
        sampling_params = SamplingParams(**values)
        sampling_params.update_from_generation_config({}, req_notify.eos_token_id)
        LwdControlDiag.cloud_request_admitted(req_notify, sampling_params)
        return sampling_params

    def step_with_batch_queue(self):
        """Drain the early-recv watermark at the head of the step (done
        MQ -> gate readiness predicate); the rest is native."""
        self._lwd_drain_irecv_completions()
        engine_core_outputs, model_executed = super().step_with_batch_queue()
        self._lwd_retire_finished_outputs(engine_core_outputs)
        return engine_core_outputs, model_executed

    def step(self):
        """Same for the synchronous step path: drain the watermark at
        the head of the step, the rest is native."""
        self._lwd_drain_irecv_completions()
        engine_core_outputs, model_executed = super().step()
        self._lwd_retire_finished_outputs(engine_core_outputs)
        return engine_core_outputs, model_executed

    def _lwd_retire_finished_outputs(self, engine_core_outputs: Any) -> None:
        if not engine_core_outputs:
            return
        for outputs in engine_core_outputs.values():
            for request_id in outputs.finished_requests or ():
                self._lwd_retire_request(request_id)
            for output in outputs.outputs:
                if output.finish_reason is not None:
                    self._lwd_retire_request(output.request_id)

    def _lwd_drain_irecv_completions(self) -> None:
        """Merge done-MQ arrivals into per-request rows and UP watermarks."""
        done_mq = getattr(self, "_lwd_done_mq", None)
        if done_mq is None:
            return
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
            record_irecv_completions,
        )

        items: list[Any] = []
        while True:
            try:
                items.append(done_mq.dequeue(timeout=0))
            except TimeoutError:
                break
            except Exception:
                lwd_logger.exception("[pre-recv] done MQ drain failed")
                break
        watermark_items: list[Any] = []
        for item in items:
            if isinstance(item, tuple) and len(item) == 3 and item[0] == "error":
                self._lwd_fail(f"[LWD] worker receive error seqno={item[1]}: {item[2]}")
                return
            if (
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], tuple)
            ):
                key, arrived = item
                watermark_items.append(key)
                self._lwd_note_arrived(key[1], arrived)
            else:
                watermark_items.append(item)
        if not watermark_items:
            return
        advanced = record_irecv_completions(watermark_items)
        for _channel, seqno in advanced:
            self._lwd_complete_seqno(seqno)
        lwd_logger.info(
            "[Lwd][pre-recv] DRAIN n=%d seqnos=%s advanced=%s ts=%.3f",
            len(watermark_items),
            sorted({seqno for _channel, seqno in watermark_items}),
            sorted({seqno for _channel, seqno in advanced}),
            time.monotonic(),
        )

    def _lwd_note_arrived(
        self, seqno: int, arrived: Iterable[tuple[str, int]] | None
    ) -> None:
        """Merge arrivals only for requests in this still-pending seqno."""
        request_ids = self._lwd_pending_seqnos.get(seqno)
        if not arrived or request_ids is None:
            return
        touched: list[tuple[str, int, int]] = []
        for request_id, rows in arrived:
            try:
                request_id, rows = str(request_id), int(rows)
            except (TypeError, ValueError):
                continue
            if (
                rows <= 0
                or request_id not in request_ids
                or request_id in self._lwd_retired_req_ids
            ):
                continue
            previous = self.scheduler.lwd_recv_rows.get(request_id, 0)
            if rows > previous:
                self.scheduler.lwd_recv_rows[request_id] = rows
                touched.append((request_id, previous, rows))
        if touched:
            lwd_logger.info("[cloud] arrival high watermarks updated: %s", touched)

    def post_step(self, model_executed: bool) -> None:
        """Flush pending c2e to the edge at the end of the step (the
        scheduler only queues them; both native step paths reach this hook)."""
        if self.scheduler.pending_c2e:
            assert self._lwd_edge is not None, "c2e produced before any edge"
            for notify in self.scheduler.pending_c2e:
                self._lwd_send_until_queued(self._lwd_edge, notify)
            self.scheduler.pending_c2e.clear()
        super().post_step(model_executed)


class LwdCloudEngineCore(LwdCloudEngineCoreMixin, EngineCoreProc):
    """Cloud PO engine of one DP rank (dp=1 and non-MoE multi-DP): the
    engine processes of a multi-DP non-MoE cloud are independent, so the
    native EngineCoreProc base is the whole story."""


class LwdCloudDPEngineCore(LwdCloudEngineCoreMixin, DPEngineCoreProc):
    """Cloud PO engine of a MoE multi-DP cloud: the native DP engine base
    carries the wave state machine, dummy batches, START_DP_WAVE handling
    and stats publishing, so the MoE engines stay in lockstep exactly as a
    centralized DP deployment; the LWD seams come from the mixin."""
