# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Edge scheduler: per-cloud-DP routing and gating + chunked prefill batch
assembly + control-plane egress (RangeNotify/seqno).

The edge side only does embedding (no decode); the request lifecycle stays
with the native scheduler throughout: enqueue -> EMBED (chunk batch
assembly / range announcement) -> on embedding completion, transition to
decode phase and stay in running awaiting the cloud result -> the cloud
result terminates the request (finish code mapped to native finish), and
late results are idempotently dropped.

One LwdCloudDpAccount holds every per-engine bookkeeping (seqno stream,
pending batches, load, gate); the routing table is pinned at admission and
never changes, so a request's chunks, aborts and results all travel
through its engine's account. Each schedule step serves one engine: a
RangeNotify, a seqno and a UP channel belong to exactly one engine. With
one cloud DP rank every account method degenerates to the previous
single-cloud behaviour.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from vllm.lwd_logger import lwd_logger
from vllm.v1.core.sched.output import (
    LwdBatch,
    LwdBatchType,
    LwdEmbedBatch,
    LwdUnembedBatch,
    SchedulerOutput,
)
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    FinishReason,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_cloud_dp_account import (
    LWD_MAX_PENDING_CHUNKS,
    LwdCloudDpAccount,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_picker import (
    LwdPrefillCandidate,
    lwd_pick_prefill_batch,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    LwdBaseScheduler,
    LwdReqPhase,
)
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LWD_NOT_FINISHED,
    LwdC2eNotify,
    LwdEdgeToCloudMessage,
    LwdRangeItem,
    LwdRangeNotify,
)
from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.request import Request

LwdSendToCloud = Callable[[int, LwdEdgeToCloudMessage], None]
"""Sends one message to the cloud DP rank of dp_idx; raises instead of
losing it."""

# Cloud finish code -> edge terminal state (frontend reason goes through the
# output channel; ERROR maps to ABORTED)
_LWD_FINISH_STATUS = {
    FinishReason.STOP: RequestStatus.FINISHED_STOPPED,
    FinishReason.LENGTH: RequestStatus.FINISHED_LENGTH_CAPPED,
    FinishReason.ABORT: RequestStatus.FINISHED_ABORTED,
    FinishReason.ERROR: RequestStatus.FINISHED_ABORTED,
}


class LwdEdgeScheduler(LwdBaseScheduler):
    """Per-cloud-DP routing/gating + chunked prefill scheduling +
    control-plane egress (RangeNotify/seqno)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Injected by the engine once every cloud accepted the handshake.
        self.lwd_send_to_cloud: LwdSendToCloud | None = None
        # One account per cloud DP rank; the count comes from the routing
        # entry of any DP of the local (edge) instance, all of which share
        # the same connections.
        lwd_config = self.vllm_config.lwd_config
        assert lwd_config is not None
        self._lwd_accounts: tuple[LwdCloudDpAccount, ...] = tuple(
            LwdCloudDpAccount(dp_idx) for dp_idx in range(len(lwd_config.connections))
        )
        self._lwd_route: dict[str, int] = {}
        """Request id -> dp_idx, pinned at admission, never changes."""
        self._lwd_last_engine = -1
        """Round-robin anchor shared by routing and engine picking."""
        # Prefix caching: manager-level hits must stay off (the edge holds
        # no KV cache -- a hit would skip embedding blocks never written);
        # the config-level flag stays on so the engine still builds the
        # block hash chain for cloud-side hits.
        self.kv_cache_manager.enable_caching = False
        # c2e notify queue: appended by the control-plane IO thread with the
        # delivering link's dp_idx, consumed within a scheduling step to
        # produce an UNEMBED batch
        self.unembed_notify_queue: deque[tuple[int, LwdC2eNotify]] = deque()
        # Whole-prompt mrope positions for multimodal requests ([3,N] CPU
        # tensor, computed once at admission and sent in RequestNotify);
        # removed on embedding completion/abort. Key present = has_mrope is
        # always true for every chunk of that request.
        self._lwd_mrope_positions_dict: dict[str, torch.Tensor] = {}
        # Lazily resolved model-class static method
        # _get_mrope_input_positions (cache the method itself rather than
        # the model class so the semantics stay direct; validated on first
        # use, unsupported families fail fast).
        self._lwd_mrope_positions_fn = None
        # Chunk unit C = min(remaining, C): taken from the local
        # long_prefill_token_threshold (native truncation boundary,
        # naturally reproduced on both sides); when unset (<=0), falls back
        # to the aligned budget max_num_scheduled_tokens. The cloud does not
        # read this value; it overwrites per the in-batch max(items)
        threshold = self.scheduler_config.long_prefill_token_threshold
        self._lwd_chunk_unit: int = (
            threshold if threshold > 0 else self.max_num_scheduled_tokens
        )

    # ------------------------------------------------------------- routing

    def _lwd_dp_of(self, request_id: str) -> int:
        """Routed dp_idx of an active request; unrouted ids are a bug."""
        return self._lwd_route[request_id]

    def _lwd_dp_of_settling(self, request_id: str) -> int | None:
        """Routed dp_idx, None for a request that already left the books
        (a late result racing its own finish)."""
        return self._lwd_route.get(request_id)

    def _lwd_route_request(self, request_id: str) -> int:
        """Pick the target engine and pin the route; the score weighs
        prefill 4 and decode 1, the same weights as the native DP load
        balancer, and ties rotate from the last served engine."""
        accounts = self._lwd_accounts
        if len(accounts) == 1:
            self._lwd_route[request_id] = 0
            accounts[0].note_prefill_admitted(request_id)
            return 0
        best = min(
            accounts,
            key=lambda account: (
                account.score(),
                (account.dp_idx - self._lwd_last_engine) % len(accounts),
            ),
        )
        self._lwd_route[request_id] = best.dp_idx
        best.note_prefill_admitted(request_id)
        lwd_logger.info(
            "[edge-route] req=%s -> dp%d (score=%d loads=%s)",
            request_id,
            best.dp_idx,
            best.score(),
            [(a.prefill_load, a.decode_load) for a in accounts],
        )
        return best.dp_idx

    def _lwd_account_of(self, request_id: str) -> LwdCloudDpAccount | None:
        dp_idx = self._lwd_dp_of_settling(request_id)
        if dp_idx is None:
            return None
        return self._lwd_accounts[dp_idx]

    # ------------------------------------------------------------- stepping

    def has_requests(self) -> bool:
        """Keep stepping while any UP send still needs watermark settlement.

        A full pending gate stops new EMBED batches, but the empty UNEMBED
        step still executes the worker and harvests completed sends. Without
        this predicate the edge can sleep forever with pending sends that no
        later batch can clear.
        """
        return (
            bool(self.unembed_notify_queue)
            or any(account.pending_chunks() for account in self._lwd_accounts)
            or self._lwd_embed_dispatchable()
        )

    def _lwd_schedule_step(self) -> SchedulerOutput:
        """Dispatch an EMBED batch first; otherwise consume c2e to produce an
        UNEMBED batch (finish in-flight work before admitting new work, to
        keep EMBED from being squeezed out of the depth quota by the decode
        return trip)."""
        LwdControlDiag.edge_step_activity(
            len(self.running),
            sum(1 for _ in self._lwd_reqs_in_phase(LwdReqPhase.DECODE, self.running)),
        )
        out = self._lwd_schedule_embed()
        if out is not None:
            return out
        return self._lwd_schedule_unembed()

    def _lwd_pick_serving_engine(self) -> int | None:
        """The engine this step serves: one that has routed work and passes
        its gate, rotating from the last served engine so a busy engine
        never starves the others. None when every engine with work is
        gated or there is no work at all."""
        routed = {
            dp_idx
            for dp_idx in self._lwd_route.values()
            if dp_idx < len(self._lwd_accounts)
        }
        if not routed:
            return None
        num_engines = len(self._lwd_accounts)
        for offset in range(num_engines):
            dp_idx = (self._lwd_last_engine + 1 + offset) % num_engines
            if dp_idx not in routed:
                continue
            account = self._lwd_accounts[dp_idx]
            if account.is_gated(
                self.max_num_running_reqs,
                self._lwd_has_prefill_chunk_inflight(dp_idx),
            ):
                self._lwd_log_gate_transition(account)
                continue
            self._lwd_last_engine = dp_idx
            return dp_idx
        return None

    def _lwd_log_gate_transition(self, account: LwdCloudDpAccount) -> None:
        held = account.pending_chunks() >= LWD_MAX_PENDING_CHUNKS
        if held != account.gate_held:
            account.gate_held = held
            lwd_logger.info(
                "[edge-sched] prefill gate %s: dp%d pending_chunks=%d",
                "HOLD" if held else "OPEN",
                account.dp_idx,
                account.pending_chunks(),
            )

    def _lwd_embed_dispatchable(self) -> bool:
        """EMBED dispatchability: some engine has routed work and passes its
        gate."""
        return self._lwd_pick_serving_engine() is not None

    def _lwd_schedule_embed(self) -> SchedulerOutput | None:
        """Chunked batch + budget clamp + RangeNotify of one engine, then
        the EMBED batch (seqno bound to the notify). None when nothing is
        dispatchable.

        The batch is never dropped once scheduled: scheduling already
        advanced num_computed_tokens, so a dropped chunk would never be
        embedded and the next offset would not match the cloud. A notify
        that cannot be sent raises instead."""
        dp_idx = self._lwd_pick_serving_engine()
        if dp_idx is None:
            return None
        out = self._lwd_schedule_batch(dp_idx)
        if not out.num_scheduled_tokens:
            return None
        self.lwd_edge_notify_range(dp_idx, out)
        return out

    def _lwd_schedule_unembed(self) -> SchedulerOutput:
        """Consume a c2e notify to produce an UNEMBED batch."""
        if self.unembed_notify_queue:
            dp_idx, notify = self.unembed_notify_queue.popleft()
            return lwd_build_unembed_batch(dp_idx, notify)
        return SchedulerOutput.make_empty()

    def update_from_output(self, scheduler_output, model_runner_output):
        """LWD harvest (the return value is the engine step output; none of
        it goes through native bookkeeping): EMBED batches register
        execution progress (finished ones stay in running and transition to
        decode phase); UNEMBED batches deliver per request via the
        worker-filled req_ids x sampled_token_ids; placeholder batches
        without lwd_batch (the underlying layer returns None on normal
        completion, replaced by the engine proxy with a placeholder) only
        consume send watermarks."""
        watermarks = getattr(model_runner_output, "lwd_up_send_watermarks", {})
        for dp_idx, watermark in watermarks.items():
            self._lwd_accounts[dp_idx].note_send_watermark(watermark)
        batch = getattr(scheduler_output, "lwd_batch", None)
        outputs: list[EngineCoreOutput] = []
        finished: set[str] = set()
        if batch is not None and batch.batch_type == LwdBatchType.LWD_UNEMBED:
            notify = scheduler_output.lwd_c2e_notify[0]
            req_ids = getattr(model_runner_output, "req_ids", None) or ()
            sampled = (
                dict(zip(req_ids, model_runner_output.sampled_token_ids))
                if req_ids
                else {}
            )
            outputs, finished = self._lwd_deliver_unembed(notify, sampled)
        # EMBED batches have no local accounting (progress was already
        # advanced optimistically during scheduling; on embedding completion
        # they naturally transition to decode phase); the harvest phase
        # uniformly reconciles and cleans up mrope positions (under resident
        # models requests only leave `requests` upon termination, so the
        # explicit per-point removal is replaced by reconciliation-style
        # reclaim)
        self._lwd_purge_mrope_positions()
        if not outputs:
            return {}
        return {
            0: EngineCoreOutputs(
                outputs=outputs,
                finished_requests=finished or None,
            )
        }

    @staticmethod
    def _lwd_row_disposition(accepted: int, has_tokens: bool, finished: bool) -> str:
        """Per-row delivery disposition (pad-and-skip, ported from the mm
        branch 7ff2ddc3e):

        - "skip": zero-accepted live row (accepted=0 and not finished) -- in
          a mixed step, prefill requests occupy only reserved slots in the
          sampling layout (Ascend MTP sampling only fills rows the draft
          participated in), so the collector computes accepted=0 via
          (sampled != -1).sum(). Such rows are protocol artifacts of the
          mixed step; they are kept in the notify only for DOWN tensor
          alignment: no delivery, no ERROR, no first-token marking (before
          the fix this form used to kill requests as ERROR);
        - "error": declares accepted>0 yet has no token = a real unembed
          failure; ERROR takes precedence over the cloud finish code
          (original semantics preserved);
        - "deliver": everything else (including zero-accepted closing rows
          that carry a finish code -- the termination signal must be passed
          through; swallowing it would lose the completion)."""
        if accepted == 0 and not finished:
            return "skip"
        if not has_tokens and not finished:
            return "error"
        return "deliver"

    def _lwd_deliver_unembed(
        self, notify: LwdC2eNotify, sampled_token_map: dict[str, list[int]]
    ) -> tuple[list[EngineCoreOutput], set[str]]:
        """Per-request delivery: row disposition goes through
        _lwd_row_disposition (zero-accepted live rows are skipped; ERROR
        only when accepted>0 yet token missing); late results not in
        awaiting are idempotently dropped."""
        outputs: list[EngineCoreOutput] = []
        finished: set[str] = set()
        # accepted is position-aligned with rows (same contract as
        # lwd_build_unembed_batch); the DISABLE_DOWN bypass carries no
        # accepted, so token presence/absence is used instead
        accepts = notify.num_accepted_tokens
        for index, request_id in enumerate(notify.req_ids):
            finish_reason = self._lwd_finish_code(notify, index)
            finished_now = finish_reason is not None
            sampled_token_ids = list(sampled_token_map.get(request_id, []))
            accepted = accepts[index] if accepts else (1 if sampled_token_ids else 0)
            disposition = self._lwd_row_disposition(
                accepted, bool(sampled_token_ids), finished_now
            )
            if disposition == "skip":
                # Zero-accepted live row: no delivery, no empty output, and
                # no needless debug-tax logging
                continue
            LwdControlDiag.edge_tokens_delivered(
                request_id,
                sampled_token_ids,
                finish_reason,
                self.vllm_config,
            )
            if disposition == "error":
                # [Lwd][eos-trace] point B: leave a trace for kill rows;
                # pair by down_seqno with A (cloud zero-accepted row) / C
                # (worker missing row) to settle the root cause
                LwdControlDiag.trace_edge_kill_row(
                    seqno=notify.down_seqno,
                    rid=request_id,
                    accepted=accepted,
                    finish_code=(
                        notify.finish_reasons[index] if notify.finish_reasons else -1
                    ),
                    batch_rows=len(notify.req_ids),
                    sampled_rows=len(sampled_token_map),
                )
                finish_reason = FinishReason.ERROR
                finished_now = True
            if not self.lwd_edge_deliver_tokens(
                request_id, sampled_token_ids, finished=finished_now
            ):
                lwd_logger.warning(
                    "[edge-deliver] drop stale cloud payload for %s (not awaiting)",
                    request_id,
                )
                continue
            if finished_now:
                # The ground truth of the cloud's stop decision lands here:
                # native finish settles the account (KV/cache release and
                # the finished_req_ids handover all go through the native
                # path). Settle the books BEFORE finish_requests: the
                # native path removes the request, after which the engine's
                # route entry and this engine's decode_load would have no
                # owner left to clean them -- leaking the load would hold
                # the gate at full slots forever (both sides then idle:
                # the edge gates every engine, the cloud has no work).
                self._lwd_settle_request(request_id)
                self.finish_requests(
                    [request_id],
                    _LWD_FINISH_STATUS.get(
                        finish_reason, RequestStatus.FINISHED_STOPPED
                    ),
                )
            outputs.append(
                EngineCoreOutput(
                    request_id,
                    sampled_token_ids,
                    finish_reason=finish_reason,
                )
            )
            if finished_now:
                finished.add(request_id)
        return outputs, finished

    @staticmethod
    def _lwd_finish_code(notify: LwdC2eNotify, index: int) -> FinishReason | None:
        """Per-request finish code: sentinel = not finished (None);
        position-aligned with req_ids, a mismatch raises IndexError
        fail-fast."""
        code = notify.finish_reasons[index]
        return None if code == LWD_NOT_FINISHED else FinishReason(code)

    def _lwd_pick_prefill_batch(
        self, dp_idx: int
    ) -> tuple[list[str], dict[str, int], int]:
        """Picker adapter layer: candidates routed to dp_idx, cap and
        free_seq from that engine's account."""
        account = self._lwd_accounts[dp_idx]
        inflight = account.load
        cap = self.max_num_scheduled_tokens
        free_seq = self.max_num_running_reqs - inflight
        # Resume candidates take only the prefill phase (decode-phase
        # residents have remaining=0, which the picker would skip anyway;
        # after the filter, a non-empty candidate list is itself the resume
        # criterion). Batch engine uniformity: only requests routed to
        # dp_idx enter -- one RangeNotify, one seqno, one UP channel.
        running = [
            LwdPrefillCandidate(
                r.request_id, r.num_prompt_tokens, r.num_computed_tokens
            )
            for r in self._lwd_reqs_in_phase(LwdReqPhase.PREFILL, self.running)
            if self._lwd_dp_of(r.request_id) == dp_idx
        ]
        waiting = [
            LwdPrefillCandidate(r.request_id, r.num_prompt_tokens, 0)
            for r in [*self.skipped_waiting, *self.waiting]
            if self._lwd_dp_of(r.request_id) == dp_idx
        ]
        req_ids, expected = lwd_pick_prefill_batch(
            running,
            waiting,
            cap=cap,
            free_seq=free_seq,
            chunk_unit=self._lwd_chunk_unit,
            single_prefill_min_tokens=(
                8000
                if self.vllm_config.model_config.hf_text_config.model_type
                == "deepseek_v4"
                else 0
            ),
        )
        return req_ids, expected, cap

    def _lwd_schedule_batch(self, dp_idx: int) -> SchedulerOutput:
        """Chunk batch assembly: the picker selects the batch of one engine,
        scheduled via the base-class visible-set mechanism (budget clamped
        to that engine's cap; the chunk split is guaranteed by the clamping
        parameters to be reproduced natively)."""
        req_ids, _expected, cap = self._lwd_pick_prefill_batch(dp_idx)
        return self._lwd_schedule(req_ids, token_budget_cap=cap)

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int]]:
        """Edge-side passthrough: encoder execution and caching are managed
        by the worker itself; native rollback would break the chunk
        contract, so nothing is scheduled, consumed, or rolled back."""
        return [], num_new_tokens, encoder_compute_budget, []

    def _lwd_has_prefill_chunk_inflight(self, dp_idx: int) -> bool:
        """Whether dp_idx has an embed request not fully sent; same
        criterion as the picker resume branch -- the two must stay
        consistent (the gate lets a resume finish before new work)."""
        return any(
            req.num_computed_tokens < req.num_prompt_tokens
            and self._lwd_route.get(req.request_id) == dp_idx
            for req in self.running
            if not req.is_finished()
        )

    def lwd_admit_request(self, request: Request) -> int:
        """Scheduler-local half of request admission: mode boundary
        validation, mrope precomputation and routing. Returns the routed
        dp_idx; the cloud announcement and enqueueing belong to the
        engine's add_request; abort_immediately goes through the native
        engine path (the ledger is pruned in the abort_requests override)."""
        self._lwd_validate_request(request)
        self._lwd_compute_mrope_positions(request)
        return self._lwd_route_request(request.request_id)

    @staticmethod
    def _lwd_request_has_mm(request: Request) -> bool:
        """Whether the request carries real multimodal data (prompt_embeds
        passthrough does not count -- it has no grid metadata, mrope is
        handled as text positions, consistent with the native
        definition)."""
        return any(
            feature.modality != "prompt_embeds" for feature in request.mm_features
        )

    def _lwd_compute_mrope_positions(self, request: Request) -> None:
        """At multimodal admission, compute the whole-prompt [3,N] mrope
        (only the edge holds the real prompt/grid); reject model families
        lacking _get_mrope_input_positions."""
        if not self.vllm_config.model_config.uses_mrope:
            return
        if not self._lwd_request_has_mm(request):
            return
        if self._lwd_mrope_positions_fn is None:
            from vllm.model_executor.models import ModelRegistry

            model_cls, _ = ModelRegistry.resolve_model_cls(
                self.vllm_config.model_config.architectures,
                self.vllm_config.model_config,
            )
            fn = getattr(model_cls, "_get_mrope_input_positions", None)
            if fn is None:
                raise ValueError(
                    f"[LWD] multimodal requests in prefill-only mode "
                    f"require a model class exposing "
                    f"_get_mrope_input_positions (Qwen3VL family, incl. "
                    f"Qwen3.5); got {model_cls.__name__} "
                    f"(request {request.request_id})"
                )
            self._lwd_mrope_positions_fn = fn
        mrope_features = [
            f for f in request.mm_features if f.modality != "prompt_embeds"
        ]
        _t_mrope = time.monotonic()
        positions, _delta = self._lwd_mrope_positions_fn(
            input_tokens=list(request.prompt_token_ids),
            mm_features=mrope_features,
            config=self.vllm_config.model_config.hf_config,
        )
        assert positions.shape[1] == request.num_prompt_tokens, (
            f"[LWD] mrope positions width {positions.shape[1]} != prompt "
            f"len {request.num_prompt_tokens} (request {request.request_id})"
        )
        self._lwd_mrope_positions_dict[request.request_id] = positions
        # [Lwd][perf] per-request latency of whole-prompt mrope computation
        # (runs on the edge engine main thread; the scheduling loop stalls
        # meanwhile; used for MM degradation attribution)

    def lwd_mrope_positions(self, request_id: str) -> bytes:
        """Return full-prompt positions for the engine's RequestNotify."""
        positions = self._lwd_mrope_positions_dict.get(request_id)
        if positions is None:
            return b""
        return positions.contiguous().numpy().tobytes()

    def lwd_edge_notify_range(
        self, dp_idx: int, scheduler_output: SchedulerOutput
    ) -> None:
        """Send one batch RangeNotify of dp_idx and attach the EMBED batch;
        a seqno of that engine is only taken by a notify that was sent."""
        built = self._lwd_build_range_items(scheduler_output.num_scheduled_tokens)
        if built is None:
            return
        items, req_ids, token_slices, prompt_offsets, has_mrope = built
        account = self._lwd_accounts[dp_idx]
        seqno = account.take_seqno()
        assert self.lwd_send_to_cloud is not None, "cloud not connected"
        lwd_logger.info(
            "[edge] range announced dp%d seqno=%d reqs=%d tokens=%d",
            dp_idx,
            seqno,
            len(items),
            sum(i.num_tokens for i in items),
        )
        self.lwd_send_to_cloud(dp_idx, LwdRangeNotify(seqno=seqno, items=items))
        account.register_batch(seqno, set(req_ids))
        # On successful send, assemble the EMBED batch and attach it to SO:
        # seqno is the pairing key of the data-plane tensor sent to the
        # cloud (same value as registered by the cloud-side RangeNotify);
        # the embed payload is this batch's per-request token slices (the
        # edge worker flattens them in req_ids order).
        scheduler_output.lwd_batch = LwdBatch(
            batch_type=LwdBatchType.LWD_EMBED,
            seqno=seqno,
            batch_meta=LwdEmbedBatch(
                req_ids=req_ids,
                token_ids=token_slices,
                prompt_offsets=prompt_offsets,
                has_mrope=has_mrope,
                dp_idx=dp_idx,
            ),
        )

    def _lwd_build_range_items(
        self, scheduled: dict[str, int]
    ) -> (
        tuple[list[LwdRangeItem], list[str], list[list[int]], list[int], list[bool]]
        | None
    ):
        """Assemble per-request RangeItems and EMBED payload slices.
        Returns None when there are no entries."""
        items: list[LwdRangeItem] = []
        req_ids: list[str] = []
        token_slices: list[list[int]] = []
        prompt_offsets: list[int] = []
        has_mrope: list[bool] = []
        for request_id, num_tokens in scheduled.items():
            request = self.requests.get(request_id)
            if request is None:
                continue
            # _update_after_schedule already advanced num_computed
            # optimistically; the start offset must step back by this
            # step's amount
            offset = request.num_computed_tokens - num_tokens
            req_has_mrope = request_id in self._lwd_mrope_positions_dict
            items.append(
                LwdRangeItem(
                    request_id=request_id,
                    offset=offset,
                    num_tokens=num_tokens,
                    has_mrope=req_has_mrope,
                )
            )
            req_ids.append(request_id)
            token_slices.append(
                list(request.prompt_token_ids[offset : offset + num_tokens])
            )
            prompt_offsets.append(offset)
            has_mrope.append(req_has_mrope)
        return (
            (items, req_ids, token_slices, prompt_offsets, has_mrope) if items else None
        )

    def lwd_edge_abort(self, request_ids: list[str]) -> None:
        """Settle aborted requests in their original DP accounts."""
        for request_id in request_ids:
            self._lwd_settle_request(request_id)

    def lwd_edge_deliver_tokens(
        self, request_id: str, token_ids: list[int], finished: bool
    ) -> bool:
        """Claim a live cloud result and move its load to decode exactly once;
        terminated/unknown requests are idempotently dropped."""
        request = self.requests.get(request_id)
        if request is None or request.is_finished():
            # A late cloud result racing the request's own finish: the
            # books may still hold it if the finish path did not pass
            # through here (stale payload drop idempotent). Clean the
            # account, or the leaked decode_load holds the gate forever.
            self._lwd_settle_request(request_id)
            lwd_logger.warning(
                "[edge-deliver] stale result for req=%s (not awaiting)",
                request_id,
            )
            return False
        self._lwd_settle_request(request_id, first_token=True)
        return True

    def _lwd_settle_request(self, request_id: str, first_token: bool = False) -> None:
        """Move load on delivery, or drop all request state on finish/abort.
        Send watermarks alone release pending batches after first token."""
        account = self._lwd_account_of(request_id)
        if account is None:
            return
        if first_token:
            account.note_prefill_done(request_id)
            return
        account.drop_request(request_id)
        was_awaiting = request_id not in account.prefill_requests
        account.note_request_settled(request_id, was_awaiting)
        self._lwd_route.pop(request_id, None)

    def _lwd_purge_mrope_positions(self) -> None:
        """Reconcile and remove mrope positions: the request has left
        requests (terminated/aborted) or finished embedding and
        transitioned to decode phase (all chunks already sent, notify no
        longer reads it). Resident models have no natural removal point of
        "leave the queue on embedding completion", so the harvest phase
        reconciles uniformly."""
        for rid in list(self._lwd_mrope_positions_dict):
            request = self.requests.get(rid)
            if request is None or self._lwd_req_phase(request) is LwdReqPhase.DECODE:
                self._lwd_mrope_positions_dict.pop(rid, None)

    def _lwd_validate_request(self, request: Request) -> None:
        """Mode boundary validation (edge capability surface): reject
        client-provided prompt_embeds/pooling/structured output/sampling
        params not carried on the wire; violations raise ValueError and are
        rejected before enqueueing."""
        if request.prompt_embeds is not None:
            raise ValueError(
                f"[LWD] prefill-only mode does not accept client-provided "
                f"prompt_embeds (request {request.request_id}); the edge "
                "is the embedding owner"
            )
        self._lwd_validate_mm_request(request)
        if request.pooling_params is not None:
            raise ValueError(
                "[LWD] prefill-only mode does not support pooling requests "
                f"(request {request.request_id})"
            )
        if request.use_structured_output:
            raise ValueError(
                "[LWD] prefill-only mode does not support structured output "
                f"(request {request.request_id})"
            )
        sp = request.sampling_params
        unsupported = [
            name
            for name, value in (
                ("logit_bias", sp.logit_bias),
                ("allowed_token_ids", sp.allowed_token_ids),
                ("logprobs", sp.logprobs),
            )
            if value is not None
        ]
        if unsupported:
            raise ValueError(
                "[LWD] prefill-only mode does not support sampling "
                f"param(s) {', '.join(unsupported)} "
                f"(request {request.request_id}): not carried on the "
                "edge->cloud wire, cloud would silently sample without them"
            )

    def _lwd_validate_mm_request(self, request: Request) -> None:
        """Multimodal allowlist: image only (the merge/mrope chain is
        implemented with image semantics; other modalities are unadapted,
        silently letting them through would compute wrongly); reject EVS
        (multimodal pruning)."""
        if not request.mm_features:
            return
        unsupported_mm = sorted(
            {
                feature.modality
                for feature in request.mm_features
                if feature.modality not in ("image", "prompt_embeds")
            }
        )
        if unsupported_mm:
            raise ValueError(
                f"[LWD] prefill-only mode supports image-only "
                f"multimodal requests; got modality "
                f"{', '.join(unsupported_mm)} "
                f"(request {request.request_id})"
            )
        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config.is_multimodal_pruning_enabled():
            raise ValueError(
                "[LWD] prefill-only mode does not support multimodal "
                f"pruning (EVS) (request {request.request_id})"
            )


def lwd_build_unembed_batch(dp_idx: int, notify: LwdC2eNotify) -> SchedulerOutput:
    """Assemble the UNEMBED batch: cloud results bypass native scheduling;
    use make_empty as the skeleton and attach LwdUnembedBatch of dp_idx."""
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = dict(
        zip(notify.req_ids, notify.num_accepted_tokens)
    )
    scheduler_output.total_num_scheduled_tokens = sum(notify.num_accepted_tokens)
    scheduler_output.lwd_batch = LwdBatch(
        batch_type=LwdBatchType.LWD_UNEMBED,
        seqno=notify.down_seqno,
        batch_meta=LwdUnembedBatch(
            req_ids=list(notify.req_ids),
            num_accept_tokens=list(notify.num_accepted_tokens),
            recv_num_elements=notify.hidden_num_elements,
            out_token_idxs=[],
            top_id_ths=list(notify.top_id_ths),
            token_ids=[list(t) for t in notify.token_ids],
            dp_idx=dp_idx,
        ),
    )
    scheduler_output.lwd_c2e_notify = [notify]
    return scheduler_output
