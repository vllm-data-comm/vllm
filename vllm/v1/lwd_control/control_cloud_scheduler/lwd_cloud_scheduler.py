# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cloud-side mixed-batch scheduler (the sole PD_mix form): each step
consumes at most one batched RangeNotify from the head of the notify queue
and dispatches it in the same batch with all decode-phase requests; see
lwd_mixed_batch_design.md + lwd_chunked_prefill_design.md for the design.

If any of the four gates (classification / ADD landing / early-receive
watermark / preflight triple check) fails, this step becomes decode-only
and the notify stays at the queue head for retry; the admitted step runs
offset reconciliation + dual-clamp reproduction + full post-step
verification for fail-fast. Incompatible with cloud-side preemption
recomputation and prefix caching (deployment must guarantee sufficient KV
and disable caching); the verification turns violations into immediate
errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.lwd_logger import lwd_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import (
    LwdBatch,
    LwdBatchType,
    LwdEmbedBatch,
    SchedulerOutput,
)
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_prefill_notify_queue import (
    LwdPrefillNotifyQueue,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    LwdBaseScheduler,
    LwdReqPhase,
    lwd_draft_slots,
    lwd_spec_factor,
)
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LWD_NOT_FINISHED,
    LwdC2eNotify,
    LwdRangeItem,
    LwdRangeNotify,
    LwdRequestNotify,
)
from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag
from vllm.v1.lwd_control.lwd_model.lwd_model_family import (
    LwdRequestMetadata,
    LwdRequestMetadataLayout,
    lwd_request_metadata_columns,
)

if TYPE_CHECKING:
    from vllm.v1.request import Request

LwdDownCarrier = tuple[torch.Tensor, list[str], int, int]
"""``ModelRunnerOutput.lwd_down_carrier`` attached by the cloud worker:
(pinned packed meta, req_ids, hidden numel, DOWN seqno)."""


@dataclass
class LwdC2eMeta:
    """Step metadata unpacked from the cloud worker's DOWN carrier; the
    scheduler forwards it to the edge ahead of the DOWN hidden tensor
    (the data plane carries only the tensor itself)."""

    # Total element count of the DOWN hidden tensor for this step.
    hidden_num_elements: int
    # Per-request sampled-token rank within the step's logits ("top-id-th"),
    # one row per request in hidden row order.
    top_id_ths: list[list[int]]
    # Per-request draft accept counts (0 for non-spec steps).
    num_accepted_tokens: list[int]
    # Request ids in hidden row order.
    req_ids: list[str]
    # DOWN channel seqno of this step's packet; the edge posts its matching
    # irecv with this exact value for pairing on the tag-less HCCL wire.
    down_seqno: int


class LwdCloudScheduler(LwdBaseScheduler):
    """Notify-gated prefill/decode mixed scheduler (PD_mix, the sole
    cloud-side form)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # seqno ledger for one-shot gate logs: recorded once on the first HOLD
        # check, cleared and recorded again on OPEN
        self._lwd_gate_logged: set[int] = set()
        # Prefill notify queue: the engine ingest thread posts edge-side
        # batched RangeNotify here; this thread consumes at most one head item
        # per step. The notify carries its own seqno, which is the UP-chain
        # pairing number of this batch. The engine side posts/sweeps via
        # self.scheduler.prefill_notify_queue by reference (thread discipline
        # belongs to the notify queue)
        self.prefill_notify_queue = LwdPrefillNotifyQueue()
        # MTP budget factor: each decode request consumes 1+k tokens per step
        # (draft tokens take budget and KV); the sole home of the formula is
        # lwd_base_scheduler.lwd_spec_factor
        self._lwd_spec_factor: int = lwd_spec_factor(self.vllm_config)
        self._lwd_draft_slots: int = lwd_draft_slots(self.vllm_config)
        lwd_logger.info(
            "[cloud-sched] chunked-prefill batches mixed "
            "with decode (one notify consumed per step at most, "
            "spec_factor=%d, draft_slots=%d, lookahead=%d)",
            self._lwd_spec_factor,
            self._lwd_draft_slots,
            self.num_lookahead_tokens,
        )
        # Diagnostics hub: the spec acceptance-rate accumulators / periodic
        # direct output and bypass payload extraction all live in
        # lwd_control_diag; the scheduler only calls them, never owns them
        self._lwd_diag = LwdControlDiag()
        # Pending c2e: update_from_output only enqueues (pure extraction, no
        # I/O); the engine post_step flushes them to the edge in one shot --
        # the edge side pre-posts DOWN recvs accordingly
        self.pending_c2e: list[LwdC2eNotify] = []
        self._lwd_request_metadata_layout = LwdRequestMetadataLayout.from_model_config(
            self.vllm_config.model_config
        )
        # Whole-request model-family metadata of the request notify, until
        # the request's first chunk carries it to the runner. Written by the
        # ingest thread before it posts the request's ADD, read and removed
        # by the scheduling thread afterwards (single-key dict operations).
        self._lwd_request_metadata: dict[str, LwdRequestMetadata] = {}

    def update_from_output(self, scheduler_output, model_runner_output):
        """After native accounting (including spec observation), unpack the
        carrier into a c2e appended to pending_c2e; the DISABLE_DOWN bypass
        passes through per-request tokens directly; empty steps without a
        carrier are skipped."""
        self._lwd_diag.observe_spec_acceptance(
            scheduler_output, model_runner_output, self.requests
        )
        engine_core_outputs = super().update_from_output(
            scheduler_output, model_runner_output
        )
        carrier = getattr(model_runner_output, "lwd_down_carrier", None)
        if carrier is not None:
            meta = self._lwd_carrier_to_meta(carrier)
            # Deliver committed counts (same source as the finish
            # decision): collector counts are device-side derivatives
            # without check_stop truncation (stop/EOS/max_tokens), and
            # the edge delivery path performs no local stop check — the
            # delivered count must be same-source with the finish
            # decision. Requests absent from the outputs get 0 (the
            # edge's pad-and-skip disposition handles them).
            committed = {
                out.request_id: len(out.new_token_ids)
                for outputs in engine_core_outputs.values()
                for out in outputs.outputs
            }
            meta.num_accepted_tokens = [committed.get(rid, 0) for rid in meta.req_ids]
            LwdControlDiag.cloud_step(self, meta, engine_core_outputs)
            finish_reasons = self._lwd_c2e_finish_reasons(
                meta.req_ids, engine_core_outputs
            )
            # [Lwd][eos-trace] Point A: leave a trace of zero-acceptance rows
            # (for kill root-cause localization; prints the raw row set --
            # zero-acceptance live-row removal was dropped together with the
            # 4-card fix; kept here only as an incidence-rate observation)
            LwdControlDiag.trace_cloud_c2e_rows(
                meta.down_seqno,
                meta.req_ids,
                meta.num_accepted_tokens,
                finish_reasons,
            )
            # Diagnostics bypass (SKIP_SAMPLE): piggyback per-request
            # accepted ids
            token_ids: list[list[int]] = []
            if LwdControlDiag.token_passthrough_enabled():
                token_ids = LwdControlDiag.bypass_token_payload(
                    meta.req_ids, engine_core_outputs
                )
            self.pending_c2e.append(
                LwdC2eNotify(
                    hidden_num_elements=meta.hidden_num_elements,
                    top_id_ths=meta.top_id_ths,
                    num_accepted_tokens=meta.num_accepted_tokens,
                    req_ids=meta.req_ids,
                    finish_reasons=finish_reasons,
                    down_seqno=meta.down_seqno,
                    token_ids=token_ids,
                )
            )
        elif LwdControlDiag.disable_down_enabled():
            self._lwd_append_diag_c2e(engine_core_outputs)
        return engine_core_outputs

    def _lwd_append_diag_c2e(self, engine_core_outputs) -> None:
        """DISABLE_DOWN diagnostics bypass: the cloud does no collect / sends
        no DOWN, but the c2e still drives edge-side delivery as usual --
        hidden=0 marks "no tensor"; row-set extraction belongs to the
        diagnostics class, wire construction stays in the scheduler. An
        empty step with no rows sends no empty notify."""
        rows = LwdControlDiag.disable_down_rows(engine_core_outputs)
        if not rows:
            return
        req_ids = list(rows)
        self.pending_c2e.append(
            LwdC2eNotify(
                hidden_num_elements=0,
                top_id_ths=[],
                num_accepted_tokens=[],
                req_ids=req_ids,
                finish_reasons=self._lwd_c2e_finish_reasons(
                    req_ids, engine_core_outputs
                ),
                down_seqno=-1,
                token_ids=[rows[rid] for rid in req_ids],
            )
        )

    @staticmethod
    def _lwd_carrier_to_meta(carrier: LwdDownCarrier) -> LwdC2eMeta:
        """Unpack the harvested payload into step metadata.

        Payload = a flat int tensor (CPU, harvested by the worker's
        fenced copy: dedicated stream wait_event on the per-step ready
        event, completion event recorded on the copy stream itself),
        laid out as "variable-length segments first, fixed-length tails
        last", so it can be decoded backwards from the tail without
        carrying extra length fields:

            vals = [ranks(variable length: seg_lens[i] rank ids per
                    request, concatenated in request order) ...
                    | counts(n_req accepted) ...
                    | seg_lens(n_req segment lengths)]

        Take n_req=2, seg_lens=[3,1], counts=[5,2] as an example:
            vals = [r0 r1 r2 | r0 | 5 2 | 3 1]
            counts   = vals[-2*n_req : -n_req] -> [5, 2]
            seg_lens = vals[-n_req:]            -> [3, 1]
            ranks    = vals[:-2*n_req]          -> [r0, r1, r2, r0]
        Then slice ranks back into per-request top_id_ths rows by seg_lens
        (here [[r0,r1,r2],[r0]]). The two magic numbers -2*n_req/-n_req are
        the widths of the two fixed-length tails, the "counts segment +
        seg_lens segment"."""
        packed, req_ids, hidden_numel, seqno = carrier
        n_req = len(req_ids)
        vals = packed.tolist()
        counts = vals[-2 * n_req : -n_req]
        seg_lens = vals[-n_req:]
        ranks_flat = vals[: -2 * n_req]
        top_id_ths: list[list[int]] = []
        off = 0
        for seg_len in seg_lens:
            top_id_ths.append(ranks_flat[off : off + seg_len])
            off += seg_len
        return LwdC2eMeta(
            hidden_num_elements=hidden_numel,
            top_id_ths=top_id_ths,
            num_accepted_tokens=list(counts),
            req_ids=list(req_ids),
            down_seqno=seqno,
        )

    @staticmethod
    def _lwd_c2e_finish_reasons(req_ids: list[str], engine_core_outputs) -> list[int]:
        """Per-request finish codes aligned with req_ids: take the code from
        outputs carrying finish_reason, fill gaps in finished_requests with
        ABORT, and NOT_FINISHED for the rest."""
        from vllm.v1.engine import FinishReason

        reasons: dict[str, int] = {}
        for outputs in engine_core_outputs.values():
            for out in outputs.outputs:
                if out.finish_reason is not None:
                    reasons.setdefault(out.request_id, int(out.finish_reason))
            for request_id in outputs.finished_requests or ():
                reasons.setdefault(request_id, int(FinishReason.ABORT))
        return [reasons.get(request_id, LWD_NOT_FINISHED) for request_id in req_ids]

    def _lwd_collect_decode_requests(self) -> list[str]:
        """Collect req_ids of decode-phase requests (prompt fully computed)
        in running/waiting/skipped order; the output can directly serve as
        input to visible-set scheduling; mid-prefill requests (num_computed
        still ramping up) do not enter the decode set -- they stay in the
        hidden set waiting for their own next chunk notify to drive their
        resume scheduling."""
        return [
            req.request_id
            for req in self._lwd_reqs_in_phase(
                LwdReqPhase.DECODE,
                self.running,
                self.waiting,
                self.skipped_waiting,
            )
        ]

    # ------------------------------------------------------------------ #
    # Notify item classification and preflight                           #
    # ------------------------------------------------------------------ #
    def _lwd_filter_ready_items(
        self, items: list[LwdRangeItem]
    ) -> tuple[list[LwdRangeItem], bool]:
        """Filter ready items out of the notify items.

        items is dedicated to prefill_notify: the items of one
        LwdRangeNotify (a prefill notify) form one edge-side prefill batch
        -- one LwdRangeItem per request in the batch (request_id + that
        chunk's offset/num_tokens + mrope carry flag), and the whole batch
        shares one seqno (the UP data-plane pairing number), hence the
        whole batch is all-or-nothing.

        Each item is checked against the cloud-side ledger (self.requests)
        -- ready (landed and awaiting prefill) kept, stale (already in
        decode phase, duplicate notify) dropped idempotently, unknown (ADD
        not yet landed) sets a flag. Returns (ready, whether any unknown);
        the caller defers the whole batch based on unknown (one UP receive
        corresponds to the whole batch of tensors, a subset cannot be
        scheduled alone).

        Any unknown means deferring the whole batch (all-or-nothing: one UP
        recv corresponds to the whole batch of tensors; only a subset
        cannot be admitted)."""
        ready: list[LwdRangeItem] = []
        unknown = False
        for item in items:
            request = self.requests.get(item.request_id)
            if request is None:
                unknown = True
                continue
            if self._lwd_req_phase(request) is LwdReqPhase.DECODE:
                lwd_logger.warning(
                    "[cloud-sched] drop stale notify item req=%s "
                    "num=%d (already prefilled; duplicate notify?)",
                    item.request_id,
                    item.num_tokens,
                )
                continue
            ready.append(item)
        return ready, unknown

    def _lwd_up_recv_complete(self, notify: LwdRangeNotify) -> bool:
        """Early-receive gate: release only after the UP data of this seqno
        is fully received; criterion = channel watermark (FIFO completion
        order)."""
        from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
            is_irecv_complete,
            watermark,
        )

        ready = is_irecv_complete(LwdChannelType.UP, notify.seqno)
        # [Lwd][pre-recv] Gate readings: use the same monotonic clock as
        # HINT/SUBMIT/DONE/REPORT/POST, so the four segments "notify arrival
        # -> worker posts recv -> worker reports completion -> engine
        # releases" can be subtracted directly. Logged once per seqno (first
        # HOLD check and OPEN) to avoid flooding on every idle step.
        seen = self._lwd_gate_logged
        if ready:
            if notify.seqno in seen:
                return True
            seen.discard(notify.seqno)
            lwd_logger.info(
                "[cloud-sched] notify seqno=%d gate OPEN watermark=%d",
                notify.seqno,
                watermark(LwdChannelType.UP),
            )
        elif notify.seqno not in seen:
            seen.add(notify.seqno)
            lwd_logger.info(
                "[cloud-sched] notify seqno=%d gate HOLD watermark=%d "
                "(UP not fully received; silent retry afterwards)",
                notify.seqno,
                watermark(LwdChannelType.UP),
            )
        return ready

    def _lwd_preflight_ok(self, items: list[LwdRangeItem], num_decode: int) -> bool:
        """Admission triple check: token and input budgets / slots (counting
        new additions only) / conservative KV estimate; decode occupancy is
        counted as 1+k. The native scheduler also charges every scheduled
        request its parallel-drafting slots against the input budget and
        reserves lookahead KV slots for it; both are counted here, or the
        native step would cut the edge's chunks short."""
        need_tokens = (
            sum(item.num_tokens for item in items) + num_decode * self._lwd_spec_factor
        )
        if need_tokens > self.max_num_scheduled_tokens:
            lwd_logger.info(
                "[cloud-sched] preflight hold: tokens %d > budget %d",
                need_tokens,
                self.max_num_scheduled_tokens,
            )
            return False
        need_inputs = need_tokens + (num_decode + len(items)) * self._lwd_draft_slots
        input_budget = self.scheduler_config.max_num_batched_tokens
        if need_inputs > input_budget:
            lwd_logger.info(
                "[cloud-sched] preflight hold: input slots %d > budget %d",
                need_inputs,
                input_budget,
            )
            return False
        # Slots count only new admissions: requests resuming a chunk are
        # already in running; counting them again by item count would
        # double-occupy slots and cause a false hold (before chunking the
        # batch was always new additions, so len(items) equaled the number
        # of new additions)
        running_ids = {req.request_id for req in self.running}
        new_items = sum(1 for item in items if item.request_id not in running_ids)
        if len(self.running) + new_items > self.max_num_running_reqs:
            lwd_logger.info(
                "[cloud-sched] preflight hold: seqs %d+%d > max %d",
                len(self.running),
                new_items,
                self.max_num_running_reqs,
            )
            return False
        lookahead = self.num_lookahead_tokens
        need_blocks = sum(
            cdiv(item.num_tokens + lookahead, self.block_size) for item in items
        ) + num_decode * cdiv(self._lwd_spec_factor + lookahead, self.block_size)
        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        if need_blocks > free_blocks:
            lwd_logger.info(
                "[cloud-sched] preflight hold: kv blocks %d > free %d",
                need_blocks,
                free_blocks,
            )
            return False
        return True

    def _lwd_schedule_step(self) -> SchedulerOutput:
        """Mixed scheduling: four gates decide each step -- item
        reconciliation (all-or-nothing) -> early-receive watermark ->
        preflight triple check -> mixed step; if any gate fails, this step
        is decode-only and the notify stays at the queue head for retry
        next step."""
        decode_ids = self._lwd_collect_decode_requests()
        notify = self.prefill_notify_queue.peek()
        if notify is not None:
            items, unknown = self._lwd_filter_ready_items(list(notify.items))
            if not items and not unknown:
                # All items invalid (finished/duplicate): pop the queue,
                # pure decode this step
                self.prefill_notify_queue.pop_front(notify)
                notify = None
            elif unknown:
                # Request ADD not yet landed (input_queue ordering lag): the
                # notify stays at the queue head; decode first this step
                lwd_logger.info(
                    "[cloud-sched] notify seqno=%d deferred: request(s) not added yet",
                    notify.seqno,
                )
                notify = None
            elif not self._lwd_up_recv_complete(notify):
                # Early-receive gate: UP tensors not fully received -> do
                # not dispatch this prefill batch this step (the notify
                # stays at the queue head, re-checked next step); this step
                # dispatches decode only. Fully received batches take
                # priority; for batches still receiving, keep decode
                # spinning instead of stuffing the wait into the worker
                # forward pass.
                notify = None
            elif not self._lwd_preflight_ok(items, len(decode_ids)):
                # Budget/slots/KV insufficient: the notify stays at the queue
                # head; retry after decode drains. This one must be logged:
                # it is the only silent exit for "data fully received and
                # the gate already open, yet prefill never dispatches" (when
                # troubleshooting a tail stall on site, check this line
                # first).
                lwd_logger.info(
                    "[cloud-sched] notify seqno=%d deferred: preflight "
                    "rejected (reqs=%d tokens=%d decode=%d pending_notify=%d)",
                    notify.seqno,
                    len(items),
                    sum(item.num_tokens for item in items),
                    len(decode_ids),
                    len(self.prefill_notify_queue),
                )
                notify = None
            else:
                return self._schedule_mixed(notify, items, decode_ids)
        return self._lwd_schedule(decode_ids)

    def _schedule_mixed(
        self,
        notify: LwdRangeNotify,
        items: list[LwdRangeItem],
        decode_ids: list[str],
    ) -> SchedulerOutput:
        """Mixed step: decode exact-consumption derivation + dual clamp
        (threshold/budget) reproduces chunk boundaries via the base class;
        consume the queue head and attach the EMBED batch."""
        req_ids = decode_ids + [item.request_id for item in items]
        out = self._lwd_schedule(
            req_ids,
            token_budget_cap=self._lwd_compute_decode_consumption(decode_ids)
            + sum(item.num_tokens for item in items),
            chunk_threshold=max(item.num_tokens for item in items),
        )
        # Consume the queue head (the notify queue pops by object identity;
        # if this notify was already removed by the abort sweep, pop_front
        # returns False, equivalent to consumed)
        self.prefill_notify_queue.pop_front(notify)
        self._lwd_attach_embed_batch(out, notify, items)
        return out

    def _lwd_compute_decode_consumption(self, decode_ids: list[str]) -> int:
        """Decode exact consumption: the exactness of the budget clamp
        relies on replicating verbatim the num_new derivation of the native
        running section (scheduler.py:381-416) -- 1+len(spec) is not the
        accurate formula:
        ① placeholder (non-zero values left over from spec rejection)
          enters the scheduling window;
        ② the max_tokens guard (placeholder>0 and fewer than 2 tokens left
          to max_tokens) and next_decode_eligible_step can skip a request
          for a whole step, consuming 0.
        Without verbatim replication, slack remains and the last truncated
        chunk gets over-scheduled (real-machine log: a decode request
        skipped by the guard consumed 0, and the last chunk of 604 was
        scheduled as 608). threshold does not constrain decode (normal
        magnitudes T ≫ num_new; pathological small chunk configs see design
        §8-C9); the budget is guaranteed sufficient by construction and
        does not affect this derivation."""
        decode_exact = 0
        for rid in decode_ids:
            req = self.requests[rid]
            placeholders = req.num_output_placeholders
            if (
                placeholders > 0
                and req.num_computed_tokens + 2 - placeholders
                >= req.num_prompt_tokens + req.max_tokens
            ):
                continue  # native max_tokens guard: skipped this step, consumes 0
            if self.current_step < req.next_decode_eligible_step:
                continue  # V2+PP cadence guard: skipped this step, consumes 0
            decode_exact += max(
                0,
                min(
                    req.num_tokens_with_spec + placeholders - req.num_computed_tokens,
                    self.max_model_len - 1 - req.num_computed_tokens,
                ),
            )
        return decode_exact

    def lwd_register_request_metadata(self, notify: LwdRequestNotify) -> None:
        """Ingest thread, before the request's ADD: check the notify's
        model-family metadata against this cloud's model and keep it for the
        request's first chunk.

        Raises:
            ValueError: The metadata does not fit the cloud model.
        """
        self._lwd_request_metadata_layout.validate(notify)
        metadata = LwdRequestMetadata.from_notify(notify)
        if not metadata.is_empty():
            self._lwd_request_metadata[notify.request_id] = metadata

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        # A request finished or aborted before its first chunk still holds
        # its metadata.
        self._lwd_request_metadata.pop(request.request_id, None)
        return super()._free_request(request, delay_free_blocks)

    def _lwd_attach_embed_batch(
        self,
        out: SchedulerOutput,
        notify: LwdRangeNotify,
        items: list[LwdRangeItem],
    ) -> None:
        """Attach the EMBED batch: the UP-chain seqno is dispatched with the
        batch to the cloud worker; the batch pairing number is the seqno
        carried by the notify (same source and value as the edge-side EMBED
        batch dispatch number). token_ids is a placeholder list -- its
        length must equal the per-request token count actually sent by the
        edge side, so that recv numel strictly equals the edge-side isend
        (HCCL P2P requires matching numel on both ends); mrope positions
        pass through per item: the runner writes them into req_state
        directly via frombuffer for MM requests (empty bytes for text
        requests, zero path change). Whole-request metadata rides the
        request's first chunk only (offset 0), where the runner caches it."""
        metadata = [
            self._lwd_request_metadata.pop(item.request_id, None)
            if item.offset == 0
            else None
            for item in items
        ]
        out.lwd_batch = LwdBatch(
            batch_type=LwdBatchType.LWD_EMBED,
            seqno=notify.seqno,
            batch_meta=LwdEmbedBatch(
                req_ids=[item.request_id for item in items],
                token_ids=[[0] * item.num_tokens for item in items],
                prompt_offsets=[item.offset for item in items],
                has_mrope=[item.has_mrope for item in items],
                mrope_positions=[item.mrope_positions for item in items],
                **lwd_request_metadata_columns(
                    [m or LwdRequestMetadata() for m in metadata]
                ),
            ),
        )
