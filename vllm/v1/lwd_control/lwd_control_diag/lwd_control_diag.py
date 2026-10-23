# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LWD control-plane diagnostics hub: on-the-spot switch reads, spec
acceptance-rate observation, diagnostic-bypass predicates and payload
extraction, debug instrumentation forwarding.

Functional classes (schedulers/engines) touch diagnostics only through
this module — no scattered env reads, no direct import of
vllm.v1.lwd_debug; diagnostic logic stays unmixed with functional
interfaces, so adding/removing switches/bypasses/observations only
changes this file. Boundary: this module only produces "predicates and
data" (switch booleans, bypass row sets, cumulative counters); it holds
no execution adapters (executor-boundary contracts such as futures and
fake outputs belong to the engine).

Diagnostic behavior definitions (troubleshooting semantics):
  - SKIP_SAMPLE (edge skips sampling): the cloud piggybacks this
    step's per-request accepted token ids on c2e, and the edge skips
    the lm_head sampling reconstruction;
  - DISABLE_DOWN (full passthrough): the cloud does not collect and
    sends no DOWN tensors; c2e carries the token ids directly with
    hidden=0, and the edge harvests and delivers them as-is.
Both are strictly == "1" and read on the spot at each call site
(changing the env in-process takes effect immediately; same name and
semantics as the read in vllm-ascend envs.py).
"""

from __future__ import annotations

import os
import time

_LWD_ENV_PREFIX = "VLLM_ASCEND_LWD_"

# Periodic log interval for the spec acceptance rate (counters are
# resident, logged periodically)
_SPEC_LOG_INTERVAL_S = 10.0


def _lwd_debug_mods():
    """Lazy import of lwd_debug: keeps this module's top level free of
    heavy dependencies (pure logic stays testable); the import of
    lwd_debug inside lwd_control is still funneled solely through here.
    Fallback: vllm.v1.lwd_debug is not migrated to main yet (issue #18
    breakpoint 4); until it lands, degrade to no-op shims so first-request
    diagnostics do not crash — the real classes take over automatically
    once the package exists."""
    try:
        from vllm.v1.lwd_debug import LwdControlLog, LwdDebug, LwdLogBase
    except ImportError:
        return _LwdDebugNoOp, _LwdDebugNoOp, _LwdDebugNoOp
    return LwdControlLog, LwdDebug, LwdLogBase


class _LwdDebugNoOp:
    """No-op stand-in for the not-yet-migrated vllm.v1.lwd_debug classes."""

    @staticmethod
    def set_debug(enabled: bool) -> None:
        pass

    @staticmethod
    def flight(running: int, awaiting: int) -> None:
        pass

    @staticmethod
    def edge_tokens_delivered(*args: object, **kwargs: object) -> None:
        pass

    @staticmethod
    def cloud_step(*args: object, **kwargs: object) -> None:
        pass

    @staticmethod
    def cloud_request_admitted(*args: object, **kwargs: object) -> None:
        pass


class LwdControlDiag:
    """The single home of control-plane diagnostic logic (functional
    classes reach diagnostics via this class / module)."""

    # ------------------------------------------------------------------ #
    # Switches (read on the spot at each call site, strictly == "1")     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def token_passthrough_enabled() -> bool:
        """Cloud-side master switch of the diagnostic bypass: True when
        either SKIP_SAMPLE (edge skips sampling) / DISABLE_DOWN (full
        passthrough) is on."""
        return (
            os.getenv(_LWD_ENV_PREFIX + "EDGE_SKIP_SAMPLE") == "1"
            or os.getenv(_LWD_ENV_PREFIX + "DISABLE_DOWN") == "1"
        )

    @staticmethod
    def disable_down_enabled() -> bool:
        """DISABLE_DOWN diagnostic switch: the cloud does not collect /
        sends no DOWN; the edge delivers via passthrough."""
        return os.getenv(_LWD_ENV_PREFIX + "DISABLE_DOWN") == "1"

    # ------------------------------------------------------------------ #
    # Debug instrumentation forwarding (sole lwd_debug import point)     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def set_debug(enabled: bool) -> None:
        """Layer log master switch (left alone if env already enabled it,
        turned on if the config section enables it, this process only)."""
        _, _, log_base = _lwd_debug_mods()
        log_base.set_debug(enabled)

    @staticmethod
    def edge_step_activity(running: int, awaiting: int) -> None:
        """Edge per-step activity instrumentation (running count /
        decode-phase resident count)."""
        control_log, _, _ = _lwd_debug_mods()
        control_log.flight(running, awaiting)

    @staticmethod
    def edge_tokens_delivered(
        request_id, sampled_token_ids, finish_reason, vllm_config
    ) -> None:
        """Edge per-request delivery instrumentation."""
        _, debug, _ = _lwd_debug_mods()
        debug.edge_tokens_delivered(
            request_id, sampled_token_ids, finish_reason, vllm_config
        )

    @staticmethod
    def cloud_step(scheduler, meta, engine_core_outputs) -> None:
        """Cloud step-metadata instrumentation."""
        _, debug, _ = _lwd_debug_mods()
        debug.cloud_step(scheduler, meta, engine_core_outputs)

    @staticmethod
    def cloud_request_admitted(wire, sampling_params) -> None:
        """Cloud request gate-passage instrumentation."""
        _, debug, _ = _lwd_debug_mods()
        debug.cloud_request_admitted(wire, sampling_params)

    # ------------------------------------------------------------------ #
    # Spec acceptance observation (counters live here, native semantics) #
    # ------------------------------------------------------------------ #
    def __init__(self) -> None:
        self._spec_draft_tokens = 0
        self._spec_accepted = 0
        self._spec_last_log = time.monotonic()

    def observe_spec_acceptance(
        self, scheduler_output, model_runner_output, requests
    ) -> None:
        """Accumulate draft/accepted counts with native semantics
        (num_draft = len(spec_ids), num_accepted = len(generated) - 1
        where the minus one drops the bonus non-draft row), logging
        running totals periodically; skips finished/unknown requests
        (same filter as native)."""
        spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        if not spec_tokens:
            return
        sampled = model_runner_output.sampled_token_ids
        req_id_to_index = model_runner_output.req_id_to_index
        for req_id, spec_ids in spec_tokens.items():
            if not spec_ids:
                continue
            request = requests.get(req_id)
            if request is None or request.is_finished():
                continue
            req_index = req_id_to_index.get(req_id)
            if req_index is None:
                continue
            generated = sampled[req_index] if sampled else []
            if not generated:
                continue
            self._spec_draft_tokens += len(spec_ids)
            self._spec_accepted += len(generated) - 1
        now = time.monotonic()
        if now - self._spec_last_log < _SPEC_LOG_INTERVAL_S:
            return
        self._spec_last_log = now
        if not self._spec_draft_tokens:
            return
        from vllm.lwd_logger import lwd_logger

        rate = self._spec_accepted / self._spec_draft_tokens * 100
        lwd_logger.info(
            "[spec] cumulative: rate=%.1f%% accepted=%d draft_tokens=%d",
            rate,
            self._spec_accepted,
            self._spec_draft_tokens,
        )

    # ------------------------------------------------------------------ #
    # Diagnostic bypass payloads                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def step_token_ids(engine_core_outputs) -> dict[str, list[int]]:
        """Per-request new_token_ids for this step (indexed by
        request_id), shared by the bypass payloads."""
        ids_by_req: dict[str, list[int]] = {}
        for outputs in engine_core_outputs.values():
            for out in outputs.outputs:
                ids_by_req[out.request_id] = list(out.new_token_ids)
        return ids_by_req

    @staticmethod
    def bypass_token_payload(
        req_ids: list[str], engine_core_outputs
    ) -> list[list[int]]:
        """SKIP_SAMPLE bypass: per-request accepted token ids aligned to
        the c2e row set; absent rows are padded empty (positionally
        aligned with req_ids)."""
        ids_by_req = LwdControlDiag.step_token_ids(engine_core_outputs)
        return [ids_by_req.get(rid, []) for rid in req_ids]

    @staticmethod
    def disable_down_rows(engine_core_outputs) -> dict[str, list[int]]:
        """DISABLE_DOWN passthrough row set: this step's per-request
        tokens + empty rows padded for requests that only appear in
        finished_requests (carrying the ABORT finish code); an empty
        step yields an empty dict (nothing to deliver, no empty notify
        is sent)."""
        ids_by_req = LwdControlDiag.step_token_ids(engine_core_outputs)
        for outputs in engine_core_outputs.values():
            for rid in outputs.finished_requests or ():
                ids_by_req.setdefault(rid, [])
        return ids_by_req

    @staticmethod
    def bypass_unembed_rows(batch) -> tuple[list[str], list[list[int]]] | None:
        """DISABLE_DOWN passthrough predicate and row set: returns
        (req_ids, bypass token rows) when the batch is UNEMBED and the
        notify carries bypass tokens (hidden=0), else None. Produces
        data only, no execution adapters — the passthrough future /
        fake output is wrapped by the edge engine."""
        if batch is None:
            return None
        from vllm.v1.core.sched.output import LwdBatchType

        if (
            batch.batch_type == LwdBatchType.LWD_UNEMBED
            and LwdControlDiag.disable_down_enabled()
            and batch.batch_meta.token_ids
        ):
            meta = batch.batch_meta
            return (
                list(meta.req_ids),
                [list(t) for t in meta.token_ids],
            )
        return None

    # ------------------------------------------------------------------ #
    # EOS tail-kill root-cause localization ([Lwd][eos-trace], silent)   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def zero_accept_rows(
        req_ids: list[str],
        accepted: list[int],
        finish_reasons: list[int],
    ) -> list[tuple[str, int, int]]:
        """Extract zero-accept rows (rid, accepted=0, finish_code) — the
        predicate data for the cloud row-assembly defect: a live row
        with zero acceptance (accepted=0 and finish=-1) is a kill
        candidate."""
        return [
            (rid, acc, fin)
            for rid, acc, fin in zip(req_ids, accepted, finish_reasons)
            if acc == 0
        ]

    @staticmethod
    def trace_cloud_c2e_rows(
        seqno: int,
        req_ids: list[str],
        accepted: list[int],
        finish_reasons: list[int],
    ) -> None:
        """[Lwd][eos-trace] Point A, cloud assembling c2e: leave a trace
        only when zero-accept rows exist (a normal MTP step accepts at
        least the root token per row, so zero-accept ~= 0 and the log
        volume stays bounded)."""
        rows = LwdControlDiag.zero_accept_rows(req_ids, accepted, finish_reasons)
        if rows:
            from vllm.lwd_logger import lwd_logger

            lwd_logger.info(
                "[eos-trace] cloud seqno=%s zero-accept rows (rid,acc,fin)=%s",
                seqno,
                rows,
            )

    @staticmethod
    def trace_edge_kill_row(
        seqno: int,
        rid: str,
        accepted: int,
        finish_code: int,
        batch_rows: int,
        sampled_rows: int,
    ) -> None:
        """[Lwd][eos-trace] Point B, edge converting to ERROR: pair with
        the A/C logs by down_seqno. accepted and sampled_rows are the
        primary evidence — accepted=0 -> cloud row assembly (a
        zero-accept live row entered the row set); accepted>0 and
        sampled_rows>0 -> a single worker row missing; sampled_rows=0
        -> the whole batch backfilled empty."""
        from vllm.lwd_logger import lwd_logger

        lwd_logger.info(
            "[eos-trace] edge kill seqno=%s rid=%s accepted=%d "
            "finish=%d batch_rows=%d sampled_rows=%d",
            seqno,
            rid,
            accepted,
            finish_code,
            batch_rows,
            sampled_rows,
        )

    @staticmethod
    def unembed_missing_rows(
        req_ids: list[str],
        accepts: list[int],
        result_req_ids: list[str],
        sampled_token_ids: list[list[int]],
    ) -> list[str]:
        """Worker backfill missing-row predicate: rows that declared
        accepted>0 yet produced no token output (absent from the
        backfill row set, or an empty token list) — Point C data. Rows
        with accepted=0 are legitimately not produced by the worker
        and do not count as missing."""
        filled = {rid: toks for rid, toks in zip(result_req_ids, sampled_token_ids)}
        return [
            rid for rid, acc in zip(req_ids, accepts) if acc > 0 and not filled.get(rid)
        ]

    @staticmethod
    def trace_worker_missing_rows(
        seqno: int,
        req_ids: list[str],
        accepts: list[int],
        result_req_ids: list[str],
        sampled_token_ids: list[list[int]],
    ) -> None:
        """[Lwd][eos-trace] Point C, after worker backfill: rows that
        declared output (accepted>0) yet delivered no tokens — their
        presence confirms the worker backfill dropped rows."""
        missing = LwdControlDiag.unembed_missing_rows(
            req_ids, accepts, result_req_ids, sampled_token_ids
        )
        if missing:
            from vllm.lwd_logger import lwd_logger

            lwd_logger.info(
                "[eos-trace] worker seqno=%s missing rows=%s",
                seqno,
                missing,
            )
