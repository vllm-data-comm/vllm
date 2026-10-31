# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import LwdBatch, LwdBatchType, SchedulerOutput
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_engine import (
    LwdCloudEngineCoreMixin,
)
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_scheduler import (
    LwdCloudScheduler,
    _LwdPrefillWindow,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    LwdBaseScheduler,
)
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LwdAbortNotify,
    LwdRangeItem,
    LwdRangeNotify,
    LwdRequestNotify,
)
from vllm.v1.request import RequestStatus


class _Request(SimpleNamespace):
    __hash__ = object.__hash__


def _request(request_id: str = "p", prompt: int = 10, computed: int = 2):
    return _Request(
        request_id=request_id,
        num_prompt_tokens=prompt,
        num_computed_tokens=computed,
        is_finished=lambda: False,
        lwd_window_cap=None,
    )


def _visible_set_scheduler(running, waiting) -> LwdBaseScheduler:
    scheduler = LwdBaseScheduler.__new__(LwdBaseScheduler)
    scheduler.running = running
    scheduler.waiting = FCFSRequestQueue(waiting)
    scheduler.skipped_waiting = FCFSRequestQueue()
    scheduler.requests = {r.request_id: r for r in running + waiting}
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.max_num_running_reqs = len(running)
    scheduler.max_num_scheduled_tokens = 100
    scheduler.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0)
    return scheduler


def _scheduler(request) -> LwdCloudScheduler:
    scheduler = LwdCloudScheduler.__new__(LwdCloudScheduler)
    scheduler.running = [request]
    scheduler.waiting = []
    scheduler.skipped_waiting = []
    scheduler.requests = {request.request_id: request}
    scheduler.lwd_recv_rows = {}
    scheduler._lwd_dropped = set()
    scheduler._lwd_abort_pending = []
    return scheduler


def _engine(prompt_lens: dict[str, int] | None = None) -> LwdCloudEngineCoreMixin:
    engine = LwdCloudEngineCoreMixin.__new__(LwdCloudEngineCoreMixin)
    engine._lwd_prompt_lens = dict(prompt_lens or {})
    engine._lwd_retired_prompt_lens = {}
    engine._lwd_retired_req_ids = set()
    engine._lwd_pending_seqnos = {}
    engine._lwd_request_seqnos = {}
    engine._lwd_latest_seqno = -1
    engine.scheduler = SimpleNamespace(lwd_recv_rows={})
    return engine


def _allow_request_add(engine: LwdCloudEngineCoreMixin) -> None:
    engine.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_hidden_size=lambda: 8,
            dtype=torch.float32,
        )
    )
    engine.input_queue = queue.Queue()
    engine.engines_running = True


def test_fragment_waits_but_final_and_budget_limited_windows_run():
    request = _request()
    scheduler = _scheduler(request)
    scheduler.lwd_recv_rows["p"] = 5

    assert scheduler._lwd_collect_prefill_windows(room=8) == []

    budget_window = scheduler._lwd_collect_prefill_windows(room=2)
    assert [(w.start, w.num_rows) for w in budget_window] == [(2, 2)]

    scheduler.lwd_recv_rows["p"] = 10
    final_window = scheduler._lwd_collect_prefill_windows(room=8)
    assert [(w.start, w.num_rows) for w in final_window] == [(2, 8)]


def test_decode_exact_is_accounted_before_prefill_windows():
    scheduler = _scheduler(_request())
    scheduler.max_num_scheduled_tokens = 10
    calls = []
    out = SchedulerOutput.make_empty()
    scheduler._lwd_collect_decode_requests = lambda: calls.append("decode") or ["d"]
    scheduler._lwd_compute_decode_consumption = lambda ids: calls.append(ids) or 3
    scheduler._lwd_collect_prefill_windows = lambda room: calls.append(room) or []
    scheduler._lwd_schedule = lambda ids: out

    assert scheduler._lwd_schedule_step() is out
    assert calls == ["decode", ["d"], 7]


def test_window_cap_is_cleared_when_native_scheduling_fails():
    request = _request()
    scheduler = _scheduler(request)
    window = _LwdPrefillWindow(request=request, num_rows=4, start=2)

    def fail_schedule(request_ids, *, token_budget_cap=None):
        assert request_ids == ["d", "p"]
        assert token_budget_cap == 7
        assert request.lwd_window_cap == 4
        raise RuntimeError("native failure")

    scheduler._lwd_schedule = fail_schedule
    with pytest.raises(RuntimeError, match="native failure"):
        scheduler._lwd_dispatch_windows([window], ["d"], decode_exact=3)
    assert request.lwd_window_cap is None


def test_w2_rejects_unarrived_rows_and_allows_native_under_scheduling():
    request = _request()
    scheduler = _scheduler(request)
    scheduler.lwd_recv_rows["p"] = 5
    window = _LwdPrefillWindow(request=request, num_rows=3, start=2)
    out = SimpleNamespace(num_scheduled_tokens={"p": 4})

    with pytest.raises(RuntimeError, match="W2 violation"):
        scheduler._lwd_verify_w2(out, [window], decode_exact=1)

    out.num_scheduled_tokens["p"] = 1
    scheduler._lwd_verify_w2(out, [window], decode_exact=1)
    scheduler._lwd_warn_under_scheduled(out, [window])


def test_visible_set_restores_native_preempted_at_head_order(monkeypatch):
    hidden_running = _request("hidden-running")
    hidden_waiting = _request("hidden-waiting")
    p1 = _request("p1")
    rest = _request("rest")
    p2 = _request("p2")
    scheduler = _visible_set_scheduler([hidden_running, p1, p2], [hidden_waiting, rest])

    def native_schedule(self, throttle_prefills=False):
        self.running = []
        self.waiting = FCFSRequestQueue([p1, rest, p2])
        out = SchedulerOutput.make_empty()
        out.preempted_req_ids = {"p1", "p2"}
        return out

    monkeypatch.setattr(AsyncScheduler, "schedule", native_schedule)
    scheduler._lwd_schedule(["p1", "rest", "p2"])

    assert [r.request_id for r in scheduler.running] == ["hidden-running"]
    assert [r.request_id for r in scheduler.waiting] == [
        "p1",
        "p2",
        "hidden-waiting",
        "rest",
    ]


def test_visible_set_preserves_non_preempted_order_when_output_is_none(monkeypatch):
    hidden_running = _request("hidden-running")
    visible_running = _request("visible-running")
    hidden_waiting = _request("hidden-waiting")
    rest1 = _request("rest1")
    rest2 = _request("rest2")
    scheduler = _visible_set_scheduler(
        [hidden_running, visible_running], [hidden_waiting, rest1, rest2]
    )

    monkeypatch.setattr(AsyncScheduler, "schedule", lambda self: None)
    assert scheduler._lwd_schedule(["visible-running", "rest1", "rest2"]) is None
    assert [r.request_id for r in scheduler.running] == [
        "hidden-running",
        "visible-running",
    ]
    assert [r.request_id for r in scheduler.waiting] == [
        "hidden-waiting",
        "rest1",
        "rest2",
    ]


def test_preemption_recomputes_window_from_zero_and_clears_cap():
    request = _request(prompt=10, computed=4)
    scheduler = _scheduler(request)
    scheduler.max_num_scheduled_tokens = 10
    scheduler.lwd_recv_rows["p"] = 10
    calls = []

    def schedule(request_ids, *, token_budget_cap=None):
        calls.append(
            (request.num_computed_tokens, request.lwd_window_cap, token_budget_cap)
        )
        out = SchedulerOutput.make_empty()
        if len(calls) == 1:
            request.num_computed_tokens = 0
            out.preempted_req_ids = {"p"}
        else:
            out.num_scheduled_tokens["p"] = 10
        return out

    scheduler._lwd_schedule = schedule
    scheduler._lwd_schedule_step()
    assert request.num_computed_tokens == 0
    assert request.lwd_window_cap is None
    assert scheduler.lwd_recv_rows == {"p": 10}

    scheduler._lwd_schedule_step()
    assert calls == [(4, 6, 6), (0, 10, 10)]
    assert request.lwd_window_cap is None


def test_abort_records_drop_and_preserves_native_iterable(monkeypatch):
    request = _request()
    scheduler = _scheduler(request)
    received = []

    def native_finish(self, request_ids, finished_status):
        received.append((request_ids, finished_status))
        return []

    monkeypatch.setattr(LwdBaseScheduler, "finish_requests", native_finish)
    scheduler.finish_requests((rid for rid in ["p"]), RequestStatus.FINISHED_ABORTED)

    assert scheduler._lwd_dropped == {"p"}
    assert scheduler._lwd_abort_pending == ["p"]
    assert received == [(("p",), RequestStatus.FINISHED_ABORTED)]


def test_abort_only_embed_and_non_embed_retry_do_not_lose_notice():
    scheduler = _scheduler(_request())
    scheduler._lwd_abort_pending = ["p"]
    out = SchedulerOutput.make_empty()

    scheduler._lwd_attach_abort_notice(out)
    assert out.lwd_batch.seqno == -1
    assert out.lwd_batch.batch_type is LwdBatchType.LWD_EMBED
    assert out.lwd_batch.batch_meta.aborted_req_ids == ["p"]
    assert scheduler._lwd_abort_pending == []

    scheduler._lwd_abort_pending = ["q"]
    out.lwd_batch = LwdBatch(LwdBatchType.LWD_UNEMBED, 4, None)
    scheduler._lwd_attach_abort_notice(out)
    assert scheduler._lwd_abort_pending == ["q"]
    assert out.lwd_batch.batch_type is LwdBatchType.LWD_UNEMBED


def test_pending_abort_keeps_native_engine_alive(monkeypatch):
    scheduler = _scheduler(_request())
    scheduler._lwd_abort_pending = ["p"]
    monkeypatch.setattr(LwdBaseScheduler, "has_requests", lambda self: False)
    assert scheduler.has_requests()


def test_full_prompt_mrope_reaches_scheduled_new_request():
    request = _request()
    request.lwd_mrope_positions = b"positions"
    scheduler = _scheduler(request)
    data = SimpleNamespace(req_id="p", lwd_mrope_positions=None)
    out = SimpleNamespace(scheduled_new_reqs=[data])

    scheduler._lwd_attach_mrope_positions(out)
    assert data.lwd_mrope_positions == b"positions"


def test_request_build_records_full_prompt_mrope():
    engine = LwdCloudEngineCoreMixin.__new__(LwdCloudEngineCoreMixin)
    engine.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_hidden_size=lambda: 8,
            dtype=torch.float32,
        )
    )
    notify = LwdRequestNotify("p", 3, mrope_positions=b"positions")

    request = engine._lwd_build_request(notify)
    assert request.num_prompt_tokens == 3
    assert request.prompt_embeds.is_meta
    assert request.lwd_mrope_positions == b"positions"


def test_range_notify_posts_only_extended_hint(monkeypatch):
    from vllm_ascend.worker.lwd_cloud_model import lwd_pre_recv

    engine = _engine({"a": 8, "b": 6})
    posted = []
    monkeypatch.setattr(
        lwd_pre_recv,
        "make_recv_hint",
        lambda *args, **kwargs: (args, kwargs),
    )
    monkeypatch.setattr(lwd_pre_recv, "post_irecv_hint", posted.append)
    notify = LwdRangeNotify(
        7,
        [LwdRangeItem("a", 2, 3), LwdRangeItem("b", 1, 2)],
    )

    engine._lwd_handle_range_notify(notify)
    args, kwargs = posted[0]
    assert args == (7, 5, 0)
    assert kwargs == {
        "req_ids": ["a", "b"],
        "rows": [3, 2],
        "prompt_offsets": [2, 1],
        "prompt_lens": [8, 6],
    }


def test_range_notify_without_prompt_length_fails_fast():
    engine = _engine()
    notify = LwdRangeNotify(7, [LwdRangeItem("a", 0, 3)])

    with pytest.raises(RuntimeError, match="before RequestNotify"):
        engine._lwd_handle_range_notify(notify)


def test_done_arrivals_merge_by_max_and_keep_watermark_items(monkeypatch):
    from vllm_ascend.worker.lwd_cloud_model import lwd_pre_recv

    class DoneMq:
        def __init__(self):
            self.items = [("UP", 1), (("UP", 2), [("a", 4), ("b", 3)])]

        def dequeue(self, timeout):
            if self.items:
                return self.items.pop(0)
            raise TimeoutError

    engine = _engine()
    engine._lwd_done_mq = DoneMq()
    engine._lwd_track_range(1, ["a"])
    engine._lwd_track_range(2, ["a", "b"])
    watermark = []
    monkeypatch.setattr(
        lwd_pre_recv,
        "record_irecv_completions",
        lambda items: watermark.extend(items) or list(items),
    )

    engine._lwd_drain_irecv_completions()
    engine._lwd_track_range(3, ["b"])
    engine._lwd_note_arrived(3, [("a", 2), ("b", 5)])
    assert watermark == [("UP", 1), ("UP", 2)]
    assert engine.scheduler.lwd_recv_rows == {"a": 4, "b": 5}


def test_cloud_abort_uses_eager_and_ordered_native_queues():
    engine = _engine({"p": 4})
    engine.aborts_queue = queue.Queue()
    engine.input_queue = queue.Queue()

    engine._lwd_handle_abort_notify(LwdAbortNotify("p"))
    assert engine.aborts_queue.get_nowait() == ["p"]
    request_type, request_ids = engine.input_queue.get_nowait()
    assert request_type.name == "ABORT"
    assert request_ids == ["p"]
    assert engine._lwd_prompt_lens == {}


def test_abort_ignores_late_arrival_publish():
    engine = _engine({"p": 4})
    engine.scheduler.lwd_recv_rows["p"] = 2
    engine.aborts_queue = queue.Queue()
    engine.input_queue = queue.Queue()
    engine._lwd_track_range(7, ["p"])

    engine._lwd_handle_abort_notify(LwdAbortNotify("p"))
    engine._lwd_note_arrived(7, [("p", 4)])
    assert engine.scheduler.lwd_recv_rows == {}


def test_abort_duplicate_range_is_idempotent_after_prompt_cleanup():
    engine = _engine({"p": 4})
    engine.aborts_queue = queue.Queue()
    engine.input_queue = queue.Queue()
    engine._lwd_track_range(7, ["p"])
    engine._lwd_handle_abort_notify(LwdAbortNotify("p"))
    engine._lwd_complete_seqno(7)
    assert engine._lwd_prompt_lens == {}

    notify = LwdRangeNotify(7, [LwdRangeItem("p", 0, 4)])
    engine._lwd_handle_range_notify(notify)
    assert engine._lwd_pending_seqnos == {}
    assert engine._lwd_latest_seqno == 7


def test_normal_finish_retires_from_outputs_and_cleans_after_drain():
    engine = _engine({"p": 4})
    engine.scheduler.requests = {}
    engine._lwd_track_range(8, ["p"])
    outputs = {0: SimpleNamespace(finished_requests=["p"], outputs=[])}

    engine._lwd_retire_finished_outputs(outputs)
    assert engine._lwd_retired_req_ids == {"p"}
    assert engine._lwd_prompt_lens == {"p": 4}

    engine._lwd_complete_seqno(8)
    assert engine._lwd_prompt_lens == {}
    assert engine._lwd_pending_seqnos == {}
    assert engine._lwd_request_seqnos == {}


def test_abort_before_mixed_range_still_posts_complete_hint(monkeypatch):
    from vllm_ascend.worker.lwd_cloud_model import lwd_pre_recv

    engine = _engine({"a": 8, "b": 6})
    engine.aborts_queue = queue.Queue()
    engine.input_queue = queue.Queue()
    engine._lwd_track_range(6, ["a"])
    engine._lwd_handle_abort_notify(LwdAbortNotify("a"))
    posted = []
    monkeypatch.setattr(
        lwd_pre_recv,
        "make_recv_hint",
        lambda *args, **kwargs: (args, kwargs),
    )
    monkeypatch.setattr(lwd_pre_recv, "post_irecv_hint", posted.append)
    notify = LwdRangeNotify(
        7,
        [LwdRangeItem("a", 2, 3), LwdRangeItem("b", 1, 2)],
    )

    engine._lwd_handle_range_notify(notify)
    args, kwargs = posted[0]
    assert args == (7, 5, 0)
    assert kwargs["req_ids"] == ["a", "b"]
    assert kwargs["prompt_lens"] == [8, 6]


@pytest.mark.parametrize("retire_kind", ["finish", "abort"])
def test_drained_retirement_allows_same_id_reuse(retire_kind):
    engine = _engine({"p": 4})
    engine._lwd_track_range(1, ["p"])
    if retire_kind == "abort":
        engine.aborts_queue = queue.Queue()
        engine.input_queue = queue.Queue()
        engine._lwd_handle_abort_notify(LwdAbortNotify("p"))
    else:
        outputs = {0: SimpleNamespace(finished_requests=["p"], outputs=[])}
        engine._lwd_retire_finished_outputs(outputs)
    assert engine._lwd_retired_req_ids == {"p"}

    engine._lwd_complete_seqno(1)
    assert engine._lwd_retired_req_ids == set()
    assert engine._lwd_retired_prompt_lens == {}
    _allow_request_add(engine)
    engine._lwd_handle_request_notify(LwdRequestNotify("p", 4))
    request_type, (request, _wave) = engine.input_queue.get_nowait()
    assert request_type.name == "ADD" and request.request_id == "p"

    engine._lwd_track_range(2, ["p"])
    engine._lwd_note_arrived(2, [("p", 4)])
    engine._lwd_note_arrived(1, [("p", 99)])
    assert engine.scheduler.lwd_recv_rows == {"p": 4}


def test_request_id_reuse_fails_while_previous_generation_drains():
    engine = _engine({"p": 4})
    engine._lwd_track_range(1, ["p"])
    engine._lwd_retire_request("p", preserve_prompt=True)
    _allow_request_add(engine)

    with pytest.raises(RuntimeError, match="previous generation"):
        engine._lwd_handle_request_notify(LwdRequestNotify("p", 4))
    assert engine.input_queue.empty()


def test_retired_state_is_bounded_after_drain():
    engine = _engine()
    for seqno in range(5):
        engine._lwd_prompt_lens["p"] = 4
        engine._lwd_track_range(seqno, ["p"])
        engine._lwd_retire_request("p", preserve_prompt=True)
        assert engine._lwd_retired_req_ids == {"p"}
        engine._lwd_complete_seqno(seqno)
        assert engine._lwd_retired_req_ids == set()
        assert engine._lwd_retired_prompt_lens == {}
        assert engine._lwd_request_seqnos == {}
        assert engine._lwd_pending_seqnos == {}


def test_finish_without_pending_clears_retired_state_immediately():
    engine = _engine({"p": 4})
    outputs = {0: SimpleNamespace(finished_requests=["p"], outputs=[])}

    engine._lwd_retire_finished_outputs(outputs)
    assert engine._lwd_retired_req_ids == set()
    assert engine._lwd_retired_prompt_lens == {}
    assert engine._lwd_prompt_lens == {}


def test_worker_error_fails_engine_without_advancing_watermark(monkeypatch):
    from vllm_ascend.worker.lwd_cloud_model import lwd_pre_recv

    class DoneMq:
        def __init__(self):
            self.items = [("error", 9, "boom"), ("UP", 10)]

        def dequeue(self, timeout):
            if self.items:
                return self.items.pop(0)
            raise TimeoutError

    engine = _engine()
    engine._lwd_done_mq = DoneMq()
    failures = []
    engine._lwd_fail = failures.append
    watermark = []
    monkeypatch.setattr(
        lwd_pre_recv,
        "record_irecv_completions",
        lambda items: watermark.extend(items),
    )

    engine._lwd_drain_irecv_completions()
    assert len(failures) == 1
    assert "seqno=9" in failures[0] and "boom" in failures[0]
    assert watermark == []


def test_prefix_caching_is_rejected_before_engine_startup():
    config = SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=True))
    with pytest.raises(ValueError, match="prefix caching is unsupported"):
        LwdCloudEngineCoreMixin(vllm_config=config, lwd_dp_size=1)
