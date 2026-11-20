# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end flow of one LWD edge-cloud pair on CPU, with Qwen3-0.6B.

Real: config and topology parsing, both engines and schedulers, the control
plane, the shared world and its data-plane groups, and vllm-ascend's
early-receive sideband. Simulated (`lwd_sim_workers`): the workers, with gloo
in place of HCCL and no model forward. The tests check the order of the
control-plane and data-plane interactions, not tensor contents.

Needs Linux and `VLLM_LWD_ASCEND_PATH` naming a vllm-ascend source tree.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

from tests.v1.lwd_control.e2e.lwd_sim_cluster import (
    LWD_ASCEND_PATH_ENV,
    LwdSimLimits,
    LwdSimRequest,
    LwdSimRun,
    lwd_ascend_path,
    lwd_run_sim,
)
from tests.v1.lwd_control.e2e.lwd_sim_model import (
    LWD_SIM_HIDDEN_SIZE,
    lwd_sim_token_rank,
)
from tests.v1.lwd_control.e2e.lwd_sim_trace import LwdSimEvent
from vllm.v1.engine import FinishReason
from vllm.v1.lwd_control.lwd_comm.lwd_messages import LWD_NOT_FINISHED

pytestmark = [
    pytest.mark.skipif(
        not sys.platform.startswith("linux"), reason="needs Linux shared memory"
    ),
    pytest.mark.skipif(
        lwd_ascend_path() is None,
        reason=f"set {LWD_ASCEND_PATH_ENV} to a vllm-ascend source tree",
    ),
]

_H = LWD_SIM_HIDDEN_SIZE
_LENGTH = int(FinishReason.LENGTH)

RangeItem = tuple[str, int, int]
"""(request_id, offset, num_tokens) of one chunk."""


def test_single_request_flow(tmp_path: Path) -> None:
    """One request over three chunks: the flow is deterministic, so both link
    directions and the cloud steps are compared with their exact sequence and
    the data plane is checked against them."""
    request = LwdSimRequest(prompt_len=600, max_tokens=3)
    run = lwd_run_sim(tmp_path, [request])
    _assert_succeeded(run)
    trace = run.trace
    (announced,) = trace.payloads("e2c", "LwdRequestNotify")
    rid = announced["request_id"]

    assert [_summary(p) for p in trace.payloads("e2c")] == [
        ("LwdRequestNotify", rid, 600, 3),
        # From the second chunk on, the in-flight request holds one decode
        # slot of its own step budget (cap = 256 - inflight * spec_factor),
        # so the second chunk is one token short.
        ("LwdRangeNotify", 0, [(rid, 0, 256)]),
        ("LwdRangeNotify", 1, [(rid, 256, 255)]),
        ("LwdRangeNotify", 2, [(rid, 511, 89)]),
    ], run.describe()
    assert [_summary(p) for p in trace.payloads("c2e")] == [
        ("LwdCloudParamsNotify", 256, 4, 0),
        *[
            ("LwdC2eNotify", m, [rid], _H, [1], [finish], [[lwd_sim_token_rank(m, 0)]])
            for m, finish in enumerate([LWD_NOT_FINISHED, LWD_NOT_FINISHED, _LENGTH])
        ],
    ], run.describe()
    steps = [
        (step["embed"]["seqno"] if step["embed"] else None, step["scheduled"])
        for step in trace.select("cloud-worker", "step")
    ]
    assert steps == [
        (0, {rid: 256}),
        (1, {rid: 255}),
        (2, {rid: 89}),
        (None, {rid: 1}),
        (None, {rid: 1}),
    ], run.describe()
    _assert_link_setup(run)
    _assert_up_chain(run)
    _assert_down_chain(run)
    _assert_request_lifecycles(run, [request])


def test_concurrent_requests_flow(tmp_path: Path) -> None:
    """Requests of different lengths batched and mixed with decodes: timing
    decides the batches, so the invariants of every interaction are checked."""
    requests = [
        LwdSimRequest(prompt_len=700, max_tokens=4),
        LwdSimRequest(prompt_len=300, max_tokens=3),
        LwdSimRequest(prompt_len=40, max_tokens=5),
        LwdSimRequest(prompt_len=17, max_tokens=2),
    ]
    run = lwd_run_sim(tmp_path, requests)
    _assert_succeeded(run)
    _assert_link_setup(run)
    _assert_up_chain(run)
    _assert_down_chain(run)
    _assert_request_lifecycles(run, requests)


def test_limit_mismatch_fails_edge_startup(tmp_path: Path) -> None:
    """An edge whose scheduling limits differ from the cloud's fails at
    startup, after the handshake and before announcing any request; the cloud
    keeps serving."""
    run = lwd_run_sim(
        tmp_path,
        [LwdSimRequest(prompt_len=16, max_tokens=1)],
        edge_limits=LwdSimLimits(max_num_seqs=2),
    )
    # The edge engine fails fast on the mismatch; the frontend has no
    # failure callback after an engine dies at startup (core_client's
    # monitor only sets engine_dead), so the driver may stay parked with
    # the failure logged instead of exiting nonzero.
    assert run.returncodes["edge"] not in (
        0,
        None,
    ) or "edge/cloud scheduling config mismatch" in run.log("edge"), run.describe()
    assert run.cloud_alive_after_edge, run.describe()
    assert run.returncodes["cloud"] == 0, run.describe()
    trace = run.trace
    _assert_link_setup(run)
    assert [_summary(p) for p in trace.payloads("c2e")] == [
        ("LwdCloudParamsNotify", 256, 4, 0)
    ], run.describe()
    assert trace.payloads("e2c") == [], run.describe()
    assert trace.select("edge-worker", "up_send") == [], run.describe()


def _assert_succeeded(run: LwdSimRun) -> None:
    assert run.returncodes == {"edge": 0, "cloud": 0}, run.describe()


def _assert_before(earlier: LwdSimEvent, later: LwdSimEvent, run: LwdSimRun) -> None:
    assert earlier.t < later.t, (
        f"expected {earlier.src}/{earlier.event} {earlier.fields} before "
        f"{later.src}/{later.event} {later.fields}\n{run.describe()}"
    )


def _assert_link_setup(run: LwdSimRun) -> None:
    """The edge registers, the cloud accepts and sends its limits before any
    other payload, and the edge announces nothing before the limits."""
    trace = run.trace
    hello = trace.frames("e2c", "link")[0]
    accept = trace.frames("c2e", "link")[0]
    payloads = trace.payloads("c2e")
    assert (hello["msg"], accept["msg"]) == ("LwdLinkHello", "LwdLinkAccept")
    assert payloads[0]["msg"] == "LwdCloudParamsNotify", run.describe()
    _assert_before(hello, accept, run)
    _assert_before(accept, payloads[0], run)
    announced = trace.payloads("e2c")
    if announced:
        _assert_before(payloads[0], announced[0], run)
    payload_frames = [e for e in trace.select("tap", "frame") if e["kind"] == "payload"]
    assert all(e["delivered"] for e in payload_frames), run.describe()


def _assert_up_chain(run: LwdSimRun) -> None:
    """Range notify k pairs with UP transfer k on both sides: the edge sends
    the batch's tokens, the cloud posts the receive once the notify arrives
    and dispatches exactly those chunks only after the receive completed (the
    early-receive gate)."""
    trace = run.trace
    notifies = trace.payloads("e2c", "LwdRangeNotify")
    chains = {
        "up_send": trace.select("edge-worker", "up_send"),
        "up_recv_post": trace.select("cloud-worker", "up_recv_post"),
        "up_recv_done": trace.select("cloud-worker", "up_recv_done"),
        "up_consume": trace.select("cloud-worker", "up_consume"),
    }
    expected_seqnos = list(range(len(notifies)))
    assert [n["seqno"] for n in notifies] == expected_seqnos, run.describe()
    for name, events in chains.items():
        assert [e["seqno"] for e in events] == expected_seqnos, (name, run.describe())
    embed_steps = [s for s in trace.select("cloud-worker", "step") if s["embed"]]
    assert [s["embed"]["seqno"] for s in embed_steps] == expected_seqnos
    for notify, send, post, done, consume, step in zip(
        notifies, *chains.values(), embed_steps
    ):
        items = _range_items(notify)
        numel = sum(num_tokens for _, _, num_tokens in items) * _H
        assert _batch_items(send) == items, run.describe()
        assert send["numel"] == post["numel"] == numel, run.describe()
        assert done["stamp"] == notify["seqno"], "UP send/receive mispaired"
        assert consume["reported"], "batch dispatched before its UP data arrived"
        assert _batch_items(step["embed"]) == items, run.describe()
        assert all(step["scheduled"][rid] == n for rid, _, n in items)
        _assert_before(notify, post, run)
        _assert_before(send, done, run)
        _assert_before(done, consume, run)


def _assert_down_chain(run: LwdSimRun) -> None:
    """Each cloud step that samples sends DOWN transfer m and C2e notify m with
    the worker's rows and ranks; the edge receives m with the announced size
    after the notify and unembeds exactly those rows."""
    trace = run.trace
    notifies = trace.payloads("c2e", "LwdC2eNotify")
    chains = {
        "down_send": trace.select("cloud-worker", "down_send"),
        "down_recv_post": trace.select("edge-worker", "down_recv_post"),
        "down_recv_done": trace.select("edge-worker", "down_recv_done"),
    }
    expected_seqnos = list(range(len(notifies)))
    assert [n["down_seqno"] for n in notifies] == expected_seqnos, run.describe()
    for name, events in chains.items():
        assert [e["seqno"] for e in events] == expected_seqnos, (name, run.describe())
    sampling_steps = [s for s in trace.select("cloud-worker", "step") if s["sampled"]]
    assert len(sampling_steps) == len(notifies), run.describe()
    for notify, send, post, done, step in zip(
        notifies, *chains.values(), sampling_steps
    ):
        req_ids = notify["req_ids"]
        assert send["req_ids"] == req_ids == post["req_ids"] == step["sampled"]
        assert (
            send["numel"]
            == notify["hidden_num_elements"]
            == post["numel"]
            == len(req_ids) * _H
        ), run.describe()
        ranks = [[rank] for rank in send["ranks"]]
        assert notify["top_id_ths"] == post["top_id_ths"] == ranks, run.describe()
        assert post["num_accept_tokens"] == notify["num_accepted_tokens"]
        assert done["stamp"] == notify["down_seqno"], "DOWN send/receive mispaired"
        _assert_before(send, notify, run)
        _assert_before(notify, post, run)


def _assert_request_lifecycles(run: LwdSimRun, requests: list[LwdSimRequest]) -> None:
    """Per request: announced before its first chunk, chunks contiguous over
    the whole prompt, the first token only after the last chunk, `max_tokens`
    tokens with LENGTH on the last row, delivered by the edge."""
    trace = run.trace
    announced = {
        n["num_prompt_tokens"]: n for n in trace.payloads("e2c", "LwdRequestNotify")
    }
    assert sorted(announced) == sorted(r.prompt_len for r in requests)
    ranges = trace.payloads("e2c", "LwdRangeNotify")
    c2es = trace.payloads("c2e", "LwdC2eNotify")
    for request in requests:
        notify = announced[request.prompt_len]
        rid = notify["request_id"]
        assert notify["max_tokens"] == request.max_tokens
        chunks = [
            (r, item) for r in ranges for item in _range_items(r) if item[0] == rid
        ]
        assert [offset for _, (_, offset, _) in chunks] == _prefix_sums(
            [n for _, (_, _, n) in chunks]
        ), (rid, run.describe())
        assert sum(n for _, (_, _, n) in chunks) == request.prompt_len
        rows = [(c, c["req_ids"].index(rid)) for c in c2es if rid in c["req_ids"]]
        assert sum(c["num_accepted_tokens"][i] for c, i in rows) == request.max_tokens
        finishes = [c["finish_reasons"][i] for c, i in rows]
        assert finishes == [LWD_NOT_FINISHED] * (len(rows) - 1) + [_LENGTH]
        _assert_before(notify, chunks[0][0], run)
        _assert_before(chunks[-1][0], rows[0][0], run)
    assert sorted(run.edge_results, key=lambda r: r["prompt_len"]) == [
        {
            "prompt_len": r.prompt_len,
            "num_tokens": r.max_tokens,
            "finish_reason": "length",
        }
        for r in sorted(requests, key=lambda r: r.prompt_len)
    ], run.describe()


def _summary(payload: LwdSimEvent) -> tuple[Any, ...]:
    """The fields of a control-link payload that the flow depends on."""
    msg = payload["msg"]
    if msg == "LwdRequestNotify":
        return (
            msg,
            payload["request_id"],
            payload["num_prompt_tokens"],
            payload["max_tokens"],
        )
    if msg == "LwdRangeNotify":
        return msg, payload["seqno"], _range_items(payload)
    if msg == "LwdCloudParamsNotify":
        return (
            msg,
            payload["max_num_batched_tokens"],
            payload["max_num_seqs"],
            payload["num_speculative_tokens"],
        )
    assert msg == "LwdC2eNotify", msg
    return (
        msg,
        payload["down_seqno"],
        payload["req_ids"],
        payload["hidden_num_elements"],
        payload["num_accepted_tokens"],
        payload["finish_reasons"],
        payload["top_id_ths"],
    )


def _range_items(notify: LwdSimEvent) -> list[RangeItem]:
    return [
        (item["request_id"], item["offset"], item["num_tokens"])
        for item in notify["items"]
    ]


def _batch_items(batch: LwdSimEvent | dict[str, Any]) -> list[RangeItem]:
    """The chunks of a worker-side EMBED batch record."""
    return list(zip(batch["req_ids"], batch["offsets"], batch["num_tokens"]))


def _prefix_sums(values: list[int]) -> list[int]:
    sums, total = [], 0
    for value in values:
        sums.append(total)
        total += value
    return sums
