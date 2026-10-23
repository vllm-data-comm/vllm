# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LwdCloudDpAccount: the per-cloud-DP bookkeeping of the edge scheduler.

Pure logic, no engine and no scheduler: the checks cover the load
transitions, the pending-batch ledger, the gate predicate and the routing
score weights.
"""

from __future__ import annotations

from vllm.v1.lwd_control.control_edge_scheduler.lwd_cloud_dp_account import (
    LWD_MAX_PENDING_CHUNKS,
    LwdCloudDpAccount,
)


def test_load_transitions_track_one_request_lifecycle() -> None:
    account = LwdCloudDpAccount(0)
    assert account.load == 0 and account.score() == 0

    account.note_prefill_admitted()
    assert (account.prefill_load, account.decode_load) == (1, 0)
    assert account.score() == 4

    account.note_prefill_done()
    assert (account.prefill_load, account.decode_load) == (0, 1)
    assert account.score() == 1

    account.note_request_settled(was_awaiting=True)
    assert account.load == 0 and account.score() == 0


def test_settling_a_prefill_request_covers_abort_before_first_token() -> None:
    account = LwdCloudDpAccount(1)
    account.note_prefill_admitted()
    account.note_request_settled(was_awaiting=False)
    assert account.load == 0


def test_pending_batches_drop_on_first_token_and_finish() -> None:
    account = LwdCloudDpAccount(0)
    for seqno in range(3):
        account.register_batch(account.take_seqno(), {"r1", "r2"} if seqno < 2 else {"r3"})
    assert account.pending_chunks() == 3
    assert account.seqno == 3

    # A first token removes that request from every batch entry; entries
    # whose requests all received their first token leave the ledger.
    account.note_first_token("r1")
    assert account.pending_chunks() == 3  # batches still hold r2 / r3
    account.note_first_token("r2")
    assert account.pending_chunks() == 1  # both r1+r2 batches drained

    # A finish drops the request from its batch without touching the loads
    # (the caller settles the loads separately).
    account.drop_request("r3")
    assert account.pending_chunks() == 0


def test_gate_blocks_on_slots_and_pending_chunks_but_not_resumes() -> None:
    account = LwdCloudDpAccount(0)
    assert not account.is_gated(max_num_running_reqs=2, has_resume_inflight=False)

    # Slots full.
    account.note_prefill_admitted()
    account.note_prefill_admitted()
    assert account.is_gated(2, has_resume_inflight=False)
    # A resume in flight is exempt: finish in-flight chunks first.
    assert not account.is_gated(2, has_resume_inflight=True)

    # Pending chunks at the cap gate the engine even with free slots.
    other = LwdCloudDpAccount(1)
    for seqno in range(LWD_MAX_PENDING_CHUNKS):
        other.register_batch(seqno, {f"r{seqno}"})
    assert other.is_gated(max_num_running_reqs=100, has_resume_inflight=False)


def test_score_matches_native_dp_balancer_weights() -> None:
    account = LwdCloudDpAccount(0)
    account.prefill_load = 3
    account.decode_load = 2
    assert account.score() == 3 * 4 + 2 * 1
