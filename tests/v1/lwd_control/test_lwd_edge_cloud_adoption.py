# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Elastic partial start: a cloud that starts after the edge is adopted
when its scheduling limits arrive on the control-plane IO thread.

The edge's startup barrier returns the accepted subset (see
``test_lwd_edge_hello_reregister.py``); the missing cloud's handshake
keeps being retried by the periodic HELLO, and its ``LwdCloudParamsNotify``
is the first runtime traffic that adopts it: sendable peer, liveness
account seed, probe hash sizes — the same routine the init path uses.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.config.lwd.topology import LwdConnectionId
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine import (
    LwdEdgeEngineCore,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_heartbeat import (
    LwdHeartbeatMonitor,
)
from vllm.v1.lwd_control.lwd_comm.lwd_messages import LwdCloudParamsNotify

CLOUD_0 = LwdConnectionId(0, 0, 0)
CLOUD_1 = LwdConnectionId(0, 1, 0)


def _engine() -> Any:
    engine: Any = object.__new__(LwdEdgeEngineCore)
    engine.lwd_config = SimpleNamespace(
        connections=[
            SimpleNamespace(cloud_id=0, dp_idx=0, connection_id=CLOUD_0),
            SimpleNamespace(cloud_id=1, dp_idx=0, connection_id=CLOUD_1),
        ]
    )
    engine.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192, max_num_seqs=64)
    )
    engine._lwd_heartbeat = LwdHeartbeatMonitor(10.0, 1.0)
    engine._lwd_heartbeat.note_adopted(CLOUD_0)
    engine._lwd_runtime_connections = {}
    # Cloud 0 is the accepted startup subset; cloud 1 starts later.
    engine._lwd_clouds = {CLOUD_0: SimpleNamespace(instance_id=0, dp_rank=0)}
    engine._lwd_cloud_params = {}
    engine._lwd_cloud_params_received = threading.Event()
    engine._lwd_coord_clients = {}
    engine._lwd_probe_manager = SimpleNamespace(_hash_sizes={})
    engine.input_queue = SimpleNamespace(put_nowait=lambda item: None)
    return engine


def _params() -> LwdCloudParamsNotify:
    return LwdCloudParamsNotify(
        max_num_batched_tokens=8192,
        max_num_seqs=64,
        num_speculative_tokens=0,
        prefix_hash_block_size=128,
    )


def test_late_cloud_params_adopt_the_connection() -> None:
    engine = _engine()
    updates: list = []
    engine.scheduler = SimpleNamespace(
        lwd_update_cloud_params=lambda connection, params: updates.append(
            (connection, params)
        )
    )

    peer = SimpleNamespace(instance_id=1, dp_rank=0)
    msg = _params()
    assert engine._lwd_dispatch_message(peer, msg)

    assert CLOUD_1 in engine._lwd_clouds  # sendable peer adopted
    assert engine._lwd_cloud_params[CLOUD_1] is msg
    assert engine._lwd_coord_clients == {}  # no coordinator addr -> no client
    assert engine._lwd_probe_manager._hash_sizes[(1, 0)] == msg.prefix_hash_block_size
    assert CLOUD_1 in engine._lwd_heartbeat._hello_last  # liveness seeded
    assert engine._lwd_cloud_params_received.is_set()
    assert updates == [(CLOUD_1, msg)]

    # A re-arrival is a replacement signal: it queues a ledger-reset tick
    # on the engine thread while keeping the adopted state intact.
    queued: list = []
    engine.input_queue = SimpleNamespace(put_nowait=queued.append)
    assert engine._lwd_dispatch_message(peer, msg)
    assert len(queued) == 1
    assert len(engine._lwd_clouds) == 2


def test_config_mismatch_on_adoption_fails_loudly() -> None:
    """The runtime adoption runs the same limits check as init: a late
    cloud with mismatched scheduling limits is a deployment error."""
    engine = _engine()
    engine.scheduler = SimpleNamespace(
        lwd_update_cloud_params=lambda connection, params: None
    )
    peer = SimpleNamespace(instance_id=1, dp_rank=0)
    bad = _params()
    bad.max_num_batched_tokens = 4096
    try:
        engine._lwd_dispatch_message(peer, bad)
    except RuntimeError as exc:
        assert "mismatch" in str(exc)
    else:  # pragma: no cover - the check must raise
        raise AssertionError("mismatched limits were accepted on adoption")
    # The connection was not half-adopted.
    assert CLOUD_1 not in engine._lwd_clouds


def test_cloud_replacement_resets_edge_seqno_ledger() -> None:
    """A replaced cloud starts with an empty prompt-length ledger, so the
    edge's seqno stream must restart from zero and every booking of the
    dead incarnation must be dropped. The first RangeNotify of the new
    stream would otherwise carry a request id the cloud never saw and its
    fail-fast ingest would kill the fresh EngineCore.

    Regression for: send a request while the cloud is down, let it fail,
    then restart the cloud -> ``RangeNotify seqno=N arrived before
    RequestNotify prompt lengths were recorded``.
    """
    from vllm.v1.lwd_control.control_edge_scheduler.lwd_cloud_dp_account import (
        LwdCloudDpAccount,
    )
    from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
        LwdEdgeScheduler,
    )

    scheduler: Any = object.__new__(LwdEdgeScheduler)
    cloud0 = LwdCloudDpAccount(CLOUD_0)
    cloud0.take_seqno()
    cloud0.take_seqno()
    cloud0.register_batch(2, {"chatcmpl-x"})
    cloud1 = LwdCloudDpAccount(CLOUD_1)
    cloud1.take_seqno()
    scheduler._lwd_accounts = {CLOUD_0: cloud0, CLOUD_1: cloud1}
    scheduler._lwd_announced_refills = {(CLOUD_0, 2), (CLOUD_1, 7)}
    scheduler._lwd_route = {
        "chatcmpl-x": CLOUD_0,
        "chatcmpl-y": CLOUD_1,
    }
    scheduler._lwd_aborted_req_ids = {CLOUD_0: ["chatcmpl-z"], CLOUD_1: []}

    orphaned = scheduler.lwd_edge_reset_connection(CLOUD_0)

    assert cloud0.seqno == 0
    assert cloud0.pending_batches == {}
    assert scheduler._lwd_announced_refills == {(CLOUD_1, 7)}
    assert scheduler._lwd_route == {"chatcmpl-y": CLOUD_1}
    assert scheduler._lwd_aborted_req_ids[CLOUD_0] == []
    assert orphaned == ["chatcmpl-x"]
    # A different cloud DP's ledger is untouched.
    assert cloud1.seqno == 1
    assert scheduler._lwd_aborted_req_ids[CLOUD_1] == []

    # Idempotent: a second registration (e.g. a link blip) finds nothing.
    assert scheduler.lwd_edge_reset_connection(CLOUD_0) == []
    assert cloud0.seqno == 0


def test_repeated_cloud_params_queues_ledger_reset() -> None:
    """A replaced cloud re-sends its scheduling limits. The first arrival
    only adopts; a repeat is the observable replacement signal and queues a
    WAKEUP reset tick on the engine thread.

    The peer-up hook is NOT a replacement signal on the edge: the edge
    control plane never unregisters a stopped cloud, so a re-registration
    is skipped and no peer-up event fires.
    """
    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine import (
        _LwdCloudRestartTick,
    )

    engine = _engine()
    queued: list = []
    engine.input_queue = SimpleNamespace(put_nowait=queued.append)
    engine.scheduler = SimpleNamespace(
        lwd_update_cloud_params=lambda connection, params: None
    )
    peer = SimpleNamespace(instance_id=0, dp_rank=0)

    assert engine._lwd_dispatch_message(peer, _params())
    assert queued == []  # startup adoption is not a replacement
    assert engine._lwd_dispatch_message(peer, _params())
    assert len(queued) == 1
    request_type, tick = queued[0]
    assert request_type == EngineCoreRequestType.WAKEUP
    assert isinstance(tick, _LwdCloudRestartTick)
    assert tick.connection == CLOUD_0


def test_restart_tick_resets_ledger_and_finishes_orphans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The WAKEUP tick runs on the engine thread (single writer) and settles
    requests still routed to the dead incarnation with an explicit ERROR,
    not a native ABORT/cancelled completion."""
    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine import (
        _LwdCloudRestartTick,
    )

    engine = _engine()
    errors: list[tuple[list[str], int]] = []
    finished: list[tuple[list[str], object]] = []
    monkeypatch.setattr(
        engine,
        "_send_error_outputs_to_client",
        lambda ids, client_index: errors.append((ids, client_index)),
    )
    engine.scheduler = SimpleNamespace(
        lwd_edge_reset_connection=lambda connection: ["chatcmpl-x"],
        requests={
            "chatcmpl-x": SimpleNamespace(client_index=0, is_finished=lambda: False)
        },
        finish_requests=lambda ids, status: finished.append((ids, status)),
    )
    engine._handle_client_request(
        EngineCoreRequestType.WAKEUP, _LwdCloudRestartTick(CLOUD_0)
    )
    assert errors == [(["chatcmpl-x"], 0)]
    assert len(finished) == 1 and finished[0][0] == ["chatcmpl-x"]


def test_restart_tick_with_empty_ledger_finishes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replacement with an already-empty ledger finishes no request and
    emits no ERROR (the reset must be idempotent)."""
    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine import (
        _LwdCloudRestartTick,
    )

    engine = _engine()
    errors: list = []
    monkeypatch.setattr(
        engine,
        "_send_error_outputs_to_client",
        lambda ids, client_index: errors.append(ids),
    )
    engine.scheduler = SimpleNamespace(
        lwd_edge_reset_connection=lambda connection: [],
        requests={},
        finish_requests=lambda ids, status: None,
    )
    engine._handle_client_request(
        EngineCoreRequestType.WAKEUP, _LwdCloudRestartTick(CLOUD_0)
    )
    assert errors == []
