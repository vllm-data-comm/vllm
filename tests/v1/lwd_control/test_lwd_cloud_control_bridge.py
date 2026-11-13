# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavioral tests for the cloud control bridge's queue boundaries."""

from __future__ import annotations

import asyncio
import contextlib
import queue
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
import pytest_asyncio

from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_control import (
    LwdCloudControlBridge,
)
from vllm.v1.lwd_control.lwd_comm.lwd_prefix import (
    LwdPrefixManifest,
    LwdProbeResult,
    LwdUsage,
)

BridgeFactory = Callable[
    [int],
    tuple[
        LwdCloudControlBridge,
        list[queue.Queue[Any]],
        queue.Queue[Any],
    ],
]


@pytest_asyncio.fixture
async def bridge_factory() -> AsyncIterator[BridgeFactory]:
    loop = asyncio.get_running_loop()
    loop.set_default_executor(
        ThreadPoolExecutor(max_workers=4, thread_name_prefix="lwd-test")
    )
    bridges: list[LwdCloudControlBridge] = []

    def make_bridge(
        engine_count: int,
    ) -> tuple[
        LwdCloudControlBridge,
        list[queue.Queue[Any]],
        queue.Queue[Any],
    ]:
        command_queues = [queue.Queue() for _ in range(engine_count)]
        event_queue: queue.Queue[Any] = queue.Queue()
        bridge = LwdCloudControlBridge(command_queues, event_queue)
        bridge.start()
        bridges.append(bridge)
        return bridge, command_queues, event_queue

    yield make_bridge

    for bridge in bridges:
        bridge.close()
        # Also release an executor wait if a failing test exits mid-fanout.
        with bridge._lock:
            waiters = list(bridge._fanout_waiters.values())
        for waiter in waiters:
            with contextlib.suppress(queue.Full):
                waiter.put_nowait((False, -1, "test cleanup"))
        bridge._dispatch_thread.join(timeout=1)
        assert not bridge._dispatch_thread.is_alive()

    current_task = asyncio.current_task()
    pending_tasks = [
        task
        for task in asyncio.all_tasks()
        if task is not current_task and not task.done()
    ]
    for task in pending_tasks:
        task.cancel()
    await asyncio.gather(*pending_tasks, return_exceptions=True)
    await loop.shutdown_default_executor()


def _manifest(request_id: str) -> LwdPrefixManifest:
    return LwdPrefixManifest(
        request_id=request_id,
        prompt_tokens=16,
        block_size=16,
        full_block_hashes=(bytes(range(32)),),
    )


def _result(request_id: str, hit_blocks: int, engine_index: int = 0) -> LwdProbeResult:
    return LwdProbeResult(
        request_id=request_id,
        instance_id="cloud-test",
        block_size=16,
        hit_blocks=hit_blocks,
        hit_tokens=hit_blocks * 16,
        engine_index=engine_index,
    )


async def _start_probe(
    bridge: LwdCloudControlBridge,
    command_queues: list[queue.Queue[Any]],
    request_id: str,
    timeout: float = 1.0,
) -> tuple[asyncio.Task[LwdProbeResult], list[dict[str, Any]]]:
    task = asyncio.create_task(bridge.probe(_manifest(request_id), timeout=timeout))
    await asyncio.sleep(0)
    commands = [command_queue.get_nowait() for command_queue in command_queues]
    assert all(command["type"] == "probe" for command in commands)
    return task, commands


def _reply(
    event_queue: queue.Queue[Any],
    request_id: str,
    engine_index: int,
    result: LwdProbeResult | None = None,
    error: str | None = None,
) -> None:
    event_queue.put(
        {
            "type": "probe",
            "request_id": request_id,
            "engine_index": engine_index,
            "ok": result is not None,
            "result": result,
            "error": error,
        }
    )


@pytest.mark.asyncio
async def test_partial_engine_failure_chooses_survivor(
    bridge_factory: BridgeFactory,
) -> None:
    bridge, command_queues, event_queue = bridge_factory(2)
    probe_task, commands = await _start_probe(
        bridge, command_queues, "partial", timeout=1.0
    )
    assert [command["reserve"] for command in commands] == [False, False]

    _reply(event_queue, "partial#probe", 0, error="engine unavailable")
    _reply(event_queue, "partial#probe", 1, _result("partial", 2))

    result = await asyncio.wait_for(probe_task, timeout=1)
    assert result.engine_index == 1
    assert result.hit_tokens == 32
    bridge.release("partial")


@pytest.mark.asyncio
async def test_equal_hits_rotate_across_engines(
    bridge_factory: BridgeFactory,
) -> None:
    bridge, command_queues, event_queue = bridge_factory(2)

    winners = []
    for attempt in range(3):
        request_id = f"tie-{attempt}"
        probe_task, _ = await _start_probe(bridge, command_queues, request_id)
        for engine_index in range(2):
            _reply(
                event_queue,
                f"{request_id}#probe",
                engine_index,
                _result(request_id, 1, engine_index),
            )
        result = await asyncio.wait_for(probe_task, timeout=1)
        winners.append(result.engine_index)
        bridge.release(request_id)

    assert winners == [0, 1, 0]


@pytest.mark.asyncio
async def test_total_timeout_cleans_waiters_and_drops_late_reply(
    bridge_factory: BridgeFactory,
) -> None:
    bridge, command_queues, event_queue = bridge_factory(2)
    probe_task, _ = await _start_probe(bridge, command_queues, "timeout", timeout=0.01)

    with pytest.raises(TimeoutError, match="fanout timed out"):
        await asyncio.wait_for(probe_task, timeout=1)
    assert bridge.registration("timeout") is None
    assert not bridge._fanout_waiters

    _reply(event_queue, "timeout#probe", 0, _result("timeout", 3))
    await asyncio.sleep(0.15)
    assert bridge.registration("timeout") is None
    assert not bridge._fanout_waiters


@pytest.mark.asyncio
async def test_usage_delivery_cleans_registration(
    bridge_factory: BridgeFactory,
) -> None:
    bridge, command_queues, event_queue = bridge_factory(1)
    probe_task, _ = await _start_probe(bridge, command_queues, "usage")
    event_queue.put(
        {
            "type": "probe",
            "request_id": "usage",
            "ok": True,
            "result": _result("usage", 1),
        }
    )
    await asyncio.wait_for(probe_task, timeout=1)

    expected = LwdUsage(prompt_tokens=16, completion_tokens=4, cached_tokens=8)
    event_queue.put({"type": "usage", "request_id": "usage", "usage": expected})
    assert await bridge.wait_usage("usage", timeout=1) is expected
    assert bridge.registration("usage") is None


@pytest.mark.asyncio
async def test_usage_timeout_cleans_registration(
    bridge_factory: BridgeFactory,
) -> None:
    bridge, command_queues, event_queue = bridge_factory(1)
    probe_task, _ = await _start_probe(bridge, command_queues, "usage-timeout")
    event_queue.put(
        {
            "type": "probe",
            "request_id": "usage-timeout",
            "ok": True,
            "result": _result("usage-timeout", 1),
        }
    )
    await asyncio.wait_for(probe_task, timeout=1)

    assert await bridge.wait_usage("usage-timeout", timeout=0.01) is None
    assert bridge.registration("usage-timeout") is None


@pytest.mark.asyncio
async def test_probe_cancellation_cleans_waiters_and_late_events_are_harmless(
    bridge_factory: BridgeFactory,
) -> None:
    bridge, command_queues, event_queue = bridge_factory(1)
    probe_task, _ = await _start_probe(bridge, command_queues, "cancel")
    assert bridge.registration("cancel") is not None

    probe_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await probe_task
    assert bridge.registration("cancel") is None
    assert not bridge._probe_futures

    event_queue.put(
        {
            "type": "probe",
            "request_id": "cancel",
            "ok": True,
            "result": _result("cancel", 1),
        }
    )
    event_queue.put(
        {
            "type": "usage",
            "request_id": "cancel",
            "usage": LwdUsage(16, 1, 0),
        }
    )
    await asyncio.sleep(0.15)
    assert bridge.registration("cancel") is None


@pytest.mark.asyncio
async def test_duplicate_supersedes_and_old_release_preserves_new_generation(
    bridge_factory: BridgeFactory,
) -> None:
    bridge, command_queues, event_queue = bridge_factory(1)
    request_id = "duplicate"
    old_probe, _ = await _start_probe(bridge, command_queues, request_id)
    old_registration = bridge.registration(request_id)
    assert old_registration is not None

    new_probe, _ = await _start_probe(bridge, command_queues, request_id)
    with pytest.raises(RuntimeError, match="superseded"):
        await old_probe
    assert (
        await bridge.wait_usage(request_id, timeout=1, registration=old_registration)
        is None
    )
    new_registration = bridge.registration(request_id)
    assert new_registration is not None
    assert new_registration is not old_registration

    bridge.release(request_id, old_registration)
    assert bridge.registration(request_id) is new_registration

    event_queue.put(
        {
            "type": "probe",
            "request_id": request_id,
            "ok": True,
            "result": _result(request_id, 1),
        }
    )
    await asyncio.wait_for(new_probe, timeout=1)

    # Existing callers without a generation token retain unconditional release.
    bridge.release(request_id)
    assert bridge.registration(request_id) is None
