# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LwdPrefillNotifyQueue 测试:FIFO、身份消费、abort 清扫与并发烟雾。"""

from __future__ import annotations

import threading

from vllm.v1.lwd_control.control_cloud_scheduler.lwd_prefill_notify_queue import (
    LwdPrefillNotifyQueue,
)
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LwdRangeItem,
    LwdRangeNotify,
)


def notify(seqno: int, items: list[LwdRangeItem] | None = None) -> LwdRangeNotify:
    return LwdRangeNotify(seqno=seqno, items=items or [])


def test_fifo_peek_pop_front():
    queue = LwdPrefillNotifyQueue()
    assert queue.peek() is None
    n0, n1 = notify(0), notify(1)
    queue.append(n0)
    queue.append(n1)
    assert len(queue) == 2
    assert queue.peek() is n0
    assert queue.pop_front(n0) is True
    assert queue.peek() is n1
    assert queue.pop_front(n1) is True
    assert len(queue) == 0


def test_pop_front_identity_mismatch():
    """身份不符(队首已被替换/摘除):返回 False,队列原样。"""
    queue = LwdPrefillNotifyQueue()
    n0, n1 = notify(0), notify(1)
    queue.append(n0)
    queue.append(n1)
    assert queue.pop_front(n1) is False
    assert len(queue) == 2
    assert queue.peek() is n0


def test_purge_mixed_items():
    """条目级清扫:摘 item、items 掏空整条移除。"""
    queue = LwdPrefillNotifyQueue()
    mixed = notify(0, [LwdRangeItem("a", 0, 5), LwdRangeItem("b", 0, 5)])
    solo = notify(1, [LwdRangeItem("c", 0, 3)])
    queue.append(mixed)
    queue.append(solo)
    assert queue.purge("a") == 1
    assert mixed.items == [LwdRangeItem("b", 0, 5)]
    assert len(queue) == 2
    assert queue.purge("b") == 1  # 掏空 -> 整条移除
    assert len(queue) == 1
    assert queue.purge("zzz") == 0


def test_purged_notify_pop_front_is_noop():
    """被 purge 摘除的 notify,pop_front 返回 False(视为已消费)。"""
    queue = LwdPrefillNotifyQueue()
    n0 = notify(0, [LwdRangeItem("a", 0, 5)])
    queue.append(n0)
    queue.purge("a")
    assert queue.pop_front(n0) is False


def test_concurrent_append_pop_monotonic():
    """并发烟雾:IO 线程持续投递,主线程按序消费,seqno 严格递增。"""
    queue = LwdPrefillNotifyQueue()
    total = 2000
    producer_error: list[Exception] = []

    def produce():
        try:
            for i in range(total):
                queue.append(notify(i))
        except Exception as exc:  # pragma: no cover
            producer_error.append(exc)

    thread = threading.Thread(target=produce)
    thread.start()
    popped: list[int] = []
    while len(popped) < total:
        head = queue.peek()
        if head is None:
            continue
        if queue.pop_front(head):
            popped.append(head.seqno)
    thread.join()
    assert not producer_error
    assert popped == sorted(popped)
    assert len(popped) == total
