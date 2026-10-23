# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""云侧 prefill 通知队列:引擎 ingest 线程投递、调度主线程消费的线程接缝。

并发纪律唯一住处——一把锁串行全部操作,替代原先散在引擎与调度器
注释里的"deque 单操作原子 / 赋值整体替换 / 对象身份弹队"契约。
锁临界区无 IO:append/peek/pop_front 每步 O(1),purge O(n) 仅
abort 时发生。条目(items 列表)在 purge 中就地重建,消费方取
list(notify.items) 快照。
"""

from __future__ import annotations

import threading
from collections import deque

from vllm.v1.lwd_control.lwd_comm.lwd_messages import LwdRangeNotify


class LwdPrefillNotifyQueue:
    """边侧 RangeNotify 的云侧通知队列(先到先消费,每步至多吃队首一条)。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queue: deque[LwdRangeNotify] = deque()

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)

    def append(self, notify: LwdRangeNotify) -> None:
        """ingest 线程投递:每条预告整条入队(重复预告即重复点名,调度器
        分类幂等剔除,无副作用;预告自带 seqno,出批时作 UP 链配对号)。"""
        with self._lock:
            self._queue.append(notify)

    def peek(self) -> LwdRangeNotify | None:
        """主线程取队首(不弹出);空队列返回 None。"""
        with self._lock:
            return self._queue[0] if self._queue else None

    def pop_front(self, notify: LwdRangeNotify) -> bool:
        """主线程消费队首:当且仅当队首 is notify 时弹出。
        身份不符返回 False(notify 已被 purge 摘除),调用方视为已消费。"""
        with self._lock:
            if self._queue and self._queue[0] is notify:
                self._queue.popleft()
                return True
            return False

    def purge(self, request_id: str) -> int:
        """abort 清扫(abort 尚未接线):逐条目摘除该请求的 item,items 掏空则
        整条移除。返回摘除 item 数(供日志)。"""
        with self._lock:
            kept: deque[LwdRangeNotify] = deque()
            removed = 0
            for notify in self._queue:
                if notify.items:
                    items = [
                        item for item in notify.items if item.request_id != request_id
                    ]
                    removed += len(notify.items) - len(items)
                    if items:
                        notify.items = items
                        kept.append(notify)
            self._queue = kept
            return removed
