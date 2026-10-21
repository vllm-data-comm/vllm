# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lwd 边/云调度器公共基类:相位谓词与可见集调度原语(三队列剔除-
调度-拼回)。

继承 AsyncScheduler,原生调度行为与 isinstance 判定不变。
"""

from __future__ import annotations

from collections.abc import Iterator
from enum import Enum, auto

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.request import Request


def lwd_spec_factor(vllm_config) -> int:
    """MTP 预算系数:decode 每请求每步消耗 1+k 个 token(草稿 token 占
    预算与 KV),无 spec 配置恒 1。唯一住处(边云同源,spec 系数以云侧
    LwdCloudParamsNotify 为准);main 的 vllm/config/lwd 是拓扑快照,无此派生值,
    故随调度原语住在基类模块。"""
    spec_config = getattr(vllm_config, "speculative_config", None)
    num_spec_tokens = (
        getattr(spec_config, "num_speculative_tokens", None)
        if spec_config is not None
        else None
    )
    return (num_spec_tokens or 0) + 1


class LwdReqPhase(Enum):
    """请求的工作相位(边云同判据):PREFILL = 嵌入未发完
    (num_computed < num_prompt),DECODE = 等云侧采样回程。"""

    PREFILL = auto()
    DECODE = auto()


class LwdBaseScheduler(AsyncScheduler):
    """边/云调度器公共基类(AsyncScheduler 子类)。"""

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        """原生兼容调度入口:step_with_batch_queue 按位置传
        throttle_prefills,签名收口在此,边云子类只实现一步角色调度。
        节流提示不消费——各角色的派发门自行决策(mm 平级)。"""
        return self._lwd_schedule_step()

    def _lwd_schedule_step(self) -> SchedulerOutput:
        """One role-specific scheduling step; edge/cloud subclasses own it."""
        raise NotImplementedError

    @staticmethod
    def _lwd_req_phase(request: Request) -> LwdReqPhase:
        """唯一相位判据,边云各处的队列×相位扫描共用。"""
        if request.num_computed_tokens < request.num_prompt_tokens:
            return LwdReqPhase.PREFILL
        return LwdReqPhase.DECODE

    def _lwd_reqs_in_phase(self, phase: LwdReqPhase, *queues) -> Iterator[Request]:
        """按相位扫多队列(已终结请求跳过——账面可能同时满足判据)。"""
        for queue in queues:
            for request in queue:
                if not request.is_finished() and self._lwd_req_phase(request) is phase:
                    yield request

    def _lwd_new_queue(self, reqs: list[Request]) -> RequestQueue:
        """新建调度策略队列并装入 reqs。"""
        queue = create_request_queue(self.policy)
        for req in reqs:
            queue.add_request(req)
        return queue

    def _lwd_schedule(
        self,
        req_ids: list[str],
        *,
        token_budget_cap: int | None = None,
        chunk_threshold: int | None = None,
    ) -> SchedulerOutput:
        """隔离调度:只调度 req_ids 指定的请求——其余从三队列临时
        摘出隐藏,步后按原队列拼回(可见集机制)。

        waiting/skipped 来源走原生准入(allocate/状态迁移/记账一样
        不少),running 来源走续跑。拼回:仍被调度的接在隐藏 running
        之后,被抢占的排 waiting 尾部,被跳过的排 skipped 队首。

        token_budget_cap / chunk_threshold 仅收紧不放宽原生上限:
        边侧传 cap 复现 picker 的 chunk 划分,云侧传双钳制复现批内
        边界(阈值=批内最大条目 + 预算=decode 精确消耗+Σchunk)。
        隐藏的 running 对原生并发准入不可见,名额按隐藏数扣减防
        超发(边侧 picker free_seq、云侧 preflight 为第一道防线,
        此处为纵深)。"""
        picked = {self.requests[req_id] for req_id in req_ids}
        from_running = [req for req in self.running if req in picked]
        from_waiting = [req for req in self.waiting if req in picked]
        from_skipped = [req for req in self.skipped_waiting if req in picked]
        self.running = [req for req in self.running if req not in picked]
        self.waiting.remove_requests(from_waiting)
        self.skipped_waiting.remove_requests(from_skipped)
        saved = (self.running, self.waiting, self.skipped_waiting)
        # 可见集归位后单独调度
        self.running = from_running
        self.waiting = self._lwd_new_queue(from_waiting)
        self.skipped_waiting = self._lwd_new_queue(from_skipped)
        saved_cap = self.max_num_running_reqs
        self.max_num_running_reqs = max(0, saved_cap - len(saved[0]))
        saved_budget = self.max_num_scheduled_tokens
        if token_budget_cap is not None:
            self.max_num_scheduled_tokens = min(saved_budget, max(0, token_budget_cap))
        sched_cfg = self.scheduler_config
        saved_threshold = sched_cfg.long_prefill_token_threshold
        if chunk_threshold is not None:
            sched_cfg.long_prefill_token_threshold = chunk_threshold
        try:
            out = super().schedule()
        finally:
            self.max_num_running_reqs = saved_cap
            self.max_num_scheduled_tokens = saved_budget
            sched_cfg.long_prefill_token_threshold = saved_threshold
            post = (self.running, self.waiting, self.skipped_waiting)
            self.running, self.waiting, self.skipped_waiting = saved
            self.running += post[0]
            self.waiting.extend(post[1])
            self.skipped_waiting.prepend_requests(post[2])
        return out
