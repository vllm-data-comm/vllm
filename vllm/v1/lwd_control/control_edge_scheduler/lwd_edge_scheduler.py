# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""边侧调度器:chunk 化 prefill 组批 + 控制面出口(RangeNotify/seqno)。

边侧只做 embedding(无 decode),请求生命周期全程留原生调度器:
入队 -> EMBED(chunk 组批/范围预告)-> 嵌入完结转 decode 相位留
running 等云结果 -> 云结果终结(完成码映射原生 finish),迟到结果
幂等丢弃。schedule() 出批全量复用原生(可见集);步进与收割走引擎
原生批队列路径。chunk 语义见 lwd_chunked_prefill_design.md §3.1。
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from vllm.lwd_logger import lwd_logger
from vllm.v1.core.sched.output import (
    LwdBatch,
    LwdBatchType,
    LwdEmbedBatch,
    LwdUnembedBatch,
    SchedulerOutput,
)
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    FinishReason,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_picker import (
    LwdPrefillCandidate,
    lwd_pick_prefill_batch,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    LwdBaseScheduler,
    LwdReqPhase,
    lwd_spec_factor,
)
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LWD_NOT_FINISHED,
    LwdC2eNotify,
    LwdEdgeToCloudMessage,
    LwdRangeItem,
    LwdRangeNotify,
)
from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.request import Request

LwdSendToCloud = Callable[[LwdEdgeToCloudMessage], None]
"""Sends one message to the cloud; raises instead of losing it."""

# 云端完成码 -> 边侧终态(前端 reason 走输出通道;ERROR 落 ABORTED)
_LWD_FINISH_STATUS = {
    FinishReason.STOP: RequestStatus.FINISHED_STOPPED,
    FinishReason.LENGTH: RequestStatus.FINISHED_LENGTH_CAPPED,
    FinishReason.ABORT: RequestStatus.FINISHED_ABORTED,
    FinishReason.ERROR: RequestStatus.FINISHED_ABORTED,
}

# 云侧 pending 上限,按未消费 chunk(seqno)计:一条 seqno = 一个云步
# 消费单位。边侧记录 seqno -> 批内请求列表,批内全部请求收到首 token
# 即该 chunk 已消费;未消费 chunk 数达到上限即停止新开 prefill——请求
# 在边侧 waiting 攒批(picker FCFS 前缀天然组多请求批),云侧 pending
# 排空后闸门自开。
_LWD_MAX_PENDING_CHUNKS = 3


class LwdEdgeScheduler(LwdBaseScheduler):
    """纯 prefill 调度语义 + 控制面出口(RangeNotify/seqno)。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Injected by the engine once the cloud accepted the handshake.
        self.lwd_send_to_cloud: LwdSendToCloud | None = None
        self._lwd_seqno = 0
        # 前缀缓存:manager 级必须关命中(命中使首条 offset!=0、被命中
        # 块从未写 KV 仍占块池);配置级保留使能以产出哈希链供云侧命中
        self.kv_cache_manager.enable_caching = False
        # pending 台账(一本账):seqno -> 批内未到账请求集。一条 seqno
        # = 一个云步消费单位;请求首 token 到账即从其批中消除,批内
        # 全部消除则整条移除——prefill 闸门水位 = len(本表)
        self._lwd_pending_batches: dict[int, set[str]] = {}
        # c2e 通告队列:控制面 IO 线程 append,调度步内消费出 UNEMBED 批
        self.unembed_notify_queue: deque[LwdC2eNotify] = deque()
        # pending 闸门状态(仅作开/关转换日志用)
        self._lwd_prefill_gate_held: bool = False
        # MTP 预算系数(decode 每请求每步 1+k token);边云预算部署对齐,
        # cap 直接读本地配置,公式唯一住处 lwd_base_scheduler.lwd_spec_factor
        self._lwd_spec_factor: int = lwd_spec_factor(self.vllm_config)
        # 多模态请求的整 prompt mrope positions([3,N] CPU 张量,准入时
        # 一次性算好,按 chunk 切片随 LwdEmbedBatch 下发 worker);嵌入
        # 完结/abort 时摘除。key 在 = 该请求各 chunk 的 has_mrope 恒真。
        self._lwd_mrope_positions_dict: dict[str, torch.Tensor] = {}
        # 惰性解析的模型类静态方法 _get_mrope_input_positions(直接缓存
        # 方法本身而非模型类,语义直达;首用即验,不支持的族 fail-fast)。
        self._lwd_mrope_positions_fn = None
        # chunk 单元 C = min(剩余, C):取本地 long_prefill_token_threshold
        # (原生截断边界,双侧天然复现);未配置(≤0)退化为对齐预算
        # max_num_scheduled_tokens。云侧不读本值,按批内 max(items) 覆写
        threshold = self.scheduler_config.long_prefill_token_threshold
        self._lwd_chunk_unit: int = (
            threshold if threshold > 0 else self.max_num_scheduled_tokens
        )
        self._lwd_check_deadlock_config()

    def _lwd_check_deadlock_config(self) -> None:
        """活性 fail-fast:最坏在途(满员)下 cap 恒 < 1 即死锁配置
        (设计 §2.6),拉起时直接拒绝。"""
        if (
            self.max_num_scheduled_tokens
            - self.max_num_running_reqs * self._lwd_spec_factor
            < 1
        ):
            raise RuntimeError(
                f"[LWD] 死锁配置: max_num_batched_tokens("
                f"{self.max_num_scheduled_tokens}) - max_num_seqs("
                f"{self.max_num_running_reqs}) * spec_factor("
                f"{self._lwd_spec_factor}) < 1, 续传 chunk 永无可调度预算; "
                "请增大 max_num_batched_tokens 或减小 max_num_seqs"
            )

    def has_requests(self) -> bool:
        """原生步进的取活谓词:有可派发的批(EMBED 可派 / c2e 待消费 /
        DISABLE_DOWN 直通积压)。闸门 HOLD 且无在飞时为 False,引擎经
        has_work 睡眠——schedule() 被调用即必有真批,空批不上队列
        (原生批队列契约,空批会以 None 结果炸收割路径)。"""
        return bool(self.unembed_notify_queue) or self._lwd_embed_dispatchable()

    def _lwd_schedule_step(self) -> SchedulerOutput:
        """EMBED 优先出批,否则消费 c2e 出 UNEMBED 批(先收尾再开新,
        防 EMBED 被 decode 回程挤出深度配额)。"""
        LwdControlDiag.edge_step_activity(
            len(self.running),
            sum(1 for _ in self._lwd_reqs_in_phase(LwdReqPhase.DECODE, self.running)),
        )
        out = self._lwd_schedule_embed()
        if out is not None:
            return out
        return self._lwd_schedule_unembed()

    def _lwd_embed_dispatchable(self) -> bool:
        """EMBED 可派判定(含闸门转换日志):续传在飞豁免两道闸门;
        否则要求满员检查通过、pending chunk 未达上限且 waiting 有货
        (pending 达上限即 HOLD,请求在 waiting 攒批,云侧排空自开)。"""
        if self._lwd_has_prefill_chunk_inflight():
            self._lwd_log_gate_open()
            return True
        pending_chunks = len(self._lwd_pending_batches)
        if (
            len(self.running) >= self.max_num_running_reqs
            or pending_chunks >= _LWD_MAX_PENDING_CHUNKS
            or not (self.waiting or self.skipped_waiting)
        ):
            if (
                pending_chunks >= _LWD_MAX_PENDING_CHUNKS
                and not self._lwd_prefill_gate_held
            ):
                self._lwd_prefill_gate_held = True
                lwd_logger.info(
                    "[edge-sched] prefill gate HOLD: pending_chunks=%d >= %d",
                    pending_chunks,
                    _LWD_MAX_PENDING_CHUNKS,
                )
            return False
        self._lwd_log_gate_open()
        return True

    def _lwd_log_gate_open(self) -> None:
        """闸门开转换日志(HOLD -> OPEN 只打一次)。"""
        if self._lwd_prefill_gate_held:
            self._lwd_prefill_gate_held = False
            lwd_logger.info("[edge-sched] prefill gate OPEN")

    def _lwd_schedule_embed(self) -> SchedulerOutput | None:
        """Chunked batch + budget clamp + RangeNotify, then the EMBED batch
        (seqno bound to the notify). None when nothing is dispatchable.

        The batch is never dropped once scheduled: scheduling already advanced
        num_computed_tokens, so a dropped chunk would never be embedded and
        the next offset would not match the cloud. A notify that cannot be
        sent raises instead."""
        if not self._lwd_embed_dispatchable():
            return None
        out = self._lwd_schedule_batch()
        if not out.num_scheduled_tokens:
            return None
        self.lwd_edge_notify_range(out)
        return out

    def _lwd_schedule_unembed(self) -> SchedulerOutput:
        """消费 c2e 通告出 UNEMBED 批;DISABLE_DOWN 直通(不派 worker)
        由引擎 executor 代理以已完成 future 承接,调度器路径统一。"""
        if self.unembed_notify_queue:
            return lwd_build_unembed_batch(self.unembed_notify_queue.popleft())
        return SchedulerOutput.make_empty()

    def update_from_output(self, scheduler_output, model_runner_output):
        """LWD 收割(返回值即引擎步输出,均不走原生入账):EMBED 批登记
        执行进度(完结者留 running 转 decode 相位);UNEMBED 批按 worker 回填的
        req_ids x sampled_token_ids 逐请求交付;无 lwd_batch 的占位批
        (底层正常完成返回 None,经引擎代理替换为占位)为 no-op。"""
        batch = getattr(scheduler_output, "lwd_batch", None)
        outputs: list[EngineCoreOutput] = []
        finished: set[str] = set()
        if batch is not None and batch.batch_type == LwdBatchType.LWD_UNEMBED:
            notify = scheduler_output.lwd_c2e_notify[0]
            req_ids = getattr(model_runner_output, "req_ids", None) or ()
            sampled = (
                dict(zip(req_ids, model_runner_output.sampled_token_ids))
                if req_ids
                else {}
            )
            outputs, finished = self._lwd_deliver_unembed(notify, sampled)
        # EMBED 批无本地出账(进度已于调度期乐观推进,嵌入完结即自然
        # 转 decode 相位);收割期统一对账清理 mrope positions(驻留
        # 模型下请求终结才出 requests,显式逐点摘除改为对账式回收)
        self._lwd_purge_mrope_positions()
        if not outputs:
            return {}
        return {
            0: EngineCoreOutputs(
                outputs=outputs,
                finished_requests=finished or None,
            )
        }

    def _lwd_deliver_unembed(
        self, notify: LwdC2eNotify, sampled_token_map: dict[str, list[int]]
    ) -> tuple[list[EngineCoreOutput], set[str]]:
        """逐请求交付:token 行缺席且未完结 = unembed 失败,ERROR 优先
        于云侧完成码;不在 awaiting 的迟到结果幂等丢弃。"""
        outputs: list[EngineCoreOutput] = []
        finished: set[str] = set()
        for index, request_id in enumerate(notify.req_ids):
            finish_reason = self._lwd_finish_code(notify, index)
            finished_now = finish_reason is not None
            sampled_token_ids = list(sampled_token_map.get(request_id, []))
            LwdControlDiag.edge_tokens_delivered(
                request_id,
                sampled_token_ids,
                finish_reason,
                self.vllm_config,
            )
            if not sampled_token_ids and not finished_now:
                # [Lwd][eos-trace] B 点:kill 行留痕,按 down_seqno 与
                # A(云零接受行)/C(worker 缺行)配对定案根因
                LwdControlDiag.trace_edge_kill_row(
                    seqno=notify.down_seqno,
                    rid=request_id,
                    accepted=(
                        notify.num_accepted_tokens[index]
                        if index < len(notify.num_accepted_tokens)
                        else -1
                    ),
                    finish_code=(
                        notify.finish_reasons[index] if notify.finish_reasons else -1
                    ),
                    batch_rows=len(notify.req_ids),
                    sampled_rows=len(sampled_token_map),
                )
                finish_reason = FinishReason.ERROR
                finished_now = True
            if not self.lwd_edge_deliver_tokens(
                request_id, sampled_token_ids, finished=finished_now
            ):
                lwd_logger.warning(
                    "[edge-deliver] drop stale cloud payload for %s (not awaiting)",
                    request_id,
                )
                continue
            if not sampled_token_ids and not finished_now:
                continue  # 无内容且未完结:不出空输出
            if finished_now:
                # 云端判停的地面真相落地:原生 finish 清账(KV/缓存释放
                # 与 finished_req_ids 交接全走原生)
                self.finish_requests(
                    [request_id],
                    _LWD_FINISH_STATUS.get(
                        finish_reason, RequestStatus.FINISHED_STOPPED
                    ),
                )
            outputs.append(
                EngineCoreOutput(
                    request_id,
                    sampled_token_ids,
                    finish_reason=finish_reason,
                )
            )
            if finished_now:
                finished.add(request_id)
        return outputs, finished

    @staticmethod
    def _lwd_finish_code(notify: LwdC2eNotify, index: int) -> FinishReason | None:
        """逐请求完成码:哨兵 = 未完结(None);与 req_ids 按位对齐,
        错配 IndexError fail-fast。"""
        code = notify.finish_reasons[index]
        return None if code == LWD_NOT_FINISHED else FinishReason(code)

    def _lwd_pick_prefill_batch(
        self,
    ) -> tuple[list[str], dict[str, int], int]:
        """picker 适配层:组装候选、算 cap/free_seq、模型族折算独占参数。"""
        inflight = len(self.running)
        cap = self.max_num_scheduled_tokens - inflight * self._lwd_spec_factor
        free_seq = self.max_num_running_reqs - inflight
        # 续传候选只取 prefill 相位(decode 相位驻留者 remaining=0,
        # picker 本也会跳过;过滤后候选举即续传判据本身)
        running = [
            LwdPrefillCandidate(
                r.request_id, r.num_prompt_tokens, r.num_computed_tokens
            )
            for r in self._lwd_reqs_in_phase(LwdReqPhase.PREFILL, self.running)
        ]
        waiting = [
            LwdPrefillCandidate(r.request_id, r.num_prompt_tokens, 0)
            for r in [*self.skipped_waiting, *self.waiting]
        ]
        req_ids, expected = lwd_pick_prefill_batch(
            running,
            waiting,
            cap=cap,
            free_seq=free_seq,
            chunk_unit=self._lwd_chunk_unit,
            single_prefill_min_tokens=(
                8000
                if self.vllm_config.model_config.hf_text_config.model_type
                == "deepseek_v4"
                else 0
            ),
        )
        return req_ids, expected, cap

    def _lwd_schedule_batch(self) -> SchedulerOutput:
        """chunk 组批:picker 选批,经基类可见集机制(预算钳到 cap)调度
        (chunk 划分由钳制参数保证被原生复现)。"""
        req_ids, _expected, cap = self._lwd_pick_prefill_batch()
        return self._lwd_schedule(req_ids, token_budget_cap=cap)

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int]]:
        """边侧直通:encoder 执行与缓存由 worker 自管;原生回滚会破坏
        chunk 契约,故不排不耗不回滚。"""
        return [], num_new_tokens, encoder_compute_budget, []

    def _lwd_has_prefill_chunk_inflight(self) -> bool:
        """running 中是否有未发完的 embed 请求;与 picker 续传分支同判据,两处须一致。"""
        return (
            next(
                self._lwd_reqs_in_phase(LwdReqPhase.PREFILL, self.running),
                None,
            )
            is not None
        )

    def lwd_admit_request(self, request: Request) -> None:
        """请求入口的调度器本地半段:模式边界校验 + mrope 预计算。
        云预告(通信)与入队归引擎 add_request,abort_immediately 走
        原生引擎路径(abort_requests 覆写里摘台账)。"""
        self._lwd_validate_request(request)
        self._lwd_compute_mrope_positions(request)

    @staticmethod
    def _lwd_request_has_mm(request: Request) -> bool:
        """请求是否带真实多模态数据(prompt_embeds passthrough 不算——
        它没有 grid 元数据,mrope 按文本位置处理,与原生口径一致)。"""
        return any(
            feature.modality != "prompt_embeds" for feature in request.mm_features
        )

    def _lwd_compute_mrope_positions(self, request: Request) -> None:
        """多模态准入时算全 prompt [3,N] mrope(边侧唯一持有真实
        prompt/grid);模型族无 _get_mrope_input_positions 即拒绝。"""
        if not self.vllm_config.model_config.uses_mrope:
            return
        if not self._lwd_request_has_mm(request):
            return
        if self._lwd_mrope_positions_fn is None:
            from vllm.model_executor.models import ModelRegistry

            model_cls, _ = ModelRegistry.resolve_model_cls(
                self.vllm_config.model_config.architectures,
                self.vllm_config.model_config,
            )
            fn = getattr(model_cls, "_get_mrope_input_positions", None)
            if fn is None:
                raise ValueError(
                    f"[LWD] multimodal requests in prefill-only mode "
                    f"require a model class exposing "
                    f"_get_mrope_input_positions (Qwen3VL family, incl. "
                    f"Qwen3.5); got {model_cls.__name__} "
                    f"(request {request.request_id})"
                )
            self._lwd_mrope_positions_fn = fn
        mrope_features = [
            f for f in request.mm_features if f.modality != "prompt_embeds"
        ]
        _t_mrope = time.monotonic()
        positions, _delta = self._lwd_mrope_positions_fn(
            input_tokens=list(request.prompt_token_ids),
            mm_features=mrope_features,
            config=self.vllm_config.model_config.hf_config,
        )
        assert positions.shape[1] == request.num_prompt_tokens, (
            f"[LWD] mrope positions width {positions.shape[1]} != prompt "
            f"len {request.num_prompt_tokens} (request {request.request_id})"
        )
        self._lwd_mrope_positions_dict[request.request_id] = positions
        # [Lwd][perf] mrope 全 prompt 计算逐请求耗时(在边引擎主线程上,
        # 期间调度循环停转;MM 劣化归因)

    def lwd_edge_notify_range(self, scheduler_output: SchedulerOutput) -> None:
        """Send one batch RangeNotify and attach the EMBED batch; a seqno is
        only taken by a notify that was sent."""
        built = self._lwd_build_range_items(scheduler_output.num_scheduled_tokens)
        if built is None:
            return
        items, req_ids, token_slices, prompt_offsets, has_mrope = built
        seqno = self._lwd_seqno
        assert self.lwd_send_to_cloud is not None, "cloud not connected"
        lwd_logger.info(
            "[edge] range announced seqno=%d reqs=%d tokens=%d",
            seqno, len(items), sum(i.num_tokens for i in items),
        )
        self.lwd_send_to_cloud(LwdRangeNotify(seqno=seqno, items=items))
        self._lwd_seqno = seqno + 1
        self._lwd_pending_batches[seqno] = set(req_ids)
        # 发送成功即组 EMBED 批挂 SO:seqno 是数据面发云张量的配对键
        # (与云侧 RangeNotify 登记同值),embed 载荷为本批逐请求的
        # token 片段(边 worker 按 req_ids 序扁平化);mrope positions
        # 已随 RangeNotify 走控制面,边 worker 不再组 aux 帧
        scheduler_output.lwd_batch = LwdBatch(
            batch_type=LwdBatchType.LWD_EMBED,
            seqno=seqno,
            batch_meta=LwdEmbedBatch(
                req_ids=req_ids,
                token_ids=token_slices,
                prompt_offsets=prompt_offsets,
                has_mrope=has_mrope,
            ),
        )

    def _lwd_build_range_items(
        self, scheduled: dict[str, int]
    ) -> (
        tuple[list[LwdRangeItem], list[str], list[list[int]], list[int], list[bool]]
        | None
    ):
        """组批内逐请求的 RangeItem 与 EMBED 载荷切片;mrope 逐 chunk
        列切片序列化为原始字节随 item 走控制面。
        无条目返回 None。"""
        items: list[LwdRangeItem] = []
        req_ids: list[str] = []
        token_slices: list[list[int]] = []
        prompt_offsets: list[int] = []
        has_mrope: list[bool] = []
        for request_id, num_tokens in scheduled.items():
            request = self.requests.get(request_id)
            if request is None:
                continue
            # _update_after_schedule 已乐观推进 num_computed,起点需回退本步量
            offset = request.num_computed_tokens - num_tokens
            positions = self._lwd_mrope_positions_dict.get(request_id)
            req_has_mrope = positions is not None
            mrope_bytes = b""
            if req_has_mrope:
                assert positions is not None
                mrope_bytes = (
                    positions[:, offset : offset + num_tokens]
                    .contiguous()
                    .numpy()
                    .tobytes()
                )
            items.append(
                LwdRangeItem(
                    request_id=request_id,
                    offset=offset,
                    num_tokens=num_tokens,
                    has_mrope=req_has_mrope,
                    mrope_positions=mrope_bytes,
                )
            )
            req_ids.append(request_id)
            token_slices.append(
                list(request.prompt_token_ids[offset : offset + num_tokens])
            )
            prompt_offsets.append(offset)
            has_mrope.append(req_has_mrope)
        return (
            (items, req_ids, token_slices, prompt_offsets, has_mrope) if items else None
        )

    def lwd_edge_abort(self, request_ids: list[str]) -> None:
        """abort 台账摘除(pending 水位防线,不摘会顶高 prefill 闸门);
        云侧 abort 通告功能未上,终结清账走原生 abort_requests。"""
        for request_id in request_ids:
            self._lwd_drop_pending_request(request_id)

    def lwd_edge_deliver_tokens(
        self, request_id: str, token_ids: list[int], finished: bool
    ) -> bool:
        """云结果对账:请求仍在原生 requests(decode 相位)则认领,
        首 token 到账即从 pending 台账消除(finished 再消除一次幂等,
        不摘会泄漏水位顶高闸门);已终结/未知请求幂等丢弃。"""
        request = self.requests.get(request_id)
        if request is None or request.is_finished():
            lwd_logger.warning(
                "[edge-deliver] stale result for req=%s (not awaiting)",
                request_id,
            )
            return False
        self._lwd_drop_pending_request(request_id)
        return True

    def _lwd_drop_pending_request(self, request_id: str) -> None:
        """从 pending 台账消除请求(首 token 到账/终结/abort 共用):
        各批剔除该请求,批内全部消除 -> 整条移除(pending 水位下降;
        不摘会只涨不降,prefill 闸门永久 HOLD)。"""
        for seqno in list(self._lwd_pending_batches):
            batch = self._lwd_pending_batches[seqno]
            batch.discard(request_id)
            if not batch:
                del self._lwd_pending_batches[seqno]

    def _lwd_purge_mrope_positions(self) -> None:
        """对账摘除 mrope positions:请求已出 requests(终结/abort)
        或嵌入完结转 decode 相位(各 chunk 已全部发出,notify 不再读)。
        驻留模型无"嵌入完结即出队"的天然摘除点,收割期统一对账。"""
        for rid in list(self._lwd_mrope_positions_dict):
            request = self.requests.get(rid)
            if request is None or self._lwd_req_phase(request) is LwdReqPhase.DECODE:
                self._lwd_mrope_positions_dict.pop(rid, None)

    def _lwd_validate_request(self, request: Request) -> None:
        """模式边界校验(边侧能力面):拒绝自带 prompt_embeds/pooling/
        结构化输出/不上 wire 的采样参数;违规抛 ValueError 于入队前拒绝。"""
        if request.prompt_embeds is not None:
            raise ValueError(
                f"[LWD] prefill-only mode does not accept client-provided "
                f"prompt_embeds (request {request.request_id}); the edge "
                "is the embedding owner"
            )
        self._lwd_validate_mm_request(request)
        if request.pooling_params is not None:
            raise ValueError(
                "[LWD] prefill-only mode does not support pooling requests "
                f"(request {request.request_id})"
            )
        if request.use_structured_output:
            raise ValueError(
                "[LWD] prefill-only mode does not support structured output "
                f"(request {request.request_id})"
            )
        sp = request.sampling_params
        unsupported = [
            name
            for name, value in (
                ("logit_bias", sp.logit_bias),
                ("allowed_token_ids", sp.allowed_token_ids),
                ("logprobs", sp.logprobs),
            )
            if value is not None
        ]
        if unsupported:
            raise ValueError(
                "[LWD] prefill-only mode does not support sampling "
                f"param(s) {', '.join(unsupported)} "
                f"(request {request.request_id}): not carried on the "
                "edge->cloud wire, cloud would silently sample without them"
            )

    def _lwd_validate_mm_request(self, request: Request) -> None:
        """多模态白名单:仅 image(merge/mrope 链路按 image 语义实现,
        其余模态未适配,静默放过即错算);拒绝 EVS(multimodal pruning)。"""
        if not request.mm_features:
            return
        unsupported_mm = sorted(
            {
                feature.modality
                for feature in request.mm_features
                if feature.modality not in ("image", "prompt_embeds")
            }
        )
        if unsupported_mm:
            raise ValueError(
                f"[LWD] prefill-only mode supports image-only "
                f"multimodal requests; got modality "
                f"{', '.join(unsupported_mm)} "
                f"(request {request.request_id})"
            )
        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config.is_multimodal_pruning_enabled():
            raise ValueError(
                "[LWD] prefill-only mode does not support multimodal "
                f"pruning (EVS) (request {request.request_id})"
            )


def lwd_build_unembed_batch(notify: LwdC2eNotify) -> SchedulerOutput:
    """组 UNEMBED 批:云结果不经原生排程,以 make_empty 为骨架挂 LwdUnembedBatch。"""
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = dict(
        zip(notify.req_ids, notify.num_accepted_tokens)
    )
    scheduler_output.total_num_scheduled_tokens = sum(notify.num_accepted_tokens)
    scheduler_output.lwd_batch = LwdBatch(
        batch_type=LwdBatchType.LWD_UNEMBED,
        seqno=notify.down_seqno,
        batch_meta=LwdUnembedBatch(
            req_ids=list(notify.req_ids),
            num_accept_tokens=list(notify.num_accepted_tokens),
            recv_num_elements=notify.hidden_num_elements,
            out_token_idxs=[],
            top_id_ths=list(notify.top_id_ths),
            token_ids=[list(t) for t in notify.token_ids],
        ),
    )
    scheduler_output.lwd_c2e_notify = [notify]
    return scheduler_output
