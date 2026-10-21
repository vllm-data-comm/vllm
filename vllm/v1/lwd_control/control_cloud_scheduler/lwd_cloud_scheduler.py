# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""云侧混排调度器(PD_mix 唯一形态):每步至多消费通知队列队首一条
批量 RangeNotify,与全部 decode 态请求同批下发;设计见
lwd_mixed_batch_design.md + lwd_chunked_prefill_design.md。

四道门(分类/ADD 落地/提前收水位/preflight 三检)任一不过即本步
decode-only,预告留队首重试;放行步 offset 对账 + 双钳制复现 +
步后全量核验 fail-fast。不兼容云侧抢占重算与 prefix caching
(部署须保证 KV 充足并关闭缓存),核验把违背变成当场报错。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.lwd_logger import lwd_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import (
    LwdBatch,
    LwdBatchType,
    LwdEmbedBatch,
    SchedulerOutput,
)
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_prefill_notify_queue import (
    LwdPrefillNotifyQueue,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    LwdBaseScheduler,
    LwdReqPhase,
    lwd_spec_factor,
)
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LWD_NOT_FINISHED,
    LwdC2eNotify,
    LwdRangeItem,
    LwdRangeNotify,
)
from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag

LwdDownCarrier = tuple[torch.Tensor, list[str], int, int]
"""`ModelRunnerOutput.lwd_down_carrier` set by the cloud worker: (pinned
packed meta, req_ids, hidden numel, DOWN seqno)."""


@dataclass
class LwdC2eMeta:
    """Step metadata unpacked from the cloud worker's DOWN carrier."""

    hidden_num_elements: int
    """Element count of the DOWN hidden tensor; the edge posts an exact-size
    receive from it."""
    top_id_ths: list[list[int]]
    """Per request, the ranks of its accepted tokens within the step's
    logits, in hidden row order."""
    num_accepted_tokens: list[int]
    req_ids: list[str]
    """Request ids in hidden row order."""
    down_seqno: int
    """DOWN channel seqno of the hidden packet; pairs the edge's receive on
    the tagless HCCL channel."""


class LwdCloudScheduler(LwdBaseScheduler):
    """notify 门控的 prefill/decode 混排调度器(PD_mix,云侧唯一形态)。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # prefill 通知队列:引擎 ingest 线程投递边侧批量 RangeNotify,本线程每步
        # 至多消费队首一条;预告自带 seqno 即本批 UP 链配对号。引擎侧经
        # self.scheduler.prefill_notify_queue 借引用投递/清扫(线程纪律
        # 归通知队列)
        self.prefill_notify_queue = LwdPrefillNotifyQueue()
        # Set by the engine from feature_ctrl.enable_early_recv: with it on, a
        # prefill batch waits until its UP data is received.
        self.lwd_early_recv = False
        # MTP 预算系数:decode 每请求每步消耗 1+k 个 token(草稿 token
        # 占预算与 KV);公式唯一住处 lwd_base_scheduler.lwd_spec_factor
        self._lwd_spec_factor: int = lwd_spec_factor(self.vllm_config)
        lwd_logger.info(
            "[cloud-sched] chunked-prefill batches mixed "
            "with decode (one notify consumed per step at most, "
            "spec_factor=%d)",
            self._lwd_spec_factor,
        )
        # 诊断中枢:spec 接受率累计器/周期直出与旁路载荷提取都住
        # lwd_control_diag,调度器只调用不持有
        self._lwd_diag = LwdControlDiag()
        # 待发 c2e:update_from_output 只入列(纯提取不做 I/O),引擎
        # post_step 统一冲刷发边——边侧据此预挂 DOWN recv
        self.pending_c2e: list[LwdC2eNotify] = []

    def update_from_output(self, scheduler_output, model_runner_output):
        """原生入账(含 spec 观测)后解 carrier 组 c2e 入 pending_c2e;
        DISABLE_DOWN 旁路逐请求 token 直传;无 carrier 的空步跳过。"""
        self._lwd_diag.observe_spec_acceptance(
            scheduler_output, model_runner_output, self.requests
        )
        engine_core_outputs = super().update_from_output(
            scheduler_output, model_runner_output
        )
        carrier = getattr(model_runner_output, "lwd_down_carrier", None)
        if carrier is not None:
            meta = self._lwd_carrier_to_meta(carrier)
            finish_reasons = self._lwd_c2e_finish_reasons(
                meta.req_ids, engine_core_outputs
            )
            LwdControlDiag.cloud_step(meta.req_ids, finish_reasons, self.requests)
            # [Lwd][eos-trace] A 点:零接受行留痕(kill 根因定位,
            # 打原始行集——零接受活行剔除已随 4 卡修复一并移除,此处
            # 仅作发生率观测)
            LwdControlDiag.trace_cloud_c2e_rows(
                meta.down_seqno,
                meta.req_ids,
                meta.num_accepted_tokens,
                finish_reasons,
            )
            # 诊断旁路(SKIP_SAMPLE):逐请求 accepted ids 捎带
            token_ids: list[list[int]] = []
            if LwdControlDiag.token_passthrough_enabled():
                token_ids = LwdControlDiag.bypass_token_payload(
                    meta.req_ids, engine_core_outputs
                )
            self.pending_c2e.append(
                LwdC2eNotify(
                    hidden_num_elements=meta.hidden_num_elements,
                    top_id_ths=meta.top_id_ths,
                    num_accepted_tokens=meta.num_accepted_tokens,
                    req_ids=meta.req_ids,
                    finish_reasons=finish_reasons,
                    down_seqno=meta.down_seqno,
                    token_ids=token_ids,
                )
            )
        elif LwdControlDiag.disable_down_enabled():
            self._lwd_append_diag_c2e(engine_core_outputs)
        return engine_core_outputs

    def _lwd_append_diag_c2e(self, engine_core_outputs) -> None:
        """DISABLE_DOWN 诊断旁路:云不做 collect/不发 DOWN,c2e 照常
        驱动边侧交付——hidden=0 标记无张量;行集提取归诊断类,wire
        构造留调度器。空步无行即不发空通告。"""
        rows = LwdControlDiag.disable_down_rows(engine_core_outputs)
        if not rows:
            return
        req_ids = list(rows)
        self.pending_c2e.append(
            LwdC2eNotify(
                hidden_num_elements=0,
                top_id_ths=[],
                num_accepted_tokens=[],
                req_ids=req_ids,
                finish_reasons=self._lwd_c2e_finish_reasons(
                    req_ids, engine_core_outputs
                ),
                down_seqno=-1,
                token_ids=[rows[rid] for rid in req_ids],
            )
        )

    @staticmethod
    def _lwd_carrier_to_meta(carrier: LwdDownCarrier) -> LwdC2eMeta:
        """pinned 载荷解包为步元数据。

        载荷 = 云侧收集后 pinned 的扁平 int 张量,按"变长段在前、定长
        尾巴在后"排布,无需另带长度字段即可从尾部反解:

            vals = [ranks(变长:每请求 seg_lens[i] 个 rank id,按请求序
                    拼接) ... | counts(n_req 个 accepted) ...
                    | seg_lens(n_req 个段长)]

        以 n_req=2、seg_lens=[3,1]、counts=[5,2] 为例:
            vals = [r0 r1 r2 | r0 | 5 2 | 3 1]
            counts   = vals[-2*n_req : -n_req] -> [5, 2]
            seg_lens = vals[-n_req:]            -> [3, 1]
            ranks    = vals[:-2*n_req]          -> [r0, r1, r2, r0]
        再按 seg_lens 把 ranks 切回逐请求的 top_id_ths 行
        (本例 [[r0,r1,r2],[r0]])。-2*n_req/-n_req 两个魔数即
        "counts 段 + seg_lens 段"两条定长尾巴的宽度。"""
        pinned, req_ids, hidden_numel, seqno = carrier
        n_req = len(req_ids)
        vals = pinned.tolist()
        counts = vals[-2 * n_req : -n_req]
        seg_lens = vals[-n_req:]
        ranks_flat = vals[: -2 * n_req]
        top_id_ths: list[list[int]] = []
        off = 0
        for seg_len in seg_lens:
            top_id_ths.append(ranks_flat[off : off + seg_len])
            off += seg_len
        return LwdC2eMeta(
            hidden_num_elements=hidden_numel,
            top_id_ths=top_id_ths,
            num_accepted_tokens=list(counts),
            req_ids=list(req_ids),
            down_seqno=seqno,
        )

    @staticmethod
    def _lwd_c2e_finish_reasons(req_ids: list[str], engine_core_outputs) -> list[int]:
        """req_ids 对齐的逐请求完成码:输出带 finish_reason 取其码,
        finished_requests 缺口按 ABORT 兜底,其余 NOT_FINISHED。"""
        from vllm.v1.engine import FinishReason

        reasons: dict[str, int] = {}
        for outputs in engine_core_outputs.values():
            for out in outputs.outputs:
                if out.finish_reason is not None:
                    reasons.setdefault(out.request_id, int(out.finish_reason))
            for request_id in outputs.finished_requests or ():
                reasons.setdefault(request_id, int(FinishReason.ABORT))
        return [reasons.get(request_id, LWD_NOT_FINISHED) for request_id in req_ids]

    def _lwd_collect_decode_requests(self) -> list[str]:
        """收集 decode 态(prompt 已算完)请求的 req_id,按 running/
        waiting/skipped 序,输出可直接作为可见集调度的入参;
        prefill 半途请求(num_computed 爬坡中)不进 decode 集,留在
        隐藏集等自己的下一条 chunk 预告驱动续排。"""
        return [
            req.request_id
            for req in self._lwd_reqs_in_phase(
                LwdReqPhase.DECODE,
                self.running,
                self.waiting,
                self.skipped_waiting,
            )
        ]

    # ------------------------------------------------------------------ #
    # notify 条目分类与 preflight                                          #
    # ------------------------------------------------------------------ #
    def _lwd_filter_ready_items(
        self, items: list[LwdRangeItem]
    ) -> tuple[list[LwdRangeItem], bool]:
        """从预告条目中筛出就绪项。

        items 专用于 prefill_notify:一条 LwdRangeNotify(prefill 预告)
        的 items 即一个边侧 prefill 批——批内逐请求一个 LwdRangeItem
        (request_id + 该 chunk 的 offset/num_tokens + mrope 携带位),
        整批共享一个 seqno(UP 数据面配对号),故整批 all-or-nothing。

        逐条对云侧账本(self.requests)核实——ready(已落地且待
        prefill)保留,stale(已转 decode 相位,重复预告)幂等丢弃,
        unknown(ADD 未落地)置标志。返回 (ready, 是否有 unknown);
        调用方据 unknown 决定整批 defer(一次 UP 收对应整批张量,
        不能只排子集)。

        unknown 存在即整批 defer(all-or-nothing:一次 UP recv 对应整批
        张量,不能只放行子集)。"""
        ready: list[LwdRangeItem] = []
        unknown = False
        for item in items:
            request = self.requests.get(item.request_id)
            if request is None:
                unknown = True
                continue
            if self._lwd_req_phase(request) is LwdReqPhase.DECODE:
                lwd_logger.warning(
                    "[cloud-sched] drop stale notify item req=%s "
                    "num=%d (already prefilled; duplicate notify?)",
                    item.request_id,
                    item.num_tokens,
                )
                continue
            ready.append(item)
        return ready, unknown

    def _lwd_up_recv_complete(self, notify: LwdRangeNotify) -> bool:
        """提前收闸门:该 seqno 的 UP 数据收完才放行;判据 = 通道水位(FIFO 完成序)。"""
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
            is_irecv_complete,
        )
        from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType

        ready = is_irecv_complete(LwdChannelType.UP, notify.seqno)
        # [Lwd][pre-recv] 闸门读数:与 HINT/SUBMIT/DONE/REPORT/POST 用同一单调钟,
        # 可把"预告到达->worker挂recv->worker收完上报->引擎放行"四段直接相减。
        return ready

    def _lwd_preflight_ok(self, items: list[LwdRangeItem], num_decode: int) -> bool:
        """放行三检:token 预算 / 名额(只数新增) / KV 保守预估;decode 占用按 1+k。"""
        need_tokens = (
            sum(item.num_tokens for item in items) + num_decode * self._lwd_spec_factor
        )
        if need_tokens > self.max_num_scheduled_tokens:
            lwd_logger.info(
                "[cloud-sched] preflight hold: tokens %d > budget %d",
                need_tokens,
                self.max_num_scheduled_tokens,
            )
            return False
        # 名额只数新增准入:续传 chunk 的请求已在 running,再按条目数
        # 计会双重占用名额造成假性 hold(chunk 化前批内恒为新增,
        # len(items) 与新增数相等)
        running_ids = {req.request_id for req in self.running}
        new_items = sum(1 for item in items if item.request_id not in running_ids)
        if len(self.running) + new_items > self.max_num_running_reqs:
            lwd_logger.info(
                "[cloud-sched] preflight hold: seqs %d+%d > max %d",
                len(self.running),
                new_items,
                self.max_num_running_reqs,
            )
            return False
        need_blocks = sum(
            cdiv(item.num_tokens, self.block_size) for item in items
        ) + num_decode * cdiv(self._lwd_spec_factor, self.block_size)
        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        if need_blocks > free_blocks:
            lwd_logger.info(
                "[cloud-sched] preflight hold: kv blocks %d > free %d",
                need_blocks,
                free_blocks,
            )
            return False
        return True

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        """混排调度:每步四道门决策——条目对账(all-or-nothing)→ 提前收
        水位 → preflight 三检 → 混排步;任一门不过即本步 decode-only,
        预告留队首下步重试。"""
        self._lwd_check_no_prefill_throttle(throttle_prefills)
        decode_ids = self._lwd_collect_decode_requests()
        notify = self.prefill_notify_queue.peek()
        if notify is not None:
            items, unknown = self._lwd_filter_ready_items(list(notify.items))
            if not items and not unknown:
                # 全部条目失效(完结/重复):弹队,本步纯 decode
                self.prefill_notify_queue.pop_front(notify)
                notify = None
            elif unknown:
                # 请求 ADD 未落地(input_queue 次序滞后):预告留队首,
                # 本步先 decode
                lwd_logger.info(
                    "[cloud-sched] notify seqno=%d deferred: request(s) not added yet",
                    notify.seqno,
                )
                notify = None
            elif self.lwd_early_recv and not self._lwd_up_recv_complete(notify):
                # 提前收闸门:UP 张量还没收完 -> 本步不下发这个 prefill 批
                # (notify 留队首,下步再判),本步只下 decode。收完的批优先,
                # 没收完的批一直让 decode 自转,不把等待塞进 worker 前向。
                notify = None
            elif not self._lwd_preflight_ok(items, len(decode_ids)):
                # 预算/名额/KV 不足:预告留队首,decode 泄压后重试。
                # 这条必须打日志:它是"数据都收完了、闸门也放行了,prefill 却
                # 一直不下发"的唯一静默出口(现场排查尾部停摆时,先看这条)。
                lwd_logger.info(
                    "[cloud-sched] notify seqno=%d deferred: preflight "
                    "rejected (reqs=%d tokens=%d decode=%d pending_notify=%d)",
                    notify.seqno,
                    len(items),
                    sum(item.num_tokens for item in items),
                    len(decode_ids),
                    len(self.prefill_notify_queue),
                )
                notify = None
            else:
                return self._schedule_mixed(notify, items, decode_ids)
        return self._lwd_schedule(decode_ids)

    def _schedule_mixed(
        self,
        notify: LwdRangeNotify,
        items: list[LwdRangeItem],
        decode_ids: list[str],
    ) -> SchedulerOutput:
        """混排步:decode 精确消耗推导 + 双钳制(阈值/预算)经基类复现
        chunk 边界;消费队首、挂 EMBED 批。"""
        req_ids = decode_ids + [item.request_id for item in items]
        out = self._lwd_schedule(
            req_ids,
            token_budget_cap=self._lwd_compute_decode_consumption(decode_ids)
            + sum(item.num_tokens for item in items),
            chunk_threshold=max(item.num_tokens for item in items),
        )
        # 消费队首(通知队列按对象身份弹出;该 notify 若已被 abort 清扫
        # 摘除,pop_front 返回 False,等价已消费)
        self.prefill_notify_queue.pop_front(notify)
        self._lwd_attach_embed_batch(out, notify, items)
        return out

    def _lwd_compute_decode_consumption(self, decode_ids: list[str]) -> int:
        """decode 精确消耗:预算钳制的精确性依赖逐字复刻原生 running 段
        (scheduler.py:381-416)的 num_new 推导——1+len(spec) 不是准确式:
        ① placeholder(spec 拒绝遗留的非零值)计入排程窗口;
        ② max_tokens 守卫(占位>0 且距 max_tokens 不足 2)与
          next_decode_eligible_step 会让请求整步跳过、消耗 0。
        不逐字复刻就会留下余量,使末位截断块被超排(真机实录:
        decode 请求被守卫跳过消耗 0,末位块 604 被排成 608)。
        threshold 不约束 decode(正常量级 T ≫ num_new,病态小 chunk
        配置见设计 §8-C9);预算由构造保证充足,不影响本推导。"""
        decode_exact = 0
        for rid in decode_ids:
            req = self.requests[rid]
            placeholders = req.num_output_placeholders
            if (
                placeholders > 0
                and req.num_computed_tokens + 2 - placeholders
                >= req.num_prompt_tokens + req.max_tokens
            ):
                continue  # 原生 max_tokens 守卫:本步跳过,消耗 0
            if self.current_step < req.next_decode_eligible_step:
                continue  # V2+PP 节拍守卫:本步跳过,消耗 0
            decode_exact += max(
                0,
                min(
                    req.num_tokens_with_spec + placeholders - req.num_computed_tokens,
                    self.max_model_len - 1 - req.num_computed_tokens,
                ),
            )
        return decode_exact

    @staticmethod
    def _lwd_attach_embed_batch(
        out: SchedulerOutput, notify: LwdRangeNotify, items: list[LwdRangeItem]
    ) -> None:
        """挂 EMBED 批:UP 链 seqno 随批下发云 worker,批配对号取预告
        自带 seqno(与边侧 EMBED 批派发号同源同值)。token_ids 为占位
        列表——长度必须等于边侧实际发送的逐请求 token 数,recv numel
        才能与边侧 isend 严格相等(HCCL P2P 要求两端 numel 匹配);
        mrope positions 逐 item 透传:runner 对 MM 请求 frombuffer 直写
        req_state(文本请求空 bytes、零路径变化)。"""
        out.lwd_batch = LwdBatch(
            batch_type=LwdBatchType.LWD_EMBED,
            seqno=notify.seqno,
            batch_meta=LwdEmbedBatch(
                req_ids=[item.request_id for item in items],
                token_ids=[[0] * item.num_tokens for item in items],
                prompt_offsets=[item.offset for item in items],
                has_mrope=[item.has_mrope for item in items],
                mrope_positions=[item.mrope_positions for item in items],
            ),
        )
