"""LWD 控制面诊断中枢:开关现读、spec 接受率观测、诊断旁路的判定与
载荷提取、debug 打点转发。

功能类(调度器/引擎)只经本模块触达诊断——不散读 env 开关、不直接
import vllm.v1.lwd_debug;诊断逻辑不与功能接口糅合,开关/旁路/观测
的增删只改这里。边界:本模块只出"判定与数据"(开关布尔、旁路行
集、累计读数),不持执行适配器(future/伪输出等 executor 边界契约
归引擎)。

诊断行为定义(排障口径):
  - SKIP_SAMPLE(边免采样):云侧把本步逐请求 accepted token ids 随
    c2e 捎带,边侧跳过 lm_head 采样还原;
  - DISABLE_DOWN(全直通):云不 collect、不发 DOWN 张量,c2e 以
    hidden=0 携带 token ids 直传,边侧收割直接交付。
两者均严格 == "1" 且按调用点现读(进程内改 env 即时生效,与
vllm-ascend 侧 envs.py 读法同名同义)。
"""

from __future__ import annotations

import os
import time

_LWD_ENV_PREFIX = "VLLM_ASCEND_LWD_"

# spec 接受率周期直出间隔(累计器常驻,周期打点)
_SPEC_LOG_INTERVAL_S = 10.0


def _lwd_debug_mods():
    """lwd_debug 惰性进口:本模块顶层保持零重依赖(纯逻辑可测),
    lwd_control 内对 lwd_debug 的 import 仍唯一收口在这里。
    Fallback: vllm.v1.lwd_debug is not migrated to main yet (issue #18
    breakpoint 4); until it lands, degrade to no-op shims so first-request
    diagnostics do not crash — the real classes take over automatically
    once the package exists."""
    try:
        from vllm.v1.lwd_debug import LwdControlLog, LwdDebug, LwdLogBase
    except ImportError:
        return _LwdDebugNoOp, _LwdDebugNoOp, _LwdDebugNoOp
    return LwdControlLog, LwdDebug, LwdLogBase


class _LwdDebugNoOp:
    """No-op stand-in for the not-yet-migrated vllm.v1.lwd_debug classes."""

    @staticmethod
    def set_debug(enabled: bool) -> None:
        pass

    @staticmethod
    def flight(running: int, awaiting: int) -> None:
        pass

    @staticmethod
    def edge_tokens_delivered(*args: object, **kwargs: object) -> None:
        pass

    @staticmethod
    def cloud_step(*args: object, **kwargs: object) -> None:
        pass

    @staticmethod
    def cloud_request_admitted(*args: object, **kwargs: object) -> None:
        pass


class LwdControlDiag:
    """控制面诊断逻辑的唯一住处(功能类经此类/本模块触达诊断)。"""

    # ------------------------------------------------------------------ #
    # 开关(调用点现读,严格 == "1")                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def token_passthrough_enabled() -> bool:
        """云侧诊断旁路总开关:SKIP_SAMPLE(边免采样)/ DISABLE_DOWN
        (全直通)任一开启即 True。"""
        return (
            os.getenv(_LWD_ENV_PREFIX + "EDGE_SKIP_SAMPLE") == "1"
            or os.getenv(_LWD_ENV_PREFIX + "DISABLE_DOWN") == "1"
        )

    @staticmethod
    def disable_down_enabled() -> bool:
        """DISABLE_DOWN 诊断开关:云不 collect/不发 DOWN,边直通交付。"""
        return os.getenv(_LWD_ENV_PREFIX + "DISABLE_DOWN") == "1"

    # ------------------------------------------------------------------ #
    # debug 打点转发(lwd_debug 在 lwd_control 内的唯一进口)            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def set_debug(enabled: bool) -> None:
        """层日志总开关(env 已开则不动,config 段开则补开,仅本进程)。"""
        _, _, log_base = _lwd_debug_mods()
        log_base.set_debug(enabled)

    @staticmethod
    def edge_step_activity(running: int, awaiting: int) -> None:
        """边侧每步活力度打点(running 数 / decode 相位驻留数)。"""
        control_log, _, _ = _lwd_debug_mods()
        control_log.flight(running, awaiting)

    @staticmethod
    def edge_tokens_delivered(
        request_id, sampled_token_ids, finish_reason, vllm_config
    ) -> None:
        """边侧逐请求交付打点。"""
        _, debug, _ = _lwd_debug_mods()
        debug.edge_tokens_delivered(
            request_id, sampled_token_ids, finish_reason, vllm_config
        )

    @staticmethod
    def cloud_step(scheduler, meta, engine_core_outputs) -> None:
        """云侧步元数据打点。"""
        _, debug, _ = _lwd_debug_mods()
        debug.cloud_step(scheduler, meta, engine_core_outputs)

    @staticmethod
    def cloud_request_admitted(wire, sampling_params) -> None:
        """云侧请求过门打点。"""
        _, debug, _ = _lwd_debug_mods()
        debug.cloud_request_admitted(wire, sampling_params)

    # ------------------------------------------------------------------ #
    # spec 接受率观测(累计器住此类,原生同口径)                       #
    # ------------------------------------------------------------------ #
    def __init__(self) -> None:
        self._spec_draft_tokens = 0
        self._spec_accepted = 0
        self._spec_last_log = time.monotonic()

    def observe_spec_acceptance(
        self, scheduler_output, model_runner_output, requests
    ) -> None:
        """按原生口径累计草稿/接受数(num_draft = len(spec_ids),
        num_accepted = len(generated) - 1 含 bonus 减一剔除非草稿行),
        周期直出运行总计;跳过已终结/未知请求(与原生同款过滤)。"""
        spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        if not spec_tokens:
            return
        sampled = model_runner_output.sampled_token_ids
        req_id_to_index = model_runner_output.req_id_to_index
        for req_id, spec_ids in spec_tokens.items():
            if not spec_ids:
                continue
            request = requests.get(req_id)
            if request is None or request.is_finished():
                continue
            req_index = req_id_to_index.get(req_id)
            if req_index is None:
                continue
            generated = sampled[req_index] if sampled else []
            if not generated:
                continue
            self._spec_draft_tokens += len(spec_ids)
            self._spec_accepted += len(generated) - 1
        now = time.monotonic()
        if now - self._spec_last_log < _SPEC_LOG_INTERVAL_S:
            return
        self._spec_last_log = now
        if not self._spec_draft_tokens:
            return
        from vllm.lwd_logger import lwd_logger

        rate = self._spec_accepted / self._spec_draft_tokens * 100
        lwd_logger.info(
            "[spec] cumulative: rate=%.1f%% accepted=%d "
            "draft_tokens=%d",
            rate, self._spec_accepted, self._spec_draft_tokens,
        )

    # ------------------------------------------------------------------ #
    # 诊断旁路载荷                                                        #
    # ------------------------------------------------------------------ #
    @staticmethod
    def step_token_ids(engine_core_outputs) -> dict[str, list[int]]:
        """本步逐请求 new_token_ids(request_id 索引),旁路载荷共用。"""
        ids_by_req: dict[str, list[int]] = {}
        for outputs in engine_core_outputs.values():
            for out in outputs.outputs:
                ids_by_req[out.request_id] = list(out.new_token_ids)
        return ids_by_req

    @staticmethod
    def bypass_token_payload(
        req_ids: list[str], engine_core_outputs
    ) -> list[list[int]]:
        """SKIP_SAMPLE 旁路:按 c2e 行集逐请求 accepted token ids,
        缺席行补空(与 req_ids 按位对齐)。"""
        ids_by_req = LwdControlDiag.step_token_ids(engine_core_outputs)
        return [ids_by_req.get(rid, []) for rid in req_ids]

    @staticmethod
    def disable_down_rows(engine_core_outputs) -> dict[str, list[int]]:
        """DISABLE_DOWN 直通行集:本步逐请求 token + 仅进
        finished_requests 的请求补空行(携带 ABORT 完成码);空步为空
        dict(无可交付内容,不发空通告)。"""
        ids_by_req = LwdControlDiag.step_token_ids(engine_core_outputs)
        for outputs in engine_core_outputs.values():
            for rid in outputs.finished_requests or ():
                ids_by_req.setdefault(rid, [])
        return ids_by_req

    @staticmethod
    def bypass_unembed_rows(batch) -> "tuple[list[str], list[list[int]]] | None":
        """DISABLE_DOWN 直通判定与行集:UNEMBED 批且通告携带旁路
        token(hidden=0)时返回 (req_ids, 旁路 token 行),否则 None。
        只出数据不出执行适配器——直通 future/伪输出由边引擎包装。"""
        if batch is None:
            return None
        from vllm.v1.core.sched.output import LwdBatchType

        if (
            batch.batch_type == LwdBatchType.LWD_UNEMBED
            and LwdControlDiag.disable_down_enabled()
            and batch.batch_meta.token_ids
        ):
            meta = batch.batch_meta
            return (
                list(meta.req_ids),
                [list(t) for t in meta.token_ids],
            )
        return None

    # ------------------------------------------------------------------ #
    # EOS 尾部 kill 根因定位([Lwd][eos-trace],正常流量零输出)         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def zero_accept_rows(
        req_ids: list[str],
        accepted: list[int],
        finish_reasons: list[int],
    ) -> list[tuple[str, int, int]]:
        """提取零接受行 (rid, accepted=0, finish_code)——云组行病灶的
        判定数据:活行零接受(accepted=0 且 finish=-1)即 kill 候选。"""
        return [
            (rid, acc, fin)
            for rid, acc, fin in zip(req_ids, accepted, finish_reasons)
            if acc == 0
        ]

    @staticmethod
    def trace_cloud_c2e_rows(
        seqno: int,
        req_ids: list[str],
        accepted: list[int],
        finish_reasons: list[int],
    ) -> None:
        """[Lwd][eos-trace] A 点·云侧组 c2e 时:存在零接受行才留痕
        (MTP 正常步每行至少接受根 token,零接受≈0,日志量可控)。"""
        rows = LwdControlDiag.zero_accept_rows(
            req_ids, accepted, finish_reasons
        )
        if rows:
            from vllm.lwd_logger import lwd_logger

            lwd_logger.info(
                "[eos-trace] cloud seqno=%s zero-accept rows "
                "(rid,acc,fin)=%s",
                seqno, rows,
            )

    @staticmethod
    def trace_edge_kill_row(
        seqno: int,
        rid: str,
        accepted: int,
        finish_code: int,
        batch_rows: int,
        sampled_rows: int,
    ) -> None:
        """[Lwd][eos-trace] B 点·边侧 ERROR 转换时:按 down_seqno 与
        A/C 日志配对。accepted 与 sampled_rows 是判定主证——
        accepted=0 → 云组行(零接受活行进了行集);accepted>0 且
        sampled_rows>0 → worker 单行缺;sampled_rows=0 → 整批空回填。"""
        from vllm.lwd_logger import lwd_logger

        lwd_logger.info(
            "[eos-trace] edge kill seqno=%s rid=%s accepted=%d "
            "finish=%d batch_rows=%d sampled_rows=%d",
            seqno, rid, accepted, finish_code, batch_rows, sampled_rows,
        )

    @staticmethod
    def unembed_missing_rows(
        req_ids: list[str],
        accepts: list[int],
        result_req_ids: list[str],
        sampled_token_ids: list[list[int]],
    ) -> list[str]:
        """worker 回填缺行判定:声明 accepted>0 的行却无 token 输出
        (不在回填行集或 token 列表为空)——C 点数据。accepted=0 的行
        worker 不产出属合法,不计缺。"""
        filled = {
            rid: toks
            for rid, toks in zip(result_req_ids, sampled_token_ids)
        }
        return [
            rid
            for rid, acc in zip(req_ids, accepts)
            if acc > 0 and not filled.get(rid)
        ]

    @staticmethod
    def trace_worker_missing_rows(
        seqno: int,
        req_ids: list[str],
        accepts: list[int],
        result_req_ids: list[str],
        sampled_token_ids: list[list[int]],
    ) -> None:
        """[Lwd][eos-trace] C 点·worker 回填后:声明有产出(accepted>0)
        却没交 token 的行——出现即 worker 回填缺行坐实。"""
        missing = LwdControlDiag.unembed_missing_rows(
            req_ids, accepts, result_req_ids, sampled_token_ids
        )
        if missing:
            from vllm.lwd_logger import lwd_logger

            lwd_logger.info(
                "[eos-trace] worker seqno=%s missing rows=%s",
                seqno, missing,
            )

