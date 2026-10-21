"""LwdControlDiag 测试:开关语义、旁路载荷提取、spec 观测累计。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag


def test_switches_strict_one_and_live_read(monkeypatch):
    """开关严格 == "1",按调用现读(进程内改 env 即时生效)。"""
    monkeypatch.delenv("VLLM_ASCEND_LWD_EDGE_SKIP_SAMPLE", raising=False)
    monkeypatch.delenv("VLLM_ASCEND_LWD_DISABLE_DOWN", raising=False)
    assert LwdControlDiag.token_passthrough_enabled() is False
    assert LwdControlDiag.disable_down_enabled() is False
    # 非 "1" 一律 False
    monkeypatch.setenv("VLLM_ASCEND_LWD_EDGE_SKIP_SAMPLE", "true")
    assert LwdControlDiag.token_passthrough_enabled() is False
    # 现读语义 + OR 关系
    monkeypatch.setenv("VLLM_ASCEND_LWD_DISABLE_DOWN", "1")
    assert LwdControlDiag.disable_down_enabled() is True
    assert LwdControlDiag.token_passthrough_enabled() is True
    monkeypatch.setenv("VLLM_ASCEND_LWD_EDGE_SKIP_SAMPLE", "1")
    assert LwdControlDiag.token_passthrough_enabled() is True


def _outputs(rows: dict[str, list[int]], finished=()):
    """伪 engine_core_outputs:{0: EngineCoreOutputs(outputs, finished)}。"""
    outputs = [
        SimpleNamespace(request_id=rid, new_token_ids=toks, finish_reason=None)
        for rid, toks in rows.items()
    ]
    return {0: SimpleNamespace(outputs=outputs, finished_requests=list(finished))}


def test_bypass_token_payload_aligned_rows():
    """SKIP_SAMPLE 载荷:按 c2e 行集对齐,缺席行补空。"""
    outputs = _outputs({"a": [1, 2], "b": [3]})
    payload = LwdControlDiag.bypass_token_payload(["a", "zzz"], outputs)
    assert payload == [[1, 2], []]


def test_disable_down_rows_pads_finished():
    """DISABLE_DOWN 行集:本步 token + finished 补空行;空步为空。"""
    outputs = _outputs({"a": [7]}, finished=["b"])
    rows = LwdControlDiag.disable_down_rows(outputs)
    assert rows == {"a": [7], "b": []}
    assert LwdControlDiag.disable_down_rows(_outputs({})) == {}


def test_spec_observation_accumulates_native_formula():
    """spec 观测按原生口径累计:draft += len(spec_ids),
    accepted += len(generated) - 1;未知/已终结请求跳过。"""
    diag = LwdControlDiag()
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={"a": [1, 2, 3], "gone": [9], "fin": [9]}
    )
    model_output = SimpleNamespace(
        sampled_token_ids=[[0, 4, 5, 6]],
        req_id_to_index={"a": 0},
    )
    requests = {
        "a": SimpleNamespace(is_finished=lambda: False),
        "fin": SimpleNamespace(is_finished=lambda: True),
    }
    diag.observe_spec_acceptance(scheduler_output, model_output, requests)
    # a: draft 3, accepted = len([0,4,5,6]) - 1 = 3;gone 未知/fin 已终结不计
    assert diag._spec_draft_tokens == 3
    assert diag._spec_accepted == 3
    # 无 spec token 的步:零开销返回
    diag.observe_spec_acceptance(
        SimpleNamespace(scheduled_spec_decode_tokens={}), model_output, requests
    )
    assert diag._spec_draft_tokens == 3


def test_bypass_unembed_rows_gating(monkeypatch):
    """直通判定:UNEMBED 批 + 开关开 + 通告携带 token 才出直通行集;
    只出 (req_ids, 旁路 token) 数据,执行适配器归边引擎。"""
    try:
        from vllm.v1.core.sched.output import LwdBatchType
    except Exception:  # pragma: no cover - py<3.10 轻量环境
        pytest.skip("sched.output 不可导入(需 Python>=3.10)")

    batch = SimpleNamespace(
        batch_type=LwdBatchType.LWD_UNEMBED,
        batch_meta=SimpleNamespace(req_ids=["a"], token_ids=[[1, 2]]),
    )
    monkeypatch.delenv("VLLM_ASCEND_LWD_DISABLE_DOWN", raising=False)
    assert LwdControlDiag.bypass_unembed_rows(batch) is None
    assert LwdControlDiag.bypass_unembed_rows(None) is None
    monkeypatch.setenv("VLLM_ASCEND_LWD_DISABLE_DOWN", "1")
    assert LwdControlDiag.bypass_unembed_rows(batch) == (["a"], [[1, 2]])
    # 无旁路 token 的 UNEMBED 批(正常批)不旁路
    empty = SimpleNamespace(
        batch_type=LwdBatchType.LWD_UNEMBED,
        batch_meta=SimpleNamespace(req_ids=["a"], token_ids=[]),
    )
    assert LwdControlDiag.bypass_unembed_rows(empty) is None


def test_debug_forwarding_degrades_to_noop_without_lwd_debug_package():
    """断点4回归(issue #18):vllm.v1.lwd_debug 未迁入 main 时,打点转发
    降级为 no-op 而不是 ModuleNotFoundError——首个请求/首个 step 必然
    经过这些转发点。包迁入后同一路径自动恢复真实打点。"""
    from vllm.v1.lwd_control.lwd_control_diag.lwd_control_diag import (
        _lwd_debug_mods,
    )

    control_log, debug, log_base = _lwd_debug_mods()
    # 无论降级与否,三个入口都必须可调用且不抛异常
    log_base.set_debug(True)
    control_log.flight(3, 1)
    debug.edge_tokens_delivered("req-1", [1, 2, 3], None, None)
    debug.cloud_step(None, None, None)
    debug.cloud_request_admitted(None, None)
    # 功能侧转发点同样不抛
    LwdControlDiag.set_debug(True)
    LwdControlDiag.edge_step_activity(3, 1)
    LwdControlDiag.cloud_request_admitted(None, None)
