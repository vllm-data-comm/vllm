"""[Lwd][eos-trace] 三点定位日志的判定函数测试。

判定矩阵(按 down_seqno 配对 A/B/C 三点):
  B.accepted=0 + A 有该行        → 云组行(零接受活行进了行集)
  B.accepted>0 + C 有该行        → worker 回填缺行
  B.accepted>0 + C 无 + 整批空   → 中间层/空回填路径
"""

from __future__ import annotations

import pytest

from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag

_NOT_FINISHED = -1


def test_zero_accept_rows_extraction():
    """A 点:只提 accepted==0 的行,带 finish 码供活/终结算例。"""
    rows = LwdControlDiag.zero_accept_rows(
        ["a", "b", "c"], [0, 3, 0], [_NOT_FINISHED, _NOT_FINISHED, 0]
    )
    assert rows == [("a", 0, _NOT_FINISHED), ("c", 0, 0)]


def test_zero_accept_rows_empty_when_none():
    assert LwdControlDiag.zero_accept_rows(
        ["a"], [1], [_NOT_FINISHED]
    ) == []


def test_unembed_missing_rows():
    """C 点:声明 accepted>0 却无 token 回填才计缺;accepted=0 不产
    token 属合法,回填列表空也属合法。"""
    # b 声明 2 个却缺 → 计缺;a 声明 0 无回填 → 合法;c 回填空列表 → 计缺
    missing = LwdControlDiag.unembed_missing_rows(
        req_ids=["a", "b", "c"],
        accepts=[0, 2, 1],
        result_req_ids=["a", "b"],
        sampled_token_ids=[[], [9]],  # a 空列表(accepted=0 合法),b 有
    )
    assert missing == ["c"]  # c 声明 1 却不在回填行集

    # 全交齐 → 无缺
    assert LwdControlDiag.unembed_missing_rows(
        ["a"], [2], ["a"], [[1, 2]]
    ) == []


def test_trace_callables_no_crash(monkeypatch):
    """三点日志函数:正常流量(零接受行/无缺行)零输出不炸;有病灶时
    走 lwd_logger 打点(此处仅验证调用安全,不验日志文本)。"""
    try:
        import vllm.lwd_logger  # noqa: F401
    except Exception:  # pragma: no cover - 轻量环境(见 conftest)
        pytest.skip("vllm.lwd_logger 完整链不可用(需 Python>=3.10)")
    monkeypatch.delenv("VLLM_ASCEND_LWD_DISABLE_DOWN", raising=False)
    # A:无零接受行 → 不打
    LwdControlDiag.trace_cloud_c2e_rows(7, ["a"], [1], [_NOT_FINISHED])
    # A:有 → 打
    LwdControlDiag.trace_cloud_c2e_rows(7, ["a"], [0], [_NOT_FINISHED])
    # B:直接调
    LwdControlDiag.trace_edge_kill_row(
        seqno=7, rid="a", accepted=0, finish_code=_NOT_FINISHED,
        batch_rows=3, sampled_rows=2,
    )
    # C:无缺/有缺
    LwdControlDiag.trace_worker_missing_rows(
        7, ["a"], [1], ["a"], [[5]]
    )
    LwdControlDiag.trace_worker_missing_rows(
        7, ["a"], [1], [], []
    )
