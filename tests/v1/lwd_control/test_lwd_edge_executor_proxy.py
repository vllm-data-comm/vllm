"""LwdEdgeExecutorProxy 测试:None 适配、透传委托、诊断直通门控。

stub/future 家族已封为内部类,本文件同时锚定"内部类不外泄"的
边界(外部只认识代理本身)。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_executor_proxy import (
    LwdEdgeExecutorProxy,
)


class _FakeFuture:
    __slots__ = ("_v",)

    def __init__(self, value):
        self._v = value

    def result(self, timeout=None):
        return self._v


class _FakeExecutor:
    marker = "inner"

    def __init__(self, value=None):
        self._value = value
        self.calls = 0

    def execute_model(self, scheduler_output, *args, **kwargs):
        self.calls += 1
        return _FakeFuture(self._value)


def test_none_result_replaced_by_empty_stub(monkeypatch):
    """契约①:底层正常完成返回 None -> 空占位(收割路径不判失败)。"""
    monkeypatch.delenv("VLLM_ASCEND_LWD_DISABLE_DOWN", raising=False)
    proxy = LwdEdgeExecutorProxy(_FakeExecutor(value=None))
    result = proxy.execute_model(object()).result()
    assert result is LwdEdgeExecutorProxy.EMPTY_MODEL_OUTPUT
    assert result.req_ids == ()
    assert result.sampled_token_ids == ()


def test_value_passes_through(monkeypatch):
    """非 None 结果原样穿透,不吞值。"""
    monkeypatch.delenv("VLLM_ASCEND_LWD_DISABLE_DOWN", raising=False)
    proxy = LwdEdgeExecutorProxy(_FakeExecutor(value="ok"))
    assert proxy.execute_model(object()).result() == "ok"


def test_getattr_delegates_to_inner():
    """契约③:其余属性/调用透传底层 executor。"""
    inner = _FakeExecutor()
    proxy = LwdEdgeExecutorProxy(inner)
    assert proxy.marker == "inner"
    assert proxy.calls == 0


def test_bypass_short_circuits_inner(monkeypatch):
    """契约②:诊断直通批不派 worker(底层零调用),旁路 token 即结果;
    开关关或不携带旁路 token 时照常下派。"""
    try:
        from vllm.v1.core.sched.output import LwdBatchType
    except Exception:  # pragma: no cover - py<3.10 轻量环境
        pytest.skip("sched.output 不可导入(需 Python>=3.10)")

    batch = SimpleNamespace(
        batch_type=LwdBatchType.LWD_UNEMBED,
        batch_meta=SimpleNamespace(req_ids=["a"], token_ids=[[1, 2]]),
    )
    inner = _FakeExecutor(value="should-not-appear")
    proxy = LwdEdgeExecutorProxy(inner)
    monkeypatch.setenv("VLLM_ASCEND_LWD_DISABLE_DOWN", "1")
    result = proxy.execute_model(SimpleNamespace(lwd_batch=batch)).result()
    assert result.req_ids == ["a"]
    assert result.sampled_token_ids == [[1, 2]]
    assert inner.calls == 0  # 未派 worker

    # 开关关:同一批照常下派底层
    monkeypatch.delenv("VLLM_ASCEND_LWD_DISABLE_DOWN")
    assert proxy.execute_model(SimpleNamespace(lwd_batch=batch)).result() == (
        "should-not-appear"
    )
    assert inner.calls == 1


def test_stub_family_is_nested():
    """边界锚点:stub/future 家族是代理的内部类,模块层不外泄同名类。"""
    import vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_executor_proxy as m

    for name in ("StubModelOutput", "NoneSafeFuture", "BypassFuture"):
        assert getattr(LwdEdgeExecutorProxy, name) is not None
        assert not hasattr(m, name), name
