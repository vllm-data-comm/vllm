"""vllm.lwd_logger(自 main 1605d817cd 字节一致移植)接口契约测试。

lwd_control 全量日志已切换到本接口(评审:日志换成最新合入的 lwd
日志接口),此处锚定三条契约:[lwd] 前缀、默认静默(DISABLE)、不向
root 传播(避免与 vllm.logger 双发)。完整链需 vllm.envs(Python>=3.10),
轻量环境模块级跳过。
"""

from __future__ import annotations

import logging

import pytest

try:
    import vllm.lwd_logger as lwd_logger_mod
except Exception:  # pragma: no cover - 轻量环境(见 conftest)
    pytest.skip(
        "vllm.lwd_logger 完整链不可用(需 Python>=3.10)",
        allow_module_level=True,
    )


def test_singleton_silent_by_default_and_no_propagate():
    """默认 VLLM_LWD_LOG_LEVEL=DISABLE:单例级别高于 CRITICAL(info 静默),
    且 propagate=False(不与全局 vllm.logger 双发)。"""
    lwd_logger = lwd_logger_mod.lwd_logger
    assert lwd_logger.propagate is False
    assert lwd_logger.getEffectiveLevel() > logging.CRITICAL
    assert lwd_logger.handlers, "至少挂一个 StreamHandler"


def test_prefix_filter_prepends_lwd_tag():
    """_LwdPrefixFilter 契约:消息前缀 "[lwd] ",% 参数就地消费。"""
    record = lwd_logger_mod.lwd_logger.makeRecord(
        "vllm.lwd", logging.INFO, __file__, 1,
        "hello %s", ("world", None), None,
    )
    for handler in lwd_logger_mod.lwd_logger.handlers:
        for flt in handler.filters:
            if isinstance(flt, lwd_logger_mod._LwdPrefixFilter):
                assert flt.filter(record) is True
    assert record.getMessage().startswith("[lwd] ")
    assert "hello world" in record.getMessage()
    assert record.args == ()


def test_level_table_disable_above_critical():
    """级别表:disable 映射为 CRITICAL+1(完全静默档),其余取标准值。"""
    levels = lwd_logger_mod._LWD_LOG_LEVELS
    assert levels["DISABLE"] == logging.CRITICAL + 1
    assert levels["INFO"] == logging.INFO
