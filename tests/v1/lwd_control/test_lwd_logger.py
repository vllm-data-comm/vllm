"""vllm.lwd_logger(自 main 1605d817cd 字节一致移植)接口契约测试。

lwd_control 全量日志已切换到本接口(评审:日志换成最新合入的 lwd
日志接口),此处锚定三条契约:[lwd] 前缀、默认静默(DISABLE)、不向
root 传播(避免与 vllm.logger 双发)。完整链需 vllm.envs(Python>=3.10),
轻量环境模块级跳过。
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

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
        "hello %s", ("world",), None,
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


@pytest.mark.parametrize("lwd_level", [None, "DEBUG"])
def test_native_reconfiguration_preserves_independent_lwd_output(
    lwd_level: str | None,
) -> None:
    env = os.environ.copy()
    env.update(
        VLLM_CONFIGURE_LOGGING="1",
        VLLM_LOGGING_LEVEL="ERROR",
        VLLM_LOGGING_STREAM="ext://sys.stdout",
        VLLM_LOGGING_COLOR="0",
        VLLM_LOGGING_PREFIX="",
        NO_COLOR="1",
    )
    env.pop("VLLM_LWD_LOG_LEVEL", None)
    env.pop("VLLM_LOGGING_CONFIG_PATH", None)
    if lwd_level is not None:
        env["VLLM_LWD_LOG_LEVEL"] = lwd_level
    script = """
import vllm.logger as native
import vllm.lwd_logger as lwd

names = ('vllm.lwd', 'vllm.v1.lwd_control.reconfiguration_probe')
loggers = [lwd.init_lwd_logger(name) for name in names]
for iteration in range(2):
    native._configure_vllm_root_logger()
    for index, logger in enumerate(loggers):
        logger.debug('debug-probe-%d-%d', iteration, index)
        logger.critical('critical-probe-%d-%d', iteration, index)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected_count = 4 if lwd_level == "DEBUG" else 0
    for marker in ("debug-probe-", "critical-probe-"):
        assert result.stdout.count(marker) == expected_count
        assert result.stdout.count("[lwd] " + marker) == expected_count
