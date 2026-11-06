# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight logger independent of the global VLLM_LOGGING_LEVEL.

Provides `lwd_logger`, a drop-in replacement for `vllm.logger.logger` whose
level is controlled by VLLM_LWD_LOG_LEVEL instead of VLLM_LOGGING_LEVEL.
Every message is prefixed with "[lwd] ".
"""

import logging
import sys

from vllm import envs
from vllm.logging_utils import ColoredFormatter, NewLineFormatter

_FORMAT = (
    f"{envs.VLLM_LOGGING_PREFIX}%(levelname)s %(asctime)s "
    "[%(fileinfo)s:%(lineno)d] %(message)s"
)
_DATE_FORMAT = "%m-%d %H:%M:%S"

_LWD_LOG_LEVELS = {
    "DISABLE": logging.CRITICAL + 1,
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
# Cache factory-owned levels once so native dictConfig cannot replace them.
_LWD_LOGGERS: dict[str, int] = {}


def _use_color() -> bool:
    if envs.NO_COLOR or envs.VLLM_LOGGING_COLOR == "0":
        return False
    if envs.VLLM_LOGGING_COLOR == "1":
        return True
    if envs.VLLM_LOGGING_STREAM == "ext://sys.stdout":
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    if envs.VLLM_LOGGING_STREAM == "ext://sys.stderr":
        return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    return False


class _LwdPrefixFilter(logging.Filter):
    """Prepend "[lwd] " to every message emitted by the lwd logger."""

    def filter(self, record):
        record.msg = "[lwd] " + record.getMessage()
        record.args = ()
        return True


def init_lwd_logger(name: str) -> logging.Logger:
    """Return a logger named `name` controlled by VLLM_LWD_LOG_LEVEL.

    Similar to `vllm.logger.init_logger`, callers pass their module name (e.g.
    `__name__`) to obtain a logger. Both the logger and its handler use
    VLLM_LWD_LOG_LEVEL, which is independent of VLLM_LOGGING_LEVEL. Every
    message is prefixed with "[lwd] ". Set VLLM_LWD_LOG_LEVEL to one of
    disable/debug/info/warning/error/critical (default: disable, which prints
    nothing).
    """
    lwd_logger = logging.getLogger(name)
    if name in _LWD_LOGGERS:
        return lwd_logger

    raw_level = envs.VLLM_LWD_LOG_LEVEL
    level = _LWD_LOG_LEVELS.get(raw_level, logging.CRITICAL + 1)
    _configure_lwd_logger(lwd_logger, level)
    _LWD_LOGGERS[name] = level
    return lwd_logger


def _configure_lwd_logger(lwd_logger: logging.Logger, level: int) -> None:
    for handler in lwd_logger.handlers[:]:
        lwd_logger.removeHandler(handler)
        if any(isinstance(flt, _LwdPrefixFilter) for flt in handler.filters):
            handler.close()
    lwd_logger.setLevel(level)
    lwd_logger.propagate = False
    lwd_logger.disabled = False

    stream = (
        sys.stdout
        if envs.VLLM_LOGGING_STREAM == "ext://sys.stdout"
        else sys.stderr
    )
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.addFilter(_LwdPrefixFilter())
    formatter_cls = ColoredFormatter if _use_color() else NewLineFormatter
    handler.setFormatter(formatter_cls(fmt=_FORMAT, datefmt=_DATE_FORMAT))

    lwd_logger.addHandler(handler)


def reconfigure_lwd_loggers() -> None:
    """Restore factory-owned loggers after native logging reconfiguration."""
    for name, level in tuple(_LWD_LOGGERS.items()):
        _configure_lwd_logger(logging.getLogger(name), level)


lwd_logger = init_lwd_logger("vllm.lwd")
