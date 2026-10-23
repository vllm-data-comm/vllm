# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""配置面适配测试:lwd_spec_factor 表驱动 + 旧配置面清零守卫。

对齐 main 的配置面(PR #4,resolve_lwd_config/init-guard)后的语义对当:
- 引擎选择判据从旧 `lwd.enabled / lwd.mode / lwd.is_edge`(未随本 PR
  移植的 vllm/config/lwd.py)改为 `lwd_config is not None` + `role`
  (显式给 lwd_config 即 LWD 开,schema 限 role ∈ {edge, cloud});
- `lwd_spec_factor` 自旧 vllm/config/lwd.py 移入 lwd_base_scheduler
  (新配置面是拓扑快照,无此派生值),语义不变:无 spec 恒 1,有 spec
  为 num_speculative_tokens + 1(None/0 同样折算为 1)。

lwd_base_scheduler 完整链需 torch/Python>=3.10;轻量环境表驱动用例
用例内跳过,源码扫描守卫照跑。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
VLLM_DIR = REPO_ROOT / "vllm"


def test_core_selects_lwd_engine_by_lwd_config_role():
    """core.py turns LWD on when `vllm_config.lwd_config` is given and picks
    the engine by its role, without importing the config package; the
    retired `lwd.enabled/mode/is_edge` gate stays gone."""
    core = (VLLM_DIR / "v1" / "engine" / "core.py").read_text()
    assert "from vllm.config.lwd import" not in core
    for retired in ("lwd.enabled", "lwd.mode", "lwd.is_edge"):
        assert retired not in core, retired
    assert "vllm_config.lwd_config" in core
    assert "if lwd is not None:" in core
    assert 'lwd.role == "edge"' in core


def test_lwd_spec_factor_single_home():
    """lwd_spec_factor 唯一住处 = lwd_base_scheduler(语义校验见同模块
    表驱动用例;此处文本守卫保证不回流旧配置面)。"""
    base = (
        VLLM_DIR / "v1" / "lwd_control" / "control_scheduler" / "lwd_base_scheduler.py"
    ).read_text()
    assert base.count("def lwd_spec_factor(") == 1


def test_lwd_spec_factor_table():
    """表驱动:无 spec 配置/缺属性/None/0 恒 1;有 spec 为 k+1。
    语义对当:与旧 vllm/config/lwd.lwd_spec_factor 完全一致(部署上
    0 草稿与无配置等价,均不占额外预算)。"""
    try:
        from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
            lwd_spec_factor,
        )
    except Exception:  # pragma: no cover - 轻量环境(见 conftest)
        pytest.skip("lwd_base_scheduler 完整链不可用(需 torch/Python>=3.10)")
    table = [
        # (vllm_config, expected)
        (SimpleNamespace(speculative_config=None), 1),
        (SimpleNamespace(), 1),
        (
            SimpleNamespace(
                speculative_config=SimpleNamespace(num_speculative_tokens=None)
            ),
            1,
        ),
        (
            SimpleNamespace(
                speculative_config=SimpleNamespace(num_speculative_tokens=0)
            ),
            1,
        ),
        (
            SimpleNamespace(
                speculative_config=SimpleNamespace(num_speculative_tokens=3)
            ),
            4,
        ),
    ]
    for cfg, expected in table:
        assert lwd_spec_factor(cfg) == expected, cfg
