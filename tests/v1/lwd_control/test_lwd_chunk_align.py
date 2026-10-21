# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""chunk 对齐 profile 测试(2026-10 lwd-debug-1019 INJECT window mismatch
回归):采纳/校验逻辑任何解释器可跑;结构体字段表、引擎接线、上游
split 属性读取面用 AST 源码守卫冻结;真实 `_mamba_block_aligned_split`
等价复现为表驱动执行用例(需 torch + vllm 可导入,轻量环境按条件跳过,
真实环境补跑)。

事故形态:边侧 chunk 划分权威,云侧按钳制参数复现;但云侧原生
schedule() 的 mamba 块对齐把 1561 截到 1536(块 128 对齐),边侧按
1561 发 embeds -> 云侧 INJECT 窗口校验 rows=1561 != scheduled=1536
崩溃。修法:LwdCloudParamsNotify 下发云侧对齐 profile,边侧采纳到
调度器,边侧原生 schedule() 跑同一段上游代码完成同款截断。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.v1.lwd_control.control_scheduler.lwd_chunk_align import (
    LWD_CHUNK_ALIGN_ADOPT_ATTRS,
    lwd_apply_chunk_align_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MESSAGES = REPO_ROOT / "vllm" / "v1" / "lwd_control" / "lwd_comm" / "lwd_messages.py"
SCHEDULER = REPO_ROOT / "vllm" / "v1" / "core" / "sched" / "scheduler.py"
CLOUD_ENGINE = (
    REPO_ROOT / "vllm" / "v1" / "lwd_control" / "control_cloud_scheduler"
    / "lwd_cloud_engine.py"
)
EDGE_ENGINE = (
    REPO_ROOT / "vllm" / "v1" / "lwd_control" / "control_edge_scheduler"
    / "lwd_edge_engine.py"
)


def make_profile(**overrides) -> SimpleNamespace:
    """云侧实况值(Qwen3.6-27B + qwen3_5_mtp 部署形态)的 stub。"""
    fields = {
        "block_size": 128,
        "hash_block_size": 128,
        "need_mamba_block_aligned_split": True,
        "mamba_prefill_checkpoint_alignment": None,
        "mamba_has_prefill_checkpoint_blocks": False,
        "mamba_partial_cache_hit": False,
        "mamba_fine_grained_prefix_cache": False,
        "use_eagle_block_drop": True,
        "num_prefill_lookahead": 3,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def make_edge_scheduler() -> SimpleNamespace:
    """边侧调度器 stub:init 派生值全是"无 mamba/无 spec"的边侧形态。"""
    sched = SimpleNamespace(
        hash_block_size=128,
        need_mamba_block_aligned_split=False,
        mamba_prefill_checkpoint_alignment=None,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_partial_cache_hit=False,
        mamba_fine_grained_prefix_cache=False,
        use_eagle_block_drop=False,
        num_prefill_lookahead=0,
    )
    return sched


def make_cache(block_size: int = 128) -> SimpleNamespace:
    return SimpleNamespace(block_size=block_size)


def test_apply_stamps_all_adopt_attrs():
    """采纳:7 个属性全部盖成云侧值;hash 尺寸保持边侧原值(仅校验)。"""
    sched = make_edge_scheduler()
    profile = make_profile()
    lwd_apply_chunk_align_profile(sched, make_cache(), profile)
    for name in LWD_CHUNK_ALIGN_ADOPT_ATTRS:
        assert getattr(sched, name) == getattr(profile, name), name
    assert sched.hash_block_size == 128


def test_apply_block_size_mismatch_raises():
    """块尺寸不一致 = 部署错位,fail-fast 而非静默采纳错值。"""
    sched = make_edge_scheduler()
    with pytest.raises(RuntimeError, match="block size mismatch"):
        lwd_apply_chunk_align_profile(sched, make_cache(16), make_profile())
    # 失败路径不得部分盖章
    assert sched.need_mamba_block_aligned_split is False


def test_apply_hash_block_size_mismatch_raises():
    sched = make_edge_scheduler()
    sched.hash_block_size = 64
    with pytest.raises(RuntimeError, match="hash block size mismatch"):
        lwd_apply_chunk_align_profile(sched, make_cache(), make_profile())


# --------------------------------------------------------------------- #
# AST 源码守卫:任何主机可跑                                            #
# --------------------------------------------------------------------- #

def _struct_fields(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    return {
        stmt.target.id
        for stmt in node.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    }


def test_struct_fields_match_adopt_attrs():
    """结构体字段 == 校验位(block/hash) + 采纳位,两侧不许漂移。"""
    fields = _struct_fields(MESSAGES, "LwdChunkAlignProfile")
    assert fields == {"block_size", "hash_block_size"} | set(LWD_CHUNK_ALIGN_ADOPT_ATTRS)


def test_cloud_params_chunk_align_defaults_to_none():
    """chunk_align 带默认 None:旧云编码可解码,边侧据此报版本错位。"""
    tree = ast.parse(MESSAGES.read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LwdCloudParamsNotify"
    )
    fields = {
        stmt.target.id: stmt for stmt in node.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    }
    assert "chunk_align" in fields
    assert fields["chunk_align"].value is not None, "chunk_align 必须带默认值"


def test_engines_wire_profile():
    """接线守卫:云侧必填、边侧必采纳且对 None fail-fast。"""
    cloud = CLOUD_ENGINE.read_text(encoding="utf-8")
    assert "lwd_build_chunk_align_profile(" in cloud
    assert "self.scheduler" in cloud and "cache_config" in cloud
    edge = EDGE_ENGINE.read_text(encoding="utf-8")
    assert "lwd_apply_chunk_align_profile(" in edge
    assert re.search(r"chunk_align is None", edge), "缺 None fail-fast"


def test_split_attr_reads_covered_by_profile():
    """漂移绊线:上游 split 新读任何 self 属性都必须进 profile 面。

    `_mamba_block_aligned_split` 只许读:上下文(cache_config/
    scheduler_config/max_num_scheduled_tokens)、校验位 hash_block_size、
    采纳位。出现新属性 = 云侧又多一个边侧不知道的截断输入,本守卫
    先于现场崩溃报警。
    """
    tree = ast.parse(SCHEDULER.read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_mamba_block_aligned_split"
    )
    body = ast.get_source_segment(SCHEDULER.read_text(encoding="utf-8"), fn)
    reads = set(re.findall(r"self\.(\w+)", body))
    allowed = (
        {"cache_config", "scheduler_config", "max_num_scheduled_tokens",
         "hash_block_size"}
        | set(LWD_CHUNK_ALIGN_ADOPT_ATTRS)
    )
    assert reads <= allowed, f"split 新读属性 {reads - allowed}: 需扩 profile"


# --------------------------------------------------------------------- #
# 真实 split 等价复现(需 torch;轻量环境跳过,真实环境补跑)            #
# --------------------------------------------------------------------- #

try:
    from vllm.v1.core.sched.scheduler import Scheduler as _Scheduler
    SCHEDULER_CLS = _Scheduler
except Exception:  # pragma: no cover
    SCHEDULER_CLS = None  # type: ignore[assignment]

needs_scheduler = pytest.mark.skipif(
    SCHEDULER_CLS is None,
    reason="需真实环境(Python>=3.10 + torch,vllm.v1.core.sched 可导入)补跑",
)


def _split_self(profile: SimpleNamespace, threshold: int = 8192) -> SimpleNamespace:
    """携带 split 读取面全量属性的 stub self(属性集由守卫测试冻结)。"""
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=profile.block_size),
        hash_block_size=profile.hash_block_size,
        max_num_scheduled_tokens=8192,
        scheduler_config=SimpleNamespace(
            long_prefill_token_threshold=threshold
        ),
        **{name: getattr(profile, name) for name in LWD_CHUNK_ALIGN_ADOPT_ATTRS},
    )


@needs_scheduler
@pytest.mark.parametrize(
    ("prompt_len", "start", "num_new", "want", "profile_overrides"),
    [
        # 事故实录:mid-prompt 预算截断块 1561 -> 128 对齐截 1536
        (4096, 0, 1561, 1536, {}),
        # 已对齐的 chunk 原样通过
        (4096, 0, 2048, 2048, {}),
        # 块内起点:先停到下一块边界(前缀命中后起点不入块的形态)
        (4096, 100, 2048, 28, {}),
        # 末块豁免对齐截断,但 eagle 退一块的 last_cache_position 停点咬合
        # (1561%128=25:last=1536,eagle 退一块 -> 1408)
        (1561, 0, 1561, 1408, {}),
        # 同上,关 eagle 退块:停 1536,余 25 逐块收尾(锁定退块差量)
        (1561, 0, 1561, 1536, {"use_eagle_block_drop": False}),
        # 从退块位置续跑的末块:停点不严格落段内,整段放行
        (1600, 1408, 192, 192, {}),
        # 块内起点的末块:逐块 stepping 到边界
        (1600, 1440, 160, 96, {}),
        # 剩余额度不足跨块:对齐后为 0(本步跳过,下步整块再排)
        (4096, 0, 100, 0, {}),
    ],
)
def test_split_reproduces_cloud_cuts(
    prompt_len, start, num_new, want, profile_overrides
):
    """云侧原生值 vs 边侧采纳值跑同一段上游 split,输出必须相等且等于
    表值——边侧从此在发 embeds 之前就切出云侧将排的窗口。"""
    profile = make_profile(**profile_overrides)
    request = SimpleNamespace(
        num_computed_tokens=start,
        num_tokens=prompt_len,
        num_prompt_tokens=prompt_len,
        shared_prefix_boundary=0,
    )
    cloud_sched = _split_self(profile)
    edge_sched = make_edge_scheduler()
    lwd_apply_chunk_align_profile(edge_sched, make_cache(), profile)
    edge_sched.cache_config = SimpleNamespace(block_size=profile.block_size)
    edge_sched.max_num_scheduled_tokens = 8192
    edge_sched.scheduler_config = SimpleNamespace(
        long_prefill_token_threshold=8192
    )
    cloud_cut = SCHEDULER_CLS._mamba_block_aligned_split(
        cloud_sched, request, num_new
    )
    edge_cut = SCHEDULER_CLS._mamba_block_aligned_split(
        edge_sched, request, num_new
    )
    assert cloud_cut == want, (cloud_cut, want)
    assert edge_cut == cloud_cut


@needs_scheduler
@pytest.mark.parametrize(
    ("num_tokens", "computed", "num_new", "lookahead", "want"),
    [
        # 多模组 MTP:chunk 端点距 prefill 末 2 < lookahead 3 -> 收 1
        (1538, 0, 1536, 3, 1535),
        # 端点即末尾(remaining=0)-> 放行
        (1538, 0, 1538, 3, 1538),
        # 边侧无 spec(lookahead=0,采纳前形态)-> 放行;采纳后与云同款收法
        (1538, 0, 1536, 0, 1536),
    ],
)
def test_reserve_prefill_lookahead_reproduces(
    num_tokens, computed, num_new, lookahead, want
):
    """lookahead 预留随 profile 采纳:边侧与云侧同式收 chunk。"""
    request = SimpleNamespace(
        num_tokens=num_tokens,
        num_computed_tokens=computed,
        num_prompt_tokens=num_tokens,
        shared_prefix_boundary=0,
    )
    sched = SimpleNamespace(num_prefill_lookahead=lookahead)
    assert (
        SCHEDULER_CLS._reserve_prefill_lookahead(sched, request, computed, num_new)
        == want
    )
