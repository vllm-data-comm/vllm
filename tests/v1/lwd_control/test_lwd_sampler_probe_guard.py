# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observation-only guard for the 2026-10-20 aivec investigation.

Two rules after the reverts: the cloud must NOT neutralize penalties (the
earlier fix was pushed on a since-disproven mechanism — restore behavior
and OBSERVE instead), and the sampler probe must stay installed so the
next fault run reports the kernel inputs first-hand. Pure source scan.
"""

from __future__ import annotations

from pathlib import Path

ENGINE = (
    Path(__file__).resolve().parents[3]
    / "vllm" / "v1" / "lwd_control" / "control_cloud_scheduler"
    / "lwd_cloud_engine.py"
)


def test_penalties_are_forwarded_not_neutralized() -> None:
    src = ENGINE.read_text(encoding="utf-8")
    assert 'values["repetition_penalty"] = 1.0' not in src, (
        "penalty neutralization crept back; the aivec root cause is not "
        "proven - observe with the probe instead of changing behavior"
    )
    assert "wire penalties" in src, "the wire-penalty observation log is gone"


def test_sampler_probe_stays_installed() -> None:
    """探针已移至 worker 进程(ascend 仓 lwd_cloud_worker);引擎文件里
    保留 wire penalties 日志即可(2026-10-20:引擎进程装补丁对 worker 进程
    的模块副本无效,内核实际在 worker 里发射)。"""
    src = ENGINE.read_text(encoding="utf-8")
    assert "wire penalties" in src, "wire penalty observation log is gone"
    assert "lwd_install_sampler_probe" not in src, (
        "sampler probe should live in the worker process, not the engine"
    )


def test_cloud_request_carries_block_hasher() -> None:
    """2026-10-20 IndexError 事故:云侧 Request 无 hasher → block_hashes
    恒空 → 首个满块 cache_full_blocks 越界。必须挂 mm 同款 hasher
    (边侧链优先、本地回退),边侧必须打包链。"""
    src = ENGINE.read_text(encoding="utf-8")
    assert "lwd_cloud_block_hasher(" in src and "req_notify.block_hashes," in src
    chain = (
        Path(__file__).resolve().parents[3]
        / "vllm" / "v1" / "lwd_control" / "lwd_hash_chain.py"
    ).read_text(encoding="utf-8")
    assert "get_request_block_hasher" in chain and "wire" in chain
    edge = (
        Path(__file__).resolve().parents[3]
        / "vllm" / "v1" / "lwd_control" / "control_edge_scheduler"
        / "lwd_edge_engine.py"
    ).read_text(encoding="utf-8")
    assert "block_hashes=list(request.block_hashes)" in edge


def test_hash_chain_imports_resolve(tmp_path=None) -> None:
    """2026-10-20 ImportError 事故:get_hash_fn_by_name 住 vllm.utils.hashing,
    不在 kv_cache_utils(engine/core.py:44 同款来源)。逐个验证模块可解析。"""
    import importlib
    chain = (
        Path(__file__).resolve().parents[3]
        / "vllm" / "v1" / "lwd_control" / "lwd_hash_chain.py"
    ).read_text(encoding="utf-8")
    assert "from vllm.utils.hashing import get_hash_fn_by_name" in chain
    assert "init_none_hash" in chain and "resolve_kv_cache_block_sizes" in chain


def test_hash_chain_prefills_meta_embeds_cache() -> None:
    """2026-10-20 meta tensor 事故:云侧 prompt_embeds 是 meta 张量(无存储),
    本地哈希回退读 .cpu() 必炸;必须在回退前预填 _prompt_embeds_per_block_hashes
    让惰性路径命中缓存、跳过读数据。哈希值带 request_id 盐,防假命中。"""
    chain = (
        Path(__file__).resolve().parents[3]
        / "vllm" / "v1" / "lwd_control" / "lwd_hash_chain.py"
    ).read_text(encoding="utf-8")
    assert "_prefill_meta_embeds_cache" in chain
    assert "_prompt_embeds_per_block_hashes" in chain
    assert "request_id" in chain  # 盐:不同请求不共享 embeds 哈希
    # 预填必须在 fallback 之前
    assert chain.index("_prefill_meta_embeds_cache(request)") < chain.index(
        "return fallback(request)"
    )
