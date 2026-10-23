# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""AST weave guard: the shared vllm worker base classes must stay LWD-free.

Scans ``vllm/v1/worker/gpu_model_runner.py``, ``worker_base.py`` and
``gpu_input_batch.py`` for LWD symbols (identifiers, attributes, import
aliases, getattr strings). Every allowlist entry is a known weave-in
scheduled for removal by the base-class sinking batches; the test fails on

- any NEW lwd symbol appearing in the base files, and
- any STALE allowlist entry whose symbol is already gone,

so the allowlist only ever shrinks. Pure AST scan, no torch needed.

Permanent exception: ``worker_base.initialize_from_config`` indexes
``kv_cache_configs`` by rpc_rank for multi-engine (LWD edge/cloud)
deployments; this stays until the upstream PR lands.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]

BASE_FILES = [
    "vllm/v1/worker/gpu_model_runner.py",
    "vllm/v1/worker/worker_base.py",
    "vllm/v1/worker/gpu_input_batch.py",
]

# All three base files are now fully LWD-free: the worker_base rpc_rank
# branch went away with the no-PP topology (protocol/topology/config live
# in the LwdConfig-generation PR; see the sibling no-PP data-plane PR).
# gpu_model_runner.py is now fully LWD-free. The spec recovery chain was
# dead code in pdmix (LwdCloudScheduler defers prefill, never evicts
# decode). The only permanent exception is the worker_base rpc_rank branch
# pending its upstream PR.
ALLOWED = {
    "vllm/v1/worker/gpu_model_runner.py": set(),
    "vllm/v1/worker/worker_base.py": set(),
    "vllm/v1/worker/gpu_input_batch.py": set(),
}


def _scan(path: Path) -> set[tuple[str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[tuple[str, str]] = set()

    def visit(node: ast.AST, func: str) -> None:
        for child in ast.iter_child_nodes(node):
            child_func = func
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                child_func = child.name
            if isinstance(child, ast.Name) and "lwd" in child.id.lower():
                found.add((func, child.id))
            elif isinstance(child, ast.Attribute) and "lwd" in child.attr.lower():
                found.add((func, child.attr))
            elif isinstance(child, ast.alias) and "lwd" in child.name.lower():
                found.add((func, child.name.split(".")[-1]))
            elif (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "getattr"
                and len(child.args) >= 2
                and isinstance(child.args[1], ast.Constant)
                and isinstance(child.args[1].value, str)
                and "lwd" in child.args[1].value.lower()
            ):
                found.add((func, f"getattr:{child.args[1].value}"))
            visit(child, child_func)

    visit(tree, "<module>")
    return found


@pytest.mark.parametrize("rel_path", BASE_FILES)
def test_base_files_free_of_new_lwd_weave(rel_path: str) -> None:
    found = _scan(_REPO_ROOT / rel_path)
    allowed = ALLOWED.get(rel_path, set())
    unexpected = found - allowed
    stale = allowed - found
    assert not unexpected, f"new lwd weave-ins in {rel_path}: {sorted(unexpected)}"
    assert not stale, f"stale allowlist entries in {rel_path}: {sorted(stale)}"
