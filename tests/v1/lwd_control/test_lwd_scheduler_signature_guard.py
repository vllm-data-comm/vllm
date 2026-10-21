# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Signature guard: the LWD scheduler schedule entry stays call-compatible
with the native Scheduler.schedule.

Upstream step_with_batch_queue passes throttle_prefills positionally; the
2026-10-19 bring-up died with TypeError on both sides because the migrated
schedulers predated that parameter. The signature now lives once in
LwdBaseScheduler.schedule (template method over _lwd_schedule_step). This
guard diffs that signature against the native one and checks both role
hooks exist, so the next upstream signature change fails here instead of
at first request. AST source-scan form, no torch required.
"""

from __future__ import annotations

import ast
from pathlib import Path

VLLM_ROOT = Path(__file__).resolve().parents[3]
NATIVE = VLLM_ROOT / "vllm" / "v1" / "core" / "sched" / "scheduler.py"
BASE = (
    VLLM_ROOT
    / "vllm"
    / "v1"
    / "lwd_control"
    / "control_scheduler"
    / "lwd_base_scheduler.py"
)
LWD_SCHEDULERS = {
    "LwdEdgeScheduler": VLLM_ROOT
    / "vllm"
    / "v1"
    / "lwd_control"
    / "control_edge_scheduler"
    / "lwd_edge_scheduler.py",
    "LwdCloudScheduler": VLLM_ROOT
    / "vllm"
    / "v1"
    / "lwd_control"
    / "control_cloud_scheduler"
    / "lwd_cloud_scheduler.py",
}


def _method(path: Path, class_name: str, method: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef) and stmt.name == method:
                    return stmt
    raise AssertionError(f"{class_name}.{method} not found in {path}")


def _positional_params(func: ast.FunctionDef) -> list[tuple[str, bool]]:
    """(name, has_default) for call-relevant positional params, self excluded."""
    args = func.args
    names = [a.arg for a in [*args.posonlyargs, *args.args]][1:]
    n_defaults = len(args.defaults)
    return [(name, idx >= len(names) - n_defaults) for idx, name in enumerate(names)]


def test_base_schedule_matches_native_signature() -> None:
    native = _positional_params(_method(NATIVE, "Scheduler", "schedule"))
    assert native, "native Scheduler.schedule params disappeared?"
    ours = _positional_params(_method(BASE, "LwdBaseScheduler", "schedule"))
    assert ours == native, (
        f"LwdBaseScheduler.schedule{ours} is not call-compatible with the "
        f"native Scheduler.schedule{native}: upstream passes these "
        f"positionally from step_with_batch_queue and the engine dies with "
        f"TypeError at first request"
    )


def test_role_hooks_exist() -> None:
    for class_name, path in LWD_SCHEDULERS.items():
        hook = _method(path, class_name, "_lwd_schedule_step")
        assert not _positional_params(hook), (
            f"{class_name}._lwd_schedule_step takes no parameters; the "
            f"throttle_prefills handling belongs to the base entry"
        )
