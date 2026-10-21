# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source guard: the LWD edge engine disables prefix caching before the
native engine builds the KV cache coordinator.

The prefill-only edge runs no transformer layers, so it holds zero cacheable
KV groups; with prefix caching left on (the default), the coordinator
(upstream hybrid and the Ascend patch alike) asserts "requires at least one
cacheable KV cache group" during scheduler construction, as seen on the
2026-10-19 bring-up. The disable lives in LwdEdgeEngineCore.__init__ (our
package, before super().__init__) so the upstream config file stays
untouched.

AST source-scan form: lwd_edge_engine.py imports the engine chain, so the
guard runs on any host and fails the moment the disable is dropped.
"""

from __future__ import annotations

import ast
from pathlib import Path

ENGINE = (
    Path(__file__).resolve().parents[3]
    / "vllm"
    / "v1"
    / "lwd_control"
    / "control_edge_scheduler"
    / "lwd_edge_engine.py"
)
CONFIG = Path(__file__).resolve().parents[3] / "vllm" / "config" / "vllm.py"


def _disables_prefix_caching(init: ast.FunctionDef) -> bool:
    for node in ast.walk(init):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Attribute) and t.attr == "enable_prefix_caching"
                for t in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and node.value.value is False
        ):
            return True
    return False


def _edge_engine_init() -> ast.FunctionDef | None:
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "LwdEdgeEngineCore":
            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
                    return stmt
    return None


def test_edge_engine_init_disables_prefix_caching() -> None:
    init = _edge_engine_init()
    assert init is not None, "LwdEdgeEngineCore.__init__ not found"
    assert _disables_prefix_caching(init), (
        "LwdEdgeEngineCore.__init__ must disable prefix caching before "
        "super().__init__: the edge holds zero cacheable KV groups and the "
        "coordinator asserts 'requires at least one cacheable group'"
    )


def test_upstream_config_file_stays_untouched() -> None:
    tree = ast.parse(CONFIG.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = ast.unparse(node.test)
            assert not (
                "lwd_config" in test and "edge" in test
            ), (
                "vllm/config/vllm.py carries an LWD edge branch; keep LWD "
                "role logic inside the lwd_control package (non-invasive)"
            )
