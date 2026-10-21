# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source guard for the LwdC2eMeta seam on ModelRunnerOutput (issue #18
breakpoint 6): the cloud scheduler does ``from vllm.v1.outputs import
LwdC2eMeta`` at runtime, so the symbol must exist in the source.

``vllm/v1/outputs.py`` imports torch, hence the AST source-scan form —
it runs on any host and still fails the moment the definition is dropped.
"""

from __future__ import annotations

import ast
from pathlib import Path

OUTPUTS = Path(__file__).resolve().parents[3] / "vllm" / "v1" / "outputs.py"

C2E_FIELDS = {
    "hidden_num_elements",
    "top_id_ths",
    "num_accepted_tokens",
    "req_ids",
    "down_seqno",
}


def _classes() -> dict[str, ast.ClassDef]:
    tree = ast.parse(OUTPUTS.read_text(encoding="utf-8"))
    return {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }


def _fields(cls: ast.ClassDef) -> set[str]:
    return {
        stmt.target.id
        for stmt in cls.body
        if isinstance(stmt, ast.AnnAssign)
        and isinstance(stmt.target, ast.Name)
    }


def test_lwd_c2e_meta_defined_with_wire_fields() -> None:
    classes = _classes()
    assert "LwdC2eMeta" in classes, (
        "LwdC2eMeta missing from vllm/v1/outputs.py; the LWD cloud scheduler "
        "imports it at runtime (issue #18 breakpoint 6)"
    )
    assert _fields(classes["LwdC2eMeta"]) == C2E_FIELDS


def test_model_runner_output_carries_lwd_c2e_meta() -> None:
    fields = _fields(_classes()["ModelRunnerOutput"])
    assert "lwd_c2e_meta" in fields, (
        "ModelRunnerOutput.lwd_c2e_meta missing; the cloud scheduler attaches "
        "the step metadata here for the control-plane forward to the edge"
    )
