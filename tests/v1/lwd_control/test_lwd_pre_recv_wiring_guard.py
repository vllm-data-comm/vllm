# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source guard: the pre-recv sideband MQ wiring stays in place.

Issue #18 breakpoint 5 (2026-10-19 hang): without the wiring, the receive
watermark never advanced, the pre-recv gate never opened and the cloud
spun empty steps forever while the edge waited for the DOWN tensor.

Final non-invasive split being pinned here - the vllm executor carries
ZERO LWD knowledge (restored to pristine upstream):
- the engine creates the hint MQ in __init__ BEFORE super() spawns the
  workers (the handle rides the environment to them);
- the done-MQ handle travels worker -> engine through the wire TCP store
  (publish_done_mq_handle during init_device, acquire_done_mq after the
  workers are ready) - the same rendezvous the deployment already
  depends on.

Equivalence history, kept per the refactor rule that unstated
equivalence is untrusted:
- vs the executor-owned version (babc9a14c7): hint MQ ordering is
  unchanged (before worker spawn); the done-MQ handle bytes are the same
  export_handle() payload, only the transport changed (READY pipe ->
  wire store); the drain semantics are untouched;
- vs the three-hunk version (3da8180139): the done handle no longer
  rides WorkerProcHandle/READY at all, so those hunks were reverted and
  the executor file is byte-identical to upstream;
- the cloud-role predicate lives in the engine-class selection
  (run_engine_core splits on role; the topology schema admits only
  edge/cloud), pinned by test_cloud_engine_class_is_cloud_role_only.

AST source-scan form, no torch required.
"""

from __future__ import annotations

import ast
from pathlib import Path

VLLM_ROOT = Path(__file__).resolve().parents[3]
EXECUTOR = VLLM_ROOT / "vllm" / "v1" / "executor" / "multiproc_executor.py"
ENGINE = (
    VLLM_ROOT
    / "vllm"
    / "v1"
    / "lwd_control"
    / "control_cloud_scheduler"
    / "lwd_cloud_engine.py"
)
CORE = VLLM_ROOT / "vllm" / "v1" / "engine" / "core.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_executor_is_lwd_free() -> None:
    src = _source(EXECUTOR).lower()
    for marker in ("irecv", "lwd", "hint_mq", "done_mq"):
        assert marker not in src, (
            f"the upstream executor mentions {marker!r}: pre-recv plumbing "
            f"must stay inside the lwd_control package and vllm_ascend"
        )


def test_engine_owns_hint_and_done_mqs() -> None:
    src = _source(ENGINE)
    assert "self._lwd_hint_mq = create_hint_mq()" in src, (
        "the engine must create the hint MQ in __init__"
    )
    super_pos = src.index("super().__init__(")
    hint_pos = src.index("self._lwd_hint_mq = create_hint_mq()")
    assert hint_pos < super_pos, (
        "the hint MQ must be created BEFORE super().__init__: the handle "
        "rides the environment and only workers spawned afterwards see it"
    )
    assert "acquire_done_mq(" in src and "get_lwd_init_method" in src, (
        "the engine must acquire the done MQ from the wire TCP store"
    )
    assert 'getattr(self, "_lwd_done_mq", None)' in src, (
        "the drain must read the engine-owned done MQ"
    )


def test_cloud_engine_class_is_cloud_role_only() -> None:
    """Pin the selection site the unconditional create_hint_mq relies on:
    LwdCloudEngineCore is instantiated only for role=="cloud" (and the
    topology schema admits only edge/cloud), which is what makes the
    engine-owned hint MQ equivalent to the executor's old explicit
    role check."""
    src = _source(CORE)
    assert 'lwd.role == "edge"' in src, (
        "run_engine_core lost the role split that guarantees the cloud "
        "engine class is cloud-only"
    )
    assert "LwdCloudEngineCore" in src, (
        "run_engine_core no longer selects LwdCloudEngineCore"
    )
