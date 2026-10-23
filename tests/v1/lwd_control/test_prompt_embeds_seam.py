# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Zero-weave guard for the prompt-embeds / MM seams on the vllm base runner.

The LWD cloud no longer threads hooks through GPUModelRunner: the
placeholder rebuild, the MM-encoder gate and the mrope
prompt-embeds-only branch all live in the vllm-ascend subclass
(``LwdCloudModelRunner`` wraps ``super()`` and presents a dynamic
``supports_mm_inputs``), so the base file stays pristine. This test locks
the zero state and the native lines the seams used to replace.

The placeholder now travels as a meta tensor on ``prompt_embeds`` itself
(shape only, no storage): ``NewRequestData`` carries no transport fields
and the engine sets no marker attribute.

Companion behavior tests: vllm-ascend
``tests/ut/worker/test_lwd_prompt_embeds_seam.py``.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
GMR = _REPO_ROOT / "vllm" / "v1" / "worker" / "gpu_model_runner.py"
SCHED_OUTPUT = _REPO_ROOT / "vllm" / "v1" / "core" / "sched" / "output.py"
CLOUD_ENGINE = (
    _REPO_ROOT / "vllm" / "v1" / "lwd_control" / "control_cloud_scheduler"
    / "lwd_cloud_engine.py"
)


def test_gpu_model_runner_zero_weave() -> None:
    src = GMR.read_text(encoding="utf-8")
    for symbol in (
        "_resolve_prompt_embeds",
        "_should_run_mm_encoder",
        "prompt-embeds only",
    ):
        assert symbol not in src, f"base weave leaked back: {symbol}"


def test_native_lines_restored() -> None:
    """The three former seam sites must be back to native passthrough form,
    preventing a silent revert of the zero-weave state."""
    src = GMR.read_text(encoding="utf-8")
    assert "prompt_embeds=new_req_data.prompt_embeds" in src
    assert "req_state.prompt_embeds = new_req_data.prompt_embeds" in src
    # mrope: native assertion restored (no prompt-embeds-only branch)
    assert '"M-RoPE requires either prompt_token_ids or prompt_embeds."' in src
    # MM gate: native inline condition restored (not a hook call)
    assert "if self.supports_mm_inputs and is_first_rank" in src


def test_placeholder_transport_protocol_removed() -> None:
    """The shape/dtype transport protocol is gone from the shared
    structures: meta tensors carry the placeholder on ``prompt_embeds``
    itself, so neither NewRequestData fields nor a request marker
    attribute may come back."""
    src = SCHED_OUTPUT.read_text(encoding="utf-8")
    for symbol in (
        "prompt_embeds_shape:",
        "prompt_embeds_dtype:",
        "lwd_embeds_placeholder",
    ):
        assert symbol not in src, f"transport protocol leaked back: {symbol}"
    assert "prompt_embeds=request.prompt_embeds" in src


def test_engine_attaches_meta_placeholder() -> None:
    """The single request-build point attaches a meta placeholder (shape
    only, empty storage): the 35MB-scale zero buffer never materializes
    in the engine process, so it cannot ride the MQ overflow slow path."""
    src = CLOUD_ENGINE.read_text(encoding="utf-8")
    assert 'device="meta"' in src
    assert "lwd_embeds_placeholder" not in src
