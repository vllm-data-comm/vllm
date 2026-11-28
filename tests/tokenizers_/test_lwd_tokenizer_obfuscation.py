# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate tests for the tokenizer-obfuscation startup bridge.

Regression scope: capability flags (reasoning/tool parsers, structured
output options) must stay on the native path so hosts without the optional
mixed_tokenizer package can start a native service with --reasoning-parser.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.tokenizers.lwd_tokenizer_obfuscation import (
    needs_obfuscation_hook,
    run_obfuscation_hook,
)

BRIDGE_IMPORT = "vllm.tokenizers.lwd_tokenizer_obfuscation.import_module"


@pytest.mark.parametrize(
    "field,value",
    [
        ("reasoning_parser", "deepseek_r1"),
        ("reasoning_parser_plugin", "/plugin/reasoning.py"),
        ("enable_auto_tool_choice", True),
        ("tool_call_parser", "hermes"),
        ("tool_parser_plugin", "/plugin/tools.py"),
        ("tool_strict_level", "strict"),
        ("structured_outputs_config", {"backend": "xgrammar"}),
        ("structured_outputs_config", {"reasoning_parser": "deepseek_r1"}),
        ("structured_outputs_config", {"reasoning_parser_plugin": "/p.py"}),
        ("structured_outputs_config", {"enable_in_reasoning": True}),
        ("structured_outputs_config", {"disable_any_whitespace": True}),
        ("structured_outputs_config", {"disable_additional_properties": True}),
        ("reasoning_config", {"reasoning_parser": "deepseek_r1"}),
        ("reasoning_config", {"reasoning_start_str": "<think>"}),
        ("reasoning_config", {"reasoning_end_str": "</think>"}),
    ],
)
def test_capability_flags_stay_native(field: str, value) -> None:
    args = SimpleNamespace(**{field: value})
    assert needs_obfuscation_hook(args) is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("enable_tokenizer_obfuscation", True),
        ("tokenizer_obfuscation_artifacts", "/artifacts/startup.json"),
        ("tokenizer_mode", "mixed_bytes"),
    ],
)
def test_explicit_markers_load_bridge(field: str, value) -> None:
    args = SimpleNamespace(**{field: value})
    assert needs_obfuscation_hook(args) is True


def test_marker_wins_over_capability_flags() -> None:
    args = SimpleNamespace(
        reasoning_parser="deepseek_r1",
        tokenizer_obfuscation_artifacts="/artifacts/startup.json",
    )
    assert needs_obfuscation_hook(args) is True


def test_preparing_marker_suppresses_bridge() -> None:
    args = SimpleNamespace(
        _tokenizer_obfuscation_preparing=True,
        enable_tokenizer_obfuscation=True,
        tokenizer_obfuscation_artifacts="/artifacts/startup.json",
        tokenizer_mode="mixed_bytes",
    )
    assert needs_obfuscation_hook(args) is False


def test_capability_flags_do_not_import_bridge() -> None:
    args = SimpleNamespace(reasoning_parser="deepseek_r1")
    with patch(BRIDGE_IMPORT) as import_module:
        run_obfuscation_hook("configure_server_args", args)
    import_module.assert_not_called()


def test_marker_routes_into_bridge_package() -> None:
    args = SimpleNamespace(tokenizer_mode="mixed_bytes")
    with patch(BRIDGE_IMPORT) as import_module:
        run_obfuscation_hook("configure_server_args", args)
    import_module.assert_called_once_with("mixed_tokenizer.scheme31_startup")
