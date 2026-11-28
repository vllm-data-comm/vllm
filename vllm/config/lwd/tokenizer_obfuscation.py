# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`tokenizer_obfuscation` block of the native `--additional-config` argument.

Central parse point for the LWD scheme-3.1 tokenizer-obfuscation deployment
options, next to `resolve_lwd_config` for the `lwd_config` block. The block
travels as inline JSON or through a YAML file via `--config`:

    {"tokenizer_obfuscation": {"enabled": true,
                               "artifacts": "/path/to/startup.json"}}
"""

from typing import Any

OBFUSCATION_CONFIG_KEY = "tokenizer_obfuscation"
"""`--additional-config` block carrying the obfuscation deployment options."""

_OBFUSCATION_CONFIG_FIELDS = frozenset({"enabled", "artifacts"})


def apply_obfuscation_additional_config(args: Any) -> None:
    """Lift the `tokenizer_obfuscation` additional-config block onto args.

    `enabled` defaults to the presence of `artifacts`. Fields set explicitly
    by a programmatic `EngineArgs` construction win over the block.
    """
    additional = getattr(args, "additional_config", None)
    if not isinstance(additional, dict) or OBFUSCATION_CONFIG_KEY not in additional:
        return
    block = additional[OBFUSCATION_CONFIG_KEY]
    if not isinstance(block, dict):
        raise ValueError(
            f"additional-config '{OBFUSCATION_CONFIG_KEY}' must be a mapping"
        )
    unknown = set(block) - _OBFUSCATION_CONFIG_FIELDS
    if unknown:
        raise ValueError(
            f"additional-config '{OBFUSCATION_CONFIG_KEY}' has unknown keys "
            f"{sorted(unknown)}; expected {sorted(_OBFUSCATION_CONFIG_FIELDS)}"
        )
    enabled = block.get("enabled")
    artifacts = block.get("artifacts")
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError(
            "additional-config 'tokenizer_obfuscation.enabled' must be a boolean"
        )
    if artifacts is not None and not isinstance(artifacts, str):
        raise ValueError(
            "additional-config 'tokenizer_obfuscation.artifacts' must be a "
            "manifest path"
        )
    if enabled is False and artifacts:
        raise ValueError(
            "additional-config 'tokenizer_obfuscation.artifacts' cannot be set "
            "while obfuscation is disabled"
        )
    if enabled is True and not artifacts:
        raise ValueError(
            "additional-config 'tokenizer_obfuscation' requires 'artifacts' "
            "when enabled"
        )
    if getattr(args, "enable_tokenizer_obfuscation", None) is None:
        args.enable_tokenizer_obfuscation = (
            enabled if enabled is not None else artifacts is not None
        )
    if getattr(args, "tokenizer_obfuscation_artifacts", None) is None:
        args.tokenizer_obfuscation_artifacts = artifacts
