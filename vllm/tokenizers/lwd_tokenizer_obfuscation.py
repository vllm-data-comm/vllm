# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LWD scheme-3.1 tokenizer obfuscation: vLLM-side startup bridge.

Optional bridge for the independently packaged `mixed_tokenizer` scheme-3.1
obfuscation feature ported onto the LWD main line. Native configurations,
including reasoning, tool and structured-output options, never import the
optional package; deployment policy and artifact loading live in that
package. This module only identifies deployments that select obfuscation.

The user-facing options travel in a `tokenizer_obfuscation` block of the
native `--additional-config` argument (inline JSON or a YAML file via
`--config`); `vllm.config.lwd.tokenizer_obfuscation` parses and lifts the
block onto the EngineArgs fields before these hooks run.
"""

import os
from importlib import import_module
from typing import Any


def needs_obfuscation_hook(args: Any) -> bool:
    """Recognize explicit obfuscation deployment markers only.

    Capability flags such as reasoning or tool parsers and an explicit
    structured-output backend are valid native configurations; they do not
    force the optional bridge. The `mixed_tokenizer` package is imported
    only when the deployment itself selects the obfuscated tokenizer.
    """
    if getattr(args, "_tokenizer_obfuscation_preparing", False):
        # The trusted builder must first resolve the unmodified native config.
        # This private in-process marker is intentionally absent from the CLI.
        return False
    return bool(
        getattr(args, "enable_tokenizer_obfuscation", False)
        or getattr(args, "tokenizer_obfuscation_artifacts", None)
        or getattr(args, "tokenizer_mode", None) == "mixed_bytes"
    )


def run_obfuscation_hook(name: str, args: Any, *extra: Any) -> None:
    """Delegate configuration without adding a dependency to native startup."""
    if needs_obfuscation_hook(args):
        module = import_module("mixed_tokenizer.scheme31_startup")
        getattr(module, name)(args, *extra)


def validate_structured_outputs_with_route(
    params: Any,
    model_config: Any,
    structured_config: Any,
    tokenizer: Any,
    validate: Any,
) -> None:
    """Keep original-tokenizer validation native; bind mixed request routing."""
    if getattr(tokenizer, "mixed", None) is not None and os.environ.get(
        "MIXED_TOKENIZER_STARTUP_MANIFEST"
    ):
        module = import_module("mixed_tokenizer.scheme31_backend_route")
        return module.validate_native_request(
            params, model_config, structured_config, tokenizer, validate
        )
    return validate(model_config, structured_config, tokenizer)


def prepare_structured_output_backend_route(
    params: Any, structured_config: Any, tokenizer: Any, backend: Any
) -> str | None:
    if getattr(tokenizer, "mixed", None) is not None and os.environ.get(
        "MIXED_TOKENIZER_STARTUP_MANIFEST"
    ):
        module = import_module("mixed_tokenizer.scheme31_backend_route")
        return module.prepare_engine_backend(
            params, structured_config, tokenizer, backend
        )
    return None


def confirm_structured_output_backend_route(selected: str | None, backend: Any) -> None:
    if selected is not None:
        module = import_module("mixed_tokenizer.scheme31_backend_route")
        module.confirm_engine_backend(selected, backend)
