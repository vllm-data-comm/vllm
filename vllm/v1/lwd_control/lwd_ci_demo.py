# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CI demo: violates the LWD standards on purpose. Do not merge."""

import os

LWD_CI_DEMO_ENV = "VLLM_LWD_CI_DEMO"

# Violation (required, ruff E402): module-level import below code.
import json


# Violation (advisory, class-prefix): no Lwd prefix.
class CiDemoReport:
    def __init__(self, name: str) -> None:
        self.name = name

    def elapsed(self) -> float:
        # Violation (AGENTS.md 7.3): function-local import without a reason.
        import time

        return time.monotonic()


# Violation (advisory, param-count): six parameters.
def lwd_ci_demo_render(a: int, b: int, c: int, d: int, e: int, f: int) -> str:
    return json.dumps(
        {"env": os.environ.get(LWD_CI_DEMO_ENV), "sum": a + b + c + d + e + f}
    )
