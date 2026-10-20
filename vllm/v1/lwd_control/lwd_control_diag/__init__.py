"""Control-plane diagnostics: switches, spec acceptance observation, bypass
decisions and payloads, and diagnostic logs. Returns decisions and data
only; owns no executor adapters and imports no scheduler, executor or
engine classes."""

from vllm.v1.lwd_control.lwd_control_diag.lwd_control_diag import (
    LwdControlDiag,
)

__all__ = ["LwdControlDiag"]
