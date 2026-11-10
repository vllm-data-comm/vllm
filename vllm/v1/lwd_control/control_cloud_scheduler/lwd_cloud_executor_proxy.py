# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cloud executor boundary: time the round trip to the workers.

The engine's step calls the executor twice -- once to run the model,
once to sample -- and both are non-blocking: they return a future, and
the step blocks on `result()` somewhere later. Timing them from the step
would therefore charge the submit and the wait to one number, and with a
batch queue deeper than one the wait does not even belong to the batch
the step just scheduled.

So the wait is timed on the future. `execute_model` charges the submit
to the step that made the call and wraps the future; whoever resolves it
charges the wait to the step that was running at that moment, which is
the step whose wall clock it occupied. The future carries the shape of
the batch it belongs to, so a step that waits on a batch scheduled three
steps ago is read against that batch rather than against whatever was
scheduled most recently.

A future that is never resolved is never timed, and that is the point of
the segments starting at None. On the batch-queue path the engine pops
and waits on the sample future alone; the execute future is held only to
re-raise its error if sampling returned None. So `exec_rpc` stays
unmeasured there and prints as a dash -- which is the honest reading,
because the execute call did take time, it was simply never awaited on
its own. The one wait covers the whole worker trip and is reported as
`batch_wait` rather than `sample_rpc`.

Everything else passes straight through: this proxy adds no behaviour,
only stamps.
"""

from __future__ import annotations

import time
from typing import Any

from vllm.v1.lwd_control.lwd_control_diag.lwd_cloud_exec_meter import (
    LwdBatchShape,
    LwdStepTiming,
)


class LwdTimedFuture:
    """A future that times its own `result()` into the open step record.

    The record is taken from the proxy at resolution rather than captured
    at creation: a record is reported once its step ends, so a future
    outliving its step would otherwise add to a line already printed.

    One step may hold two futures -- execute then sample -- so the
    segment to charge is chosen per future.
    """

    __slots__ = ("_fut", "_proxy", "_field", "_shape")

    def __init__(
        self,
        fut: Any,
        proxy: LwdCloudExecutorProxy,
        field: str,
        shape: LwdBatchShape,
    ) -> None:
        self._fut = fut
        self._proxy = proxy
        self._field = field
        self._shape = shape

    def result(self, timeout: float | None = None) -> Any:
        started = time.monotonic()
        try:
            return self._fut.result(timeout)
        finally:
            timing = self._proxy.lwd_timing
            if timing is not None:
                timing.add_wait(self._field, time.monotonic() - started)
                # The batch this step waited on, which with a deeper
                # queue is not the one it scheduled.
                timing.shape = self._shape

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fut, name)


class LwdCloudExecutorProxy:
    """Executor boundary proxy, wrapped around the cloud engine's
    `model_executor` at assembly."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._timing: LwdStepTiming | None = None

    @property
    def lwd_timing(self) -> LwdStepTiming | None:
        """The open step record, or None before the first step."""
        return self._timing

    def lwd_open_step(self, timing: LwdStepTiming) -> None:
        """The step about to run. Calls made from here on charge their
        submit to this record, and their waits to whichever record is
        open when they happen."""
        self._timing = timing

    def execute_model(self, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        timing = self._timing
        if timing is None:
            return self._inner.execute_model(scheduler_output, *args, **kwargs)
        started = time.monotonic()
        fut = self._inner.execute_model(scheduler_output, *args, **kwargs)
        timing.submit += time.monotonic() - started
        # A blocking executor returns the output itself; only a future
        # has a wait worth timing.
        if not hasattr(fut, "result"):
            return fut
        return LwdTimedFuture(fut, self, "exec_rpc", timing.shape)

    def sample_tokens(self, grammar_output: Any, *args: Any, **kwargs: Any) -> Any:
        timing = self._timing
        if timing is None:
            return self._inner.sample_tokens(grammar_output, *args, **kwargs)
        started = time.monotonic()
        out = self._inner.sample_tokens(grammar_output, *args, **kwargs)
        elapsed = time.monotonic() - started
        if not hasattr(out, "result"):
            # Blocking path: the call itself was the wait.
            timing.add_wait("sample_rpc", elapsed)
            return out
        timing.submit += elapsed
        return LwdTimedFuture(out, self, "sample_rpc", timing.shape)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
