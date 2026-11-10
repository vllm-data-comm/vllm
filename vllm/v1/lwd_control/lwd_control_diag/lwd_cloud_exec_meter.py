# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Where a cloud step's wall clock goes, when one takes seconds.

`lwd_ttft_meter`'s step figure is the interval between two commits, so a
step that took eight seconds and a step that waited eight seconds for
work look the same there. Four of its six TTFT segments close on the step
thread as well, so one stalled step widens `flight`, `fill`, `queue` and
`exec` together -- four readings of one event, and none of them says what
the thread was doing.

This meter times the step itself, split at the boundaries the native
engine already has, so each segment names one suspect:

  - pre_gap: the previous step's end to this one's start, less the two
    segments below. The input queue, the request-count publish, and
    waiting for work -- not idle, because the input handlers run here too.
  - dummy: the dummy batch a DP rank runs when it has nothing of its own,
    to keep the MoE ranks in lockstep. Work done for someone else.
  - dp_sync: the all-reduce that decides whether the wave continues. A
    blocking collective, so it costs this rank whatever the slowest rank
    is still doing -- the one segment here that measures another process.
  - sched: the scheduler choosing a batch, LWD admission included.
  - submit: handing the batch to the executor (argument marshalling and
    the worker MQ enqueue), not the execution.
  - grammar: the structured-output bitmask.
  - exec_rpc: waiting for the worker's execute_model. Input preparation,
    the forward pass, collectives and the response trip home, which only
    the worker-side meter can tell apart.
  - sample_rpc: sampling, bookkeeping, the DOWN collect and submit, and
    the output thread's work.
  - abort: aborts that arrived during execution.
  - update: committing tokens and building the c2e notifies.
  - c2e_send / native_post: the post-step hook, split at the send loop so
    a full outbound queue to the edge is visible as the wait it is.

`exec_rpc` is named for what it measures: the round trip, not the model.
A long one does not say the NPU was busy.

The timings ride on the future rather than on "the last batch
scheduled", because a batch queue deeper than one returns a batch that
was scheduled several steps ago; reading the latest shape would pair
every duration with the wrong batch. The shape of the batch that
produced a slow step is on its own line, because an eight-second step
over 8k prompt rows and one over 30 decode rows are different faults.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from vllm.logger import init_logger
from vllm.v1.lwd_control.lwd_control_diag.lwd_meter_base import (
    LwdIntervalMeter,
    LwdSpan,
)

logger = init_logger(__name__)

# A step worth naming individually. The steady state here is 120-900 ms
# depending on how much prefill the step carried, so a second is clear of
# every healthy shape and still catches anything that stands out.
LWD_SLOW_STEP_S = 1.0


@dataclass(slots=True)
class LwdBatchShape:
    """What one step was asked to run. Snapshotted by the scheduler while
    it still knows, and carried to wherever that batch's timings land."""

    decode_reqs: int = 0
    decode_tokens: int = 0
    prefill_reqs: int = 0
    prefill_windows: int = 0
    prefill_tokens: int = 0
    room: int = 0
    reserved: int = 0
    queue_depth: int = 0
    blocked_partial: int = 0
    blocked_no_arrival: int = 0
    running: int = 0
    waiting: int = 0

    def __str__(self) -> str:
        return (
            f"prefill={self.prefill_tokens}/{self.prefill_reqs}"
            f"of{self.prefill_windows} decode={self.decode_tokens}/"
            f"{self.decode_reqs} room={self.room} reserved={self.reserved} "
            f"queued={self.queue_depth} blocked={self.blocked_partial}/"
            f"{self.blocked_no_arrival} running={self.running} "
            f"waiting={self.waiting}"
        )


@dataclass(slots=True)
class LwdStepTiming:
    """One step's segments, in the order they run. Seconds, monotonic."""

    pre_gap: float = 0.0
    dummy: float = 0.0
    dp_sync: float = 0.0
    sched: float = 0.0
    submit: float = 0.0
    grammar: float = 0.0
    exec_rpc: float = 0.0
    sample_rpc: float = 0.0
    abort: float = 0.0
    update: float = 0.0
    c2e_send: float = 0.0
    native_post: float = 0.0
    c2e_retries: int = 0
    down_seqno: int = -1
    shape: LwdBatchShape = field(default_factory=LwdBatchShape)

    def total(self) -> float:
        """Everything the step thread spent, waiting for work included."""
        return (
            self.pre_gap
            + self.dummy
            + self.dp_sync
            + self.sched
            + self.submit
            + self.grammar
            + self.exec_rpc
            + self.sample_rpc
            + self.abort
            + self.update
            + self.c2e_send
            + self.native_post
        )

    def in_step(self) -> float:
        """Everything but the wait that preceded it. The dummy batch and
        the DP all-reduce count: the thread was running them, and a rank
        that spends its step keeping others in lockstep is the finding,
        not the background."""
        return self.total() - self.pre_gap


class LwdCloudExecMeter(LwdIntervalMeter):
    """Per-step breakdown of the cloud engine's loop.

    Reports two kinds of line. The interval line is every step of the
    window averaged, which says whether a segment is slow in general.
    The slow line is one step named on its own with the batch it ran,
    which says whether a long step was doing something or waiting for
    something -- the averages cannot, because one eight-second step in a
    window of eighty hides inside them.
    """

    def __init__(self) -> None:
        self._last_step_end = 0.0
        super().__init__()

    def _reset(self) -> None:
        # `_last_step_end` is not cleared: the gap across a report
        # boundary belongs to one of the two windows, and dropping it
        # would drop the long ones first.
        self._steps = 0
        self._slow = 0
        self._retries = 0
        self._total = LwdSpan()
        self._pre_gap = LwdSpan()
        self._dummy = LwdSpan()
        self._dp_sync = LwdSpan()
        self._sched = LwdSpan()
        self._submit = LwdSpan()
        self._grammar = LwdSpan()
        self._exec_rpc = LwdSpan()
        self._sample_rpc = LwdSpan()
        self._abort = LwdSpan()
        self._update = LwdSpan()
        self._c2e_send = LwdSpan()
        self._native_post = LwdSpan()

    def note_path(self, batch_queue_size: int, async_scheduling: bool) -> None:
        """Which native step path this deployment runs, said once at
        startup. `step_with_batch_queue` returns a batch scheduled some
        steps earlier, and that changes how every number here reads; the
        choice is derived from config, so printing it beats assuming."""
        logger.info(
            "[lwd][cloud-exec] step path: batch_queue_size=%d "
            "async_scheduling=%s step_fn=%s",
            batch_queue_size,
            async_scheduling,
            "step_with_batch_queue" if batch_queue_size > 1 else "step",
        )

    def note_step(self, timing: LwdStepTiming) -> None:
        """One trip through the engine's step."""
        self._steps += 1
        self._retries += timing.c2e_retries
        self._total.add(timing.in_step())
        self._pre_gap.add(timing.pre_gap)
        self._dummy.add(timing.dummy)
        self._dp_sync.add(timing.dp_sync)
        self._sched.add(timing.sched)
        self._submit.add(timing.submit)
        self._grammar.add(timing.grammar)
        self._exec_rpc.add(timing.exec_rpc)
        self._sample_rpc.add(timing.sample_rpc)
        self._abort.add(timing.abort)
        self._update.add(timing.update)
        self._c2e_send.add(timing.c2e_send)
        self._native_post.add(timing.native_post)
        if timing.in_step() >= LWD_SLOW_STEP_S:
            self._slow += 1
            self._report_slow(timing)
        self._roll()

    def note_step_end(self) -> float:
        """Close the step and open the gap before the next one. Returns
        that gap once the next step starts."""
        self._last_step_end = time.monotonic()
        return self._last_step_end

    def take_pre_gap(self) -> float:
        """How long since the last step ended. Zero on the first step."""
        if not self._last_step_end:
            return 0.0
        return time.monotonic() - self._last_step_end

    def _report_slow(self, timing: LwdStepTiming) -> None:
        logger.info(
            "[lwd][cloud-slow-step] %.0fms: pre_gap=%.0f dummy=%.0f "
            "dp_sync=%.0f sched=%.0f "
            "submit=%.0f grammar=%.0f exec_rpc=%.0f sample_rpc=%.0f "
            "abort=%.0f update=%.0f c2e_send=%.0f native_post=%.0f "
            "(ms); retries=%d down=%d; %s; ts=%.3f",
            timing.in_step() * 1000,
            timing.pre_gap * 1000,
            timing.dummy * 1000,
            timing.dp_sync * 1000,
            timing.sched * 1000,
            timing.submit * 1000,
            timing.grammar * 1000,
            timing.exec_rpc * 1000,
            timing.sample_rpc * 1000,
            timing.abort * 1000,
            timing.update * 1000,
            timing.c2e_send * 1000,
            timing.native_post * 1000,
            timing.c2e_retries,
            timing.down_seqno,
            timing.shape,
            # Wall clock, for lining this line up against the edge's log
            # and the other DP rank's. Every duration above is monotonic.
            time.time(),
        )

    def _report(self) -> None:
        if not self._steps:
            return
        logger.info(
            "[lwd][cloud-exec] %d steps (%d slow): step=%s pre_gap=%s; "
            "dummy=%s dp_sync=%s sched=%s submit=%s grammar=%s "
            "exec_rpc=%s sample_rpc=%s "
            "abort=%s update=%s c2e_send=%s native_post=%s (avg/max ms); "
            "c2e retries=%d",
            self._steps,
            self._slow,
            self._total,
            self._pre_gap,
            self._dummy,
            self._dp_sync,
            self._sched,
            self._submit,
            self._grammar,
            self._exec_rpc,
            self._sample_rpc,
            self._abort,
            self._update,
            self._c2e_send,
            self._native_post,
            self._retries,
        )


class LwdCloudOutputMeter(LwdIntervalMeter):
    """What the executor's output thread spends between a step finishing
    on the device and its result reaching the engine.

    This thread is off the step thread's critical path by design, so its
    cost appears nowhere in the step breakdown -- yet a step cannot be
    collected until it has run, so a slow output thread shows up as
    `exec_rpc` on the *next* step and reads as a slow forward pass.

      - meta_harvest: waiting for the per-step ready event and copying
        the DOWN metadata off the device.
      - output_get: resolving an async runner output.
      - enqueue: handing the result to the engine's response queue.

    Called from the executor's output thread, which is the only thread
    that touches this meter.
    """

    def _reset(self) -> None:
        self._outputs = 0
        self._rows = 0
        self._harvest = LwdSpan()
        self._output_get = LwdSpan()
        self._enqueue = LwdSpan()
        self._total = LwdSpan()

    def note_output(
        self,
        harvest_s: float,
        output_get_s: float,
        enqueue_s: float,
        hidden_numel: int,
    ) -> None:
        """One output crossing back to the engine."""
        self._outputs += 1
        self._rows += hidden_numel
        self._harvest.add(harvest_s)
        self._output_get.add(output_get_s)
        self._enqueue.add(enqueue_s)
        self._total.add(harvest_s + output_get_s + enqueue_s)
        self._roll()

    def _report(self) -> None:
        if not self._outputs:
            return
        logger.info(
            "[lwd][cloud-output] %d outputs: total=%s meta_harvest=%s "
            "output_get=%s enqueue=%s (avg/max ms); hidden elems/output=%d",
            self._outputs,
            self._total,
            self._harvest,
            self._output_get,
            self._enqueue,
            self._rows // self._outputs if self._outputs else 0,
        )
