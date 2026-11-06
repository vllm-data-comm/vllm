# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stand-in edge and cloud workers for the LWD flow simulation on CPU.

They keep the contract of the vllm-ascend LWD workers that the vLLM engines
and schedulers depend on (EMBED/UNEMBED dispatch on the edge, the early-receive
sideband and the DOWN carrier on the cloud) but run no model and replace HCCL
with gloo point-to-point on the same UP/DOWN rank pair. Like HCCL P2P, gloo
pairs sends and receives in posting order, so every payload is filled with its
seqno and the receiver records which send it was paired with.

Selected with `worker_cls`; every data-plane step is recorded to the trace.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType
from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
    attach_hint_mq,
    create_report_mq,
    hint_fields,
)

from tests.v1.lwd_control.e2e.lwd_sim_model import (
    LWD_SIM_SAMPLED_TOKEN_ID,
    lwd_sim_token_rank,
)
from tests.v1.lwd_control.e2e.lwd_sim_trace import LwdSimTraceWriter
from vllm.distributed.device_communicators.shm_broadcast import Handle
from vllm.distributed.lwd import LwdChannelEndpoints, LwdParallelGroupsSpec
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.tasks import SupportedTask
from vllm.v1.core.sched.output import (
    GrammarOutput,
    LwdBatch,
    LwdBatchType,
    LwdEmbedBatch,
    LwdUnembedBatch,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_scheduler import (
    LwdDownCarrier,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

# KV blocks the cloud reports: room for every scenario prompt at once.
_LWD_SIM_NUM_KV_BLOCKS = 64
_LWD_SIM_KV_CACHE_LAYOUT = "LBNHC"
# float32 holds the seqno stamps exactly.
_LWD_SIM_PAYLOAD_DTYPE = torch.float32
_LWD_SIM_HINT_TIMEOUT_S = 0.0005
_LWD_SIM_POLL_INTERVAL_S = 0.002
_LWD_SIM_RECV_POLL_S = 0.05
_LWD_SIM_THREAD_JOIN_TIMEOUT_S = 1.0


def _lwd_sim_stamp(buffer: torch.Tensor) -> int:
    return int(buffer[0].item()) if buffer.numel() else -1


class _LwdSimWorkerBase(WorkerBase):
    """The world, model-parallel groups and UP/DOWN channels as the LWD
    workers build them, on gloo; no model."""

    trace_source: str

    def init_device(self) -> None:
        self.device = torch.device("cpu")
        # The LWD world args (global rank, topology world size) override these.
        init_distributed_environment(
            self.parallel_config.world_size,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            backend="gloo",
        )
        ensure_model_parallel_initialized(
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size,
        )
        lwd_config = self.vllm_config.lwd_config
        assert lwd_config is not None
        spec = LwdParallelGroupsSpec.from_topology(lwd_config.topology)
        endpoints = spec.channel_endpoints_of(dist.get_rank())
        assert endpoints is not None, "the simulation runs one rank per instance"
        ranks = [endpoints.edge_rank, endpoints.cloud_rank]
        # A world collective: both sides create UP first, then DOWN.
        self._up_group = dist.new_group(ranks, backend="gloo")
        self._down_group = dist.new_group(ranks, backend="gloo")
        self._peer = self._channel_peer(endpoints)
        self._hidden_size = self.model_config.get_hidden_size()
        self._sends: list[tuple[dist.Work, torch.Tensor]] = []
        self.trace = LwdSimTraceWriter(self.trace_source)

    def _channel_peer(self, endpoints: LwdChannelEndpoints) -> int:
        raise NotImplementedError

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        return

    def compile_or_warm_up_model(self) -> CompilationTimes:
        return CompilationTimes(language_model=0.0, encoder=0.0)

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return ("generate",)

    def get_supported_kv_cache_layouts(self) -> list[str]:
        return [_LWD_SIM_KV_CACHE_LAYOUT]

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        return

    def _send(self, group: dist.ProcessGroup, seqno: int, numel: int) -> None:
        """Post a stamped send, keeping the buffer referenced until shutdown
        (gloo isends may not report completion without a wait, so completed
        ones are not pruned; a run holds a few buffers only)."""
        payload = torch.full((numel,), float(seqno), dtype=_LWD_SIM_PAYLOAD_DTYPE)
        work = dist.isend(payload, dst=self._peer, group=group)
        self._sends.append((work, payload))


class LwdSimEdgeWorker(_LwdSimWorkerBase):
    """Edge: one UP send per EMBED batch, one DOWN receive per UNEMBED batch
    returning the accepted tokens; no KV cache, as with zero decoder layers."""

    trace_source = "edge-worker"

    def _channel_peer(self, endpoints: LwdChannelEndpoints) -> int:
        return endpoints.cloud_rank

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return {}

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | None:
        lwd_batch = scheduler_output.lwd_batch
        if lwd_batch is None:
            return None
        meta = lwd_batch.batch_meta
        if lwd_batch.batch_type == LwdBatchType.LWD_EMBED:
            assert isinstance(meta, LwdEmbedBatch)
            self._send_up(lwd_batch.seqno, meta)
            return None
        assert isinstance(meta, LwdUnembedBatch)
        return self._receive_down(lwd_batch.seqno, meta)

    def _send_up(self, seqno: int, meta: LwdEmbedBatch) -> None:
        num_tokens = [len(token_ids) for token_ids in meta.token_ids]
        numel = sum(num_tokens) * self._hidden_size
        self.trace.record(
            "up_send",
            seqno=seqno,
            numel=numel,
            req_ids=list(meta.req_ids),
            offsets=list(meta.prompt_offsets),
            num_tokens=num_tokens,
        )
        self._send(self._up_group, seqno, numel)

    def _receive_down(self, seqno: int, meta: LwdUnembedBatch) -> ModelRunnerOutput:
        self.trace.record(
            "down_recv_post",
            seqno=seqno,
            numel=meta.recv_num_elements,
            req_ids=list(meta.req_ids),
            num_accept_tokens=list(meta.num_accept_tokens),
            top_id_ths=[list(ranks) for ranks in meta.top_id_ths],
        )
        buffer = torch.empty(meta.recv_num_elements, dtype=_LWD_SIM_PAYLOAD_DTYPE)
        dist.recv(buffer, src=self._peer, group=self._down_group)
        self.trace.record("down_recv_done", seqno=seqno, stamp=_lwd_sim_stamp(buffer))
        return ModelRunnerOutput(
            req_ids=list(meta.req_ids),
            req_id_to_index={req_id: i for i, req_id in enumerate(meta.req_ids)},
            sampled_token_ids=[
                [LWD_SIM_SAMPLED_TOKEN_ID] * accepted
                for accepted in meta.num_accept_tokens
            ],
        )


@dataclass
class _LwdSimUpRecv:
    work: dist.Work
    buffer: torch.Tensor
    reported: bool = False
    consumed: bool = False


class LwdSimCloudWorker(_LwdSimWorkerBase):
    """Cloud endpoint: an early-receive thread posts the UP receive of each
    hint and reports its completion; each step samples one token for every
    request with a complete prompt and sends their DOWN rows, attaching the
    carrier the cloud scheduler unpacks."""

    trace_source = "cloud-worker"

    def init_device(self) -> None:
        super().init_device()
        self._prompt_lens: dict[str, int] = {}
        self._step: tuple[list[str], list[str]] | None = None
        self._down_seqno = 0
        self._up_recvs: dict[int, _LwdSimUpRecv] = {}
        self._up_lock = threading.Lock()
        hint_mq = attach_hint_mq()
        assert hint_mq is not None, "the cloud engine exports the hint MQ"
        self._hint_mq = hint_mq
        self._done_mq = create_report_mq()
        self._stopped = threading.Event()
        self._early_recv_thread = threading.Thread(
            target=self._early_recv_loop, name="lwd-sim-early-recv", daemon=True
        )
        self._early_recv_thread.start()

    def _channel_peer(self, endpoints: LwdChannelEndpoints) -> int:
        return endpoints.edge_rank

    def lwd_early_recv_done_handle(self) -> Handle:
        return self._done_mq.export_handle()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        spec = FullAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=self.model_config.get_num_kv_heads(self.parallel_config),
            head_size=self.model_config.get_head_size(),
            dtype=self.model_config.dtype,
        )
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        return {f"model.layers.{i}.self_attn.attn": spec for i in range(num_layers)}

    def determine_available_memory(self) -> int:
        specs = self.get_kv_cache_spec().values()
        return _LWD_SIM_NUM_KV_BLOCKS * sum(spec.page_size_bytes for spec in specs)

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | None:
        for req_id in scheduler_output.finished_req_ids:
            self._prompt_lens.pop(req_id, None)
        if not scheduler_output.total_num_scheduled_tokens:
            return EMPTY_MODEL_RUNNER_OUTPUT
        lwd_batch = scheduler_output.lwd_batch
        if lwd_batch is not None:
            self._consume_up(lwd_batch.seqno)
        num_computed = self._register_requests(scheduler_output)
        scheduled = scheduler_output.num_scheduled_tokens
        req_ids = list(scheduled)
        sampled = [
            req_id
            for req_id in req_ids
            if num_computed[req_id] + scheduled[req_id] >= self._prompt_lens[req_id]
        ]
        self.trace.record(
            "step",
            scheduled=dict(scheduled),
            embed=_lwd_sim_embed_fields(lwd_batch),
            sampled=sampled,
        )
        self._step = (req_ids, sampled)
        return None

    def _register_requests(self, scheduler_output: SchedulerOutput) -> dict[str, int]:
        """Tokens computed before this step, per scheduled request."""
        num_computed: dict[str, int] = {}
        for new_req in scheduler_output.scheduled_new_reqs:
            assert new_req.prompt_embeds is not None, "cloud prompts are embeds"
            self._prompt_lens[new_req.req_id] = new_req.prompt_embeds.shape[0]
            num_computed[new_req.req_id] = new_req.num_computed_tokens
        cached = scheduler_output.scheduled_cached_reqs
        num_computed.update(zip(cached.req_ids, cached.num_computed_tokens))
        return num_computed

    def sample_tokens(self, grammar_output: GrammarOutput | None) -> ModelRunnerOutput:
        assert self._step is not None, "sample_tokens without execute_model"
        req_ids, sampled = self._step
        self._step = None
        sampled_set = set(sampled)
        output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[
                [LWD_SIM_SAMPLED_TOKEN_ID] if req_id in sampled_set else []
                for req_id in req_ids
            ],
        )
        if sampled:
            output.lwd_down_carrier = self._send_down(sampled)  # type: ignore[attr-defined]
        return output

    def _send_down(self, req_ids: list[str]) -> LwdDownCarrier:
        seqno = self._down_seqno
        self._down_seqno += 1
        numel = len(req_ids) * self._hidden_size
        ranks = [lwd_sim_token_rank(seqno, row) for row in range(len(req_ids))]
        self.trace.record(
            "down_send", seqno=seqno, numel=numel, req_ids=req_ids, ranks=ranks
        )
        self._send(self._down_group, seqno, numel)
        # Non-spec layout of the real worker: [ranks | accepted | segment lengths].
        packed = torch.tensor(ranks + [1] * (2 * len(req_ids)), dtype=torch.int32)
        return packed, req_ids, numel, seqno

    def _early_recv_loop(self) -> None:
        while not self._stopped.is_set():
            try:
                hint = self._hint_mq.dequeue(timeout=_LWD_SIM_HINT_TIMEOUT_S)
            except TimeoutError:
                pass
            except Exception:
                if self._stopped.is_set():
                    return
                raise
            else:
                self._post_up_recv(hint)
            self._report_up_recvs()
            time.sleep(_LWD_SIM_POLL_INTERVAL_S)

    def _post_up_recv(self, hint: dict[str, Any]) -> None:
        fields = hint_fields(hint)
        assert fields is not None, f"malformed early-receive hint {hint}"
        seqno, num_tokens, _ = fields
        numel = num_tokens * self._hidden_size
        self.trace.record("up_recv_post", seqno=seqno, numel=numel)
        buffer = torch.empty(numel, dtype=_LWD_SIM_PAYLOAD_DTYPE)
        work = dist.irecv(buffer, src=self._peer, group=self._up_group)
        with self._up_lock:
            self._up_recvs[seqno] = _LwdSimUpRecv(work, buffer)

    def _report_up_recvs(self) -> None:
        with self._up_lock:
            pending = [(s, r) for s, r in self._up_recvs.items() if not r.reported]
        for seqno, recv in pending:
            # An unwaited gloo work never reports completion through
            # is_completed(); a timed wait drives the op, and only a work
            # that then reports completed is done.
            try:
                recv.work.wait(timedelta(seconds=_LWD_SIM_RECV_POLL_S))
            except Exception:  # noqa: BLE001 -- timeout; retried next round
                continue
            if not recv.work.is_completed():
                continue
            self.trace.record(
                "up_recv_done", seqno=seqno, stamp=_lwd_sim_stamp(recv.buffer)
            )
            self._done_mq.enqueue((LwdChannelType.UP, seqno))
            with self._up_lock:
                recv.reported = True
                if recv.consumed:
                    del self._up_recvs[seqno]

    def _consume_up(self, seqno: int) -> None:
        """The batch's UP data must have been received: the engine gate only
        dispatches a batch whose completion was reported."""
        with self._up_lock:
            recv = self._up_recvs.get(seqno)
            reported = recv is not None and recv.reported
        self.trace.record("up_consume", seqno=seqno, reported=reported)
        if recv is None:
            raise RuntimeError(f"UP seqno {seqno} dispatched before it was posted")
        if not recv.reported:
            # Only the gate-missed path: waiting here is safe because the
            # comm thread no longer touches a reported work.
            recv.work.wait()
        with self._up_lock:
            recv.consumed = True
            if recv.reported:
                del self._up_recvs[seqno]

    def shutdown(self) -> None:
        self._stopped.set()
        self._early_recv_thread.join(_LWD_SIM_THREAD_JOIN_TIMEOUT_S)


def _lwd_sim_embed_fields(lwd_batch: LwdBatch | None) -> dict[str, Any] | None:
    if lwd_batch is None:
        return None
    meta = lwd_batch.batch_meta
    assert isinstance(meta, LwdEmbedBatch)
    return {
        "seqno": lwd_batch.seqno,
        "req_ids": list(meta.req_ids),
        "offsets": list(meta.prompt_offsets),
        "num_tokens": [len(token_ids) for token_ids in meta.token_ids],
    }
