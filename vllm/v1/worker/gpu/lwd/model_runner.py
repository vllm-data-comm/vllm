# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_tp_group
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.lwd.cloud_input import LwdCloudBatchView
from vllm.v1.worker.gpu.lwd.down_payload import (
    LwdDownBatch,
    LwdDownPayload,
    LwdDownPayloadBuilder,
)
from vllm.v1.worker.gpu.lwd.model_state import LwdCloudModelStateMixin
from vllm.v1.worker.gpu.lwd.penalties import LwdCloudPenaltiesState
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.output import SamplerOutput


def _validate_lwd_v2_config(vllm_config: VllmConfig) -> None:
    speculative_config = vllm_config.speculative_config
    if speculative_config is not None and (
        speculative_config.method != "mtp"
        or speculative_config.use_step3p5_mtp()
        or speculative_config.use_multi_module_mtp()
    ):
        raise ValueError("[LWD] MRV2 currently supports only standard MTP speculation")
    parallel_config = vllm_config.parallel_config
    if parallel_config.enable_batch_sharded_sampling:
        raise ValueError("[LWD] MRV2 does not support batch-sharded sampling")
    if parallel_config.prefill_context_parallel_size != 1:
        raise ValueError("[LWD] MRV2 PCP support is deferred")
    if parallel_config.decode_context_parallel_size != 1:
        raise ValueError("[LWD] MRV2 DCP support is deferred")
    if parallel_config.enable_dbo:
        raise ValueError("[LWD] MRV2 DBO is not supported")


class LwdCloudModelRunnerMixin(GPUModelRunner):
    model_state: LwdCloudModelStateMixin
    runner_kind = "v2"

    def init_lwd_cloud_runner(self, worker: Any) -> None:
        _validate_lwd_v2_config(self.vllm_config)
        self.worker = worker
        self._lwd_pending_down_payload: LwdDownPayload | None = None
        self._lwd_down_builder = LwdDownPayloadBuilder(self.device)

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        super().load_model(*args, **kwargs)
        if not isinstance(self.model_state, LwdCloudModelStateMixin):
            raise TypeError(
                "[LWD] MRV2 requires the LWD model state, got "
                f"{type(self.model_state).__name__}"
            )
        prompt_stage = getattr(self.worker, "_lwd_stager", None)
        self.model_state.bind_lwd_prompt_stage(prompt_stage)
        self._install_lwd_penalties()
        self._install_lwd_model_wrapper()

    def _install_lwd_penalties(self) -> None:
        if self.sampler is None:
            return
        previous = self.sampler.penalties_state
        penalties = LwdCloudPenaltiesState(self.vllm_config, previous.req_states)
        self.sampler.penalties_state = penalties
        self.sampler.logits_processors = [
            penalties if processor is previous else processor
            for processor in self.sampler.logits_processors
        ]

    def _install_lwd_model_wrapper(self) -> None:
        return

    def update_requests(self, scheduler_output: SchedulerOutput) -> None:
        super().update_requests(scheduler_output)
        if self.worker is None:
            return
        view = LwdCloudBatchView(
            req_ids=list(scheduler_output.scheduled_cached_reqs.req_ids),
            idx_mapping_np=np.array(
                [
                    self.req_states.req_id_to_index[req_id]
                    for req_id in scheduler_output.scheduled_cached_reqs.req_ids
                ],
                dtype=np.intp,
            ),
            num_scheduled_tokens=np.array(
                [
                    scheduler_output.num_scheduled_tokens[req_id]
                    for req_id in scheduler_output.scheduled_cached_reqs.req_ids
                ],
                dtype=np.int32,
            ),
            num_computed_tokens_np=np.array(
                scheduler_output.scheduled_cached_reqs.num_computed_tokens,
                dtype=np.int32,
            ),
            prefill_len_np=np.array(
                [
                    self.req_states.prefill_len.np[
                        self.req_states.req_id_to_index[req_id]
                    ]
                    for req_id in scheduler_output.scheduled_cached_reqs.req_ids
                ],
                dtype=np.int32,
            ),
            query_start_loc_np=np.array([], dtype=np.int32),
        )
        self.model_state.lwd_input_assembler.align_consumed(view)

    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: GrammarOutput | None,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
        wire_endpoint = get_tp_group().is_first_rank
        if not wire_endpoint:
            return super().sample(hidden_states, input_batch, grammar_output)
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        req_ids = list(input_batch.req_ids)
        valid_rows = input_batch.idx_mapping_np >= 0
        num_draft = input_batch.num_draft_tokens_per_req
        if num_draft is not None:
            num_draft = num_draft.copy()
        self.model_state.clear_lwd_sample_logits()
        sampler_output, num_sampled, num_rejected = super().sample(
            hidden_states, input_batch, grammar_output
        )
        logits = self.model_state.take_lwd_sample_logits()
        if logits is not None:
            if logits.shape[0] != sample_hidden_states.shape[0]:
                raise RuntimeError("[LWD] captured logits do not match sampled rows")
            self._lwd_pending_down_payload = self._lwd_down_builder.build(
                LwdDownBatch(
                    sample_hidden_states=sample_hidden_states,
                    logits=logits,
                    sampled_token_ids=sampler_output.sampled_token_ids,
                    req_ids=req_ids,
                    valid_rows=valid_rows,
                    num_draft_tokens_per_req=num_draft,
                )
            )
        return sampler_output, num_sampled, num_rejected

    def take_lwd_pending_down_payload(self) -> LwdDownPayload | None:
        payload = self._lwd_pending_down_payload
        self._lwd_pending_down_payload = None
        return payload

    def lwd_release_finished(self, finished_req_ids: list[str]) -> None:
        for req_id in finished_req_ids:
            req_index = self.req_states.req_id_to_index.get(req_id)
            if req_index is not None:
                self.model_state.next_prompt_embeds[req_index].zero_()


class LwdCloudModelRunnerV2(LwdCloudModelRunnerMixin, GPUModelRunner):
    def __init__(
        self, vllm_config: Any, device: torch.device, worker: Any = None
    ) -> None:
        super().__init__(vllm_config, device)
        self.init_lwd_cloud_runner(worker)
