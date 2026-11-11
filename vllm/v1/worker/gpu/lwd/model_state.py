# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.config import VllmConfig
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.lwd.cloud_input import (
    LwdCloudBatchView,
    LwdCloudInputAssembler,
    LwdPromptStage,
)
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState


class LwdCloudModelStateMixin(DefaultModelState):
    def init_lwd_cloud_state(self, enable_lookahead: bool) -> None:
        self.prompt_embeds_state = None
        self.remote_inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.inputs_embeds_size),
            dtype=self.dtype,
            device=self.device,
        )
        self.next_prompt_embeds = torch.zeros(
            (self.max_num_reqs, self.inputs_embeds_size),
            dtype=self.dtype,
            device=self.device,
        )
        self.lwd_input_assembler = LwdCloudInputAssembler(
            device=self.device,
            hidden_size=self.inputs_embeds_size,
            max_num_tokens=self.max_num_tokens,
            enable_lookahead=enable_lookahead,
        )
        self._lwd_sample_logits: torch.Tensor | None = None
        self._lwd_req_indices: dict[str, int] = {}

    def bind_lwd_prompt_stage(self, prompt_stage: LwdPromptStage | None) -> None:
        self.lwd_input_assembler.bind_prompt_stage(prompt_stage)

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        self.next_prompt_embeds[req_index].zero_()
        self._lwd_req_indices[new_req_data.req_id] = req_index
        payload = new_req_data.lwd_mrope_positions
        if not payload:
            return
        prompt_len = new_req_data.prompt_len
        if prompt_len <= 0:
            raise ValueError("[LWD] M-RoPE prompt length is missing")
        expected_numel = 3 * prompt_len
        if len(payload) != expected_numel * 8:
            raise ValueError(
                f"[LWD] invalid M-RoPE payload for {new_req_data.req_id}: "
                f"{len(payload)} != {expected_numel * 8}"
            )
        positions = torch.frombuffer(bytearray(payload), dtype=torch.int64).view(
            3, prompt_len
        )
        delta = int(positions.max().item()) + 1 - prompt_len
        if self.rope_state is not None:
            self.rope_state.update_prefill_positions(req_index, positions, delta)

    def remove_request(self, req_id: str) -> None:
        req_index = self._lwd_req_indices.pop(req_id, None)
        if req_index is not None:
            self.next_prompt_embeds[req_index].zero_()
        super().remove_request(req_id)

    def dummy_inputs_embeds(self, num_tokens: int) -> torch.Tensor:
        return self.remote_inputs_embeds[:num_tokens]

    def prepare_inputs_embeds(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        del scheduled_encoder_inputs
        num_tokens = input_batch.num_tokens
        safe_input_ids = input_batch.input_ids[:num_tokens].clone()
        for index, req_id in enumerate(input_batch.req_ids):
            del req_id
            start = int(input_batch.query_start_loc_np[index])
            end = int(input_batch.query_start_loc_np[index + 1])
            prompt_len = int(input_batch.prefill_len_np[index])
            computed = int(input_batch.num_computed_tokens_np[index])
            prompt_rows = min(max(prompt_len - computed, 0), end - start)
            if prompt_rows > 0:
                safe_input_ids[start : start + prompt_rows] = 0
        input_embeddings = self.model.embed_input_ids(safe_input_ids)
        self.remote_inputs_embeds[:num_tokens].copy_(input_embeddings)
        view = LwdCloudBatchView(
            req_ids=list(input_batch.req_ids),
            idx_mapping_np=input_batch.idx_mapping_np,
            num_scheduled_tokens=input_batch.num_scheduled_tokens,
            num_computed_tokens_np=input_batch.num_computed_tokens_np,
            prefill_len_np=input_batch.prefill_len_np,
            query_start_loc_np=input_batch.query_start_loc_np,
        )
        self.lwd_input_assembler.stage_prompt_window(
            view,
            self.remote_inputs_embeds,
            self.next_prompt_embeds,
            input_batch.input_ids[:num_tokens],
        )
        return self.remote_inputs_embeds[: input_batch.num_tokens_after_padding]

    def capture_lwd_sample_logits(self, logits: torch.Tensor) -> None:
        self._lwd_sample_logits = logits

    def compute_sample_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = super().compute_sample_logits(hidden_states)
        self.capture_lwd_sample_logits(logits)
        return logits

    def clear_lwd_sample_logits(self) -> None:
        self._lwd_sample_logits = None

    def take_lwd_sample_logits(self) -> torch.Tensor | None:
        return self._lwd_sample_logits


class LwdCloudModelState(LwdCloudModelStateMixin, DefaultModelState):
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: torch.nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self.init_lwd_cloud_state(
            enable_lookahead=vllm_config.num_prefill_lookahead_tokens > 0
        )
