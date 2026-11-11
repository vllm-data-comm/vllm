# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.lwd.model_state import LwdCloudModelStateMixin
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator


class LwdCloudMTPSpeculatorMixin(MTPSpeculator):
    model_state: LwdCloudModelStateMixin

    def load_model(self, target_model: torch.nn.Module) -> None:
        super().load_model(target_model)
        self.supports_mm_inputs = False
        self.inputs_embeds = torch.zeros(
            self.max_num_tokens,
            self.vllm_config.model_config.get_inputs_embeds_size(),
            dtype=self.dtype,
            device=self.device,
        )
        self._lwd_draft_batch: InputBatch | None = None
        self._lwd_use_remote_embeds = False

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        self._lwd_draft_batch = None if dummy_run else input_batch
        self._lwd_use_remote_embeds = False
        try:
            return super().propose(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                dp_sync_state,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile=is_profile,
                num_speculative_tokens=num_speculative_tokens,
            )
        finally:
            self._lwd_draft_batch = None
            self._lwd_use_remote_embeds = False

    def on_prefill_begin(self, num_reqs: int) -> None:
        super().on_prefill_begin(num_reqs)
        if self._lwd_draft_batch is not None:
            self._prepare_lwd_prefill_embeds(self._lwd_draft_batch)

    def on_prefill_end(self, num_reqs: int) -> None:
        super().on_prefill_end(num_reqs)
        self._lwd_draft_batch = None
        self._lwd_use_remote_embeds = False

    def _prepare_draft_model_inputs(
        self,
        input_buffers: InputBatch | InputBuffers,
        num_tokens: int,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> dict[str, torch.Tensor | None]:
        model_inputs = super()._prepare_draft_model_inputs(
            input_buffers, num_tokens, mm_inputs
        )
        if self._lwd_use_remote_embeds:
            assert self.inputs_embeds is not None
            model_inputs["inputs_embeds"] = self.inputs_embeds[:num_tokens]
        return model_inputs

    def _prepare_lwd_prefill_embeds(self, input_batch: InputBatch) -> None:
        assert self.inputs_embeds is not None
        num_tokens = input_batch.num_tokens
        has_prompt = False
        for index in range(input_batch.num_reqs):
            if input_batch.idx_mapping_np[index] < 0:
                continue
            if self._lwd_prompt_rows(input_batch, index) > 0:
                has_prompt = True

        self._lwd_use_remote_embeds = has_prompt
        if not has_prompt:
            return
        safe_input_ids = self.input_buffers.input_ids[:num_tokens].clone()
        for index in range(input_batch.num_reqs):
            if input_batch.idx_mapping_np[index] < 0:
                continue
            start = int(input_batch.query_start_loc_np[index])
            prompt_rows = self._lwd_prompt_rows(input_batch, index)
            if prompt_rows > 0:
                safe_input_ids[start : start + prompt_rows - 1] = 0

        self.inputs_embeds[:num_tokens].copy_(
            self.model.embed_input_ids(safe_input_ids)
        )
        self.inputs_embeds[num_tokens : input_batch.num_tokens_after_padding].zero_()
        for index in range(input_batch.num_reqs):
            self._overlay_lwd_prompt(input_batch, index)

    @staticmethod
    def _lwd_prompt_rows(input_batch: InputBatch, index: int) -> int:
        computed = int(input_batch.num_computed_tokens_np[index])
        prompt_len = int(input_batch.prefill_len_np[index])
        query_len = int(
            input_batch.query_start_loc_np[index + 1]
            - input_batch.query_start_loc_np[index]
        )
        return min(max(prompt_len - computed, 0), query_len)

    def _overlay_lwd_prompt(self, input_batch: InputBatch, index: int) -> None:
        assert self.inputs_embeds is not None
        if input_batch.idx_mapping_np[index] < 0:
            return
        prompt_rows = self._lwd_prompt_rows(input_batch, index)
        if prompt_rows == 0:
            return
        req_state_idx = int(input_batch.idx_mapping_np[index])
        start = int(input_batch.query_start_loc_np[index])
        computed = int(input_batch.num_computed_tokens_np[index])
        remote = self.model_state.remote_inputs_embeds
        if prompt_rows > 1:
            self.inputs_embeds[start : start + prompt_rows - 1].copy_(
                remote[start + 1 : start + prompt_rows]
            )
        if computed + prompt_rows < int(input_batch.prefill_len_np[index]):
            self.inputs_embeds[start + prompt_rows - 1].copy_(
                self.model_state.next_prompt_embeds[req_state_idx]
            )


class LwdCloudMTPSpeculator(LwdCloudMTPSpeculatorMixin, MTPSpeculator):
    pass
