# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import numpy as np
import torch

from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.lwd.model_state import LwdCloudModelState
from vllm.v1.worker.gpu.lwd.mtp_speculator import LwdCloudMTPSpeculator


class _EmbeddingModel(torch.nn.Module):
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.to(torch.float32).unsqueeze(-1)


def _batch(computed: int) -> InputBatch:
    buffers = InputBuffers(3, 4, torch.device("cpu"))
    buffers.input_ids[:4] = torch.tensor([0, 0, 7, 0])
    batch = InputBatch.make_dummy(1, 4, buffers)
    return replace(
        batch,
        req_ids=["r1"],
        idx_mapping=torch.tensor([2]),
        idx_mapping_np=np.array([2]),
        num_scheduled_tokens=np.array([3], dtype=np.int32),
        query_start_loc=torch.tensor([0, 3]),
        query_start_loc_np=np.array([0, 3], dtype=np.int32),
        seq_lens=torch.tensor([computed + 3]),
        num_computed_tokens_np=np.array([computed], dtype=np.int32),
        prefill_len_np=np.array([4], dtype=np.int32),
        has_prefill=computed < 4,
    )


def _speculator() -> LwdCloudMTPSpeculator:
    speculator = LwdCloudMTPSpeculator.__new__(LwdCloudMTPSpeculator)
    speculator.max_num_tokens = 4
    speculator.dtype = torch.float32
    speculator.device = torch.device("cpu")
    speculator.input_buffers = InputBuffers(3, 4, speculator.device)
    speculator.input_buffers.input_ids[:4] = torch.tensor([0, 0, 7, 0])
    speculator.inputs_embeds = torch.zeros((4, 1))
    speculator.supports_mm_inputs = False
    speculator.hidden_states = torch.zeros((4, 1))
    speculator.model = _EmbeddingModel()
    speculator.model_state = LwdCloudModelState.__new__(LwdCloudModelState)
    speculator.model_state.remote_inputs_embeds = torch.tensor(
        [[1.0], [2.0], [3.0], [4.0]]
    )
    speculator.model_state.next_prompt_embeds = torch.tensor([[0.0], [0.0], [11.0]])
    speculator._lwd_draft_batch = None
    speculator._lwd_use_remote_embeds = False
    return speculator


def test_nonfinal_prefill_uses_shifted_remote_and_lookahead_embeddings() -> None:
    speculator = _speculator()
    batch = _batch(computed=0)
    speculator._lwd_draft_batch = batch

    speculator.on_prefill_begin(1)
    model_inputs = speculator._prepare_draft_model_inputs(
        speculator.input_buffers, 4, None
    )

    inputs_embeds = model_inputs["inputs_embeds"]
    assert inputs_embeds is not None
    assert speculator.inputs_embeds is not None
    assert inputs_embeds.squeeze(-1).tolist() == [2.0, 3.0, 11.0, 0.0]
    assert speculator.inputs_embeds.data_ptr() == inputs_embeds.data_ptr()
    speculator.on_prefill_end(1)


def test_pure_decode_keeps_native_token_input_path() -> None:
    speculator = _speculator()
    batch = _batch(computed=4)
    speculator._lwd_draft_batch = batch

    speculator.on_prefill_begin(1)
    model_inputs = speculator._prepare_draft_model_inputs(
        speculator.input_buffers, 4, None
    )

    input_ids = model_inputs["input_ids"]
    assert not speculator._lwd_use_remote_embeds
    assert model_inputs["inputs_embeds"] is None
    assert input_ids is not None
    assert input_ids.data_ptr() == speculator.input_buffers.input_ids.data_ptr()
    speculator.on_prefill_end(1)
