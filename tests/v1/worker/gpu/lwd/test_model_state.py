# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from transformers import PreTrainedConfig

from vllm.config import VllmConfig
from vllm.config.model import ModelConfig
from vllm.v1.worker.gpu.lwd.model_state import LwdCloudModelState


class _EmbeddingModel(torch.nn.Module):
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.to(torch.bfloat16)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * 2


class _Qwen3Config(PreTrainedConfig):
    model_type = "qwen3"


def _model_config(tmp_path) -> ModelConfig:
    hf_config = _Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        hidden_size=2,
        vocab_size=8,
        num_hidden_layers=0,
        num_attention_heads=1,
        num_key_value_heads=1,
    )
    hf_config.model_type = "qwen3"
    hf_config.save_pretrained(tmp_path)
    return ModelConfig(model=str(tmp_path), enable_prompt_embeds=True, max_model_len=16)


def test_lwd_state_owns_fixed_buffers_and_captures_logits(tmp_path) -> None:
    config = VllmConfig(model_config=_model_config(tmp_path))
    state = LwdCloudModelState(config, _EmbeddingModel(), None, torch.device("cpu"))

    assert state.prompt_embeds_state is None
    dummy_ptr = state.dummy_inputs_embeds(1).data_ptr()
    assert dummy_ptr == state.remote_inputs_embeds.data_ptr()

    logits = state.compute_sample_logits(torch.tensor([1.0, 2.0]))
    assert logits.tolist() == [2.0, 4.0]
    assert state.take_lwd_sample_logits() is logits

    state.clear_lwd_sample_logits()
    assert state.take_lwd_sample_logits() is None
