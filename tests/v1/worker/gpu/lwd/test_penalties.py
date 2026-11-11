# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.lwd.penalties import LwdCloudPenaltiesState
from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsProcRequestState
from vllm.v1.worker.gpu.states import RequestState


def test_prompt_mask_is_cleared_without_touching_output_counts(monkeypatch) -> None:
    request_state = RequestState(
        max_num_reqs=1,
        max_model_len=8,
        max_num_batched_tokens=8,
        num_speculative_steps=0,
        vocab_size=8,
        device=torch.device("cpu"),
    )
    logits_state = LogitsProcRequestState.from_request_state(request_state)
    state = LwdCloudPenaltiesState(VllmConfig(), logits_state)
    state.prompt_bin_mask.fill_(1)
    state._new_penalties_reqs.append(0)

    def count_output(*args) -> None:
        state.output_bin_counts[0, 3] = 2

    monkeypatch.setattr("vllm.v1.worker.gpu.sample.penalties.bincount", count_output)
    state.apply_staged_writes()

    assert not state.prompt_bin_mask.any()
    assert state.output_bin_counts[0, 3].item() == 2
