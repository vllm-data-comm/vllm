# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.sample.output import SamplerOutput


def test_async_output_exposes_v1_compatible_inner_output() -> None:
    inner = ModelRunnerOutput(req_ids=[], req_id_to_index={})
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.empty((0, 1), dtype=torch.int64),
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=torch.empty(0, dtype=torch.int32),
        num_rejected=torch.empty(0, dtype=torch.int32),
    )
    output = AsyncOutput(
        model_runner_output=inner,
        sampler_output=sampler_output,
        num_sampled_tokens=torch.empty(0, dtype=torch.int32),
        main_stream=torch.cuda.Stream(),
        copy_stream=torch.cuda.Stream(),
        check_ep_fault=False,
        pending_aux_output=None,
    )

    assert output._model_runner_output is inner
