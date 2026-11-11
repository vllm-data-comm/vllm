# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.v1.worker.gpu.lwd.down_payload import LwdDownBatch, LwdDownPayloadBuilder
from vllm.v1.worker.gpu.lwd.pinned_staging import LwdPinnedStaging
from vllm.v1.worker.gpu.lwd.token_rank import LwdTokenRankEncoder


def test_rank_encoder_uses_logit_then_token_id_order() -> None:
    encoder = LwdTokenRankEncoder()
    logits = torch.tensor([[4.0, 4.0, 4.0]]).repeat(3, 1)
    selected = torch.tensor([1, 2, -1])

    ranks = encoder.encode(logits, selected)

    assert ranks.tolist() == [1, 2, 0]


def test_pinned_staging_reuses_capacity() -> None:
    staging = LwdPinnedStaging(dtype=torch.int64, min_capacity=4, device="cpu")
    first = staging.stage_to_device(np.array([1, 2]), 2)
    second = staging.stage_to_device(np.array([3, 4]), 2)

    assert first.tolist() == [1, 2]
    assert second.tolist() == [3, 4]
    assert first.data_ptr() != second.data_ptr()


def test_plain_down_payload_preserves_valid_request_layout() -> None:
    builder = LwdDownPayloadBuilder(torch.device("cpu"))
    batch = LwdDownBatch(
        sample_hidden_states=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        logits=torch.tensor(
            [[0.0, 3.0, 1.0], [0.0, 2.0, 1.0], [0.0, 1.0, 2.0]],
            dtype=torch.float32,
        ),
        sampled_token_ids=torch.tensor([[1], [2], [2]]),
        req_ids=["a", "b", "c"],
        valid_rows=np.array([True, False, True]),
    )

    payload = builder.build(batch)

    assert payload is not None
    assert payload.hidden.tolist() == [[1.0, 2.0], [5.0, 6.0]]
    assert payload.meta_dev.tolist() == [0, 0, 1, 1, 1, 1]
    assert payload.seg_lens_host is None
    assert payload.req_ids == ["a", "c"]


def test_spec_down_payload_truncates_rejected_rows() -> None:
    builder = LwdDownPayloadBuilder(torch.device("cpu"))
    batch = LwdDownBatch(
        sample_hidden_states=torch.arange(12, dtype=torch.float32).view(4, 3),
        logits=torch.tensor(
            [
                [0.0, 3.0, 1.0, 2.0],
                [0.0, 3.0, 1.0, 2.0],
                [0.0, 2.0, 1.0, 3.0],
                [0.0, 2.0, 1.0, 3.0],
            ]
        ),
        sampled_token_ids=torch.tensor([[1, 2, -1], [3, 2, -1]]),
        req_ids=["a", "b"],
        valid_rows=np.array([True, True]),
        num_draft_tokens_per_req=np.array([2, 1], dtype=np.int32),
    )

    payload = builder.build(batch)

    assert payload is not None
    assert payload.hidden.shape == (4, 3)
    assert payload.meta_dev.tolist() == [0, 2, 0, 0, 2, 1]
    assert payload.seg_lens_host == [3, 1]
    assert payload.req_ids == ["a", "b"]


def test_down_payload_returns_none_without_valid_requests() -> None:
    builder = LwdDownPayloadBuilder(torch.device("cpu"))
    batch = LwdDownBatch(
        sample_hidden_states=torch.empty((1, 2)),
        logits=torch.empty((1, 3)),
        sampled_token_ids=torch.tensor([[1]]),
        req_ids=["a"],
        valid_rows=np.array([False]),
    )

    assert builder.build(batch) is None
