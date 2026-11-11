# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import numpy as np
import torch

from vllm.v1.worker.gpu.lwd.cloud_input import (
    LwdCloudBatchView,
    LwdCloudInputAssembler,
)


@dataclass
class _StageRequest:
    host: torch.Tensor


class _PromptStage:
    def __init__(self, host: torch.Tensor, hidden_size: int) -> None:
        self.request = _StageRequest(host)
        self.hidden_size = hidden_size
        self.recv = host.shape[0] // hidden_size
        self.consumed = 0

    def take_req(self, req_id: str) -> _StageRequest:
        return self.request

    def recv_rows(self, req_id: str) -> int:
        return self.recv

    def consumed_rows(self, req_id: str) -> int:
        return self.consumed

    def segment_for(self, req_id: str, n_rows: int) -> tuple[int, int, int]:
        start = self.consumed * self.hidden_size
        return start, n_rows * self.hidden_size, 0

    def consume(self, req_id: str, n_rows: int) -> None:
        self.consumed += n_rows

    def rewind_consumed(self, req_id: str, to_rows: int) -> bool:
        self.consumed = to_rows
        return True

    def skip_consumed(self, req_id: str, to_rows: int) -> bool:
        if to_rows < self.consumed or to_rows > self.recv:
            raise ValueError("skip target outside arrival window")
        self.consumed = to_rows
        return True


def _view(computed: int, scheduled: int, prefill_len: int) -> LwdCloudBatchView:
    return LwdCloudBatchView(
        req_ids=["r1"],
        idx_mapping_np=np.array([0]),
        num_scheduled_tokens=np.array([scheduled], dtype=np.int32),
        num_computed_tokens_np=np.array([computed], dtype=np.int32),
        prefill_len_np=np.array([prefill_len], dtype=np.int32),
        query_start_loc_np=np.array([0, scheduled], dtype=np.int32),
    )


def test_stage_window_injects_remote_and_lookahead_rows() -> None:
    stage = _PromptStage(torch.arange(1, 9, dtype=torch.bfloat16), 2)
    assembler = LwdCloudInputAssembler(
        device=torch.device("cpu"),
        hidden_size=2,
        max_num_tokens=4,
        enable_lookahead=True,
        prompt_stage=stage,
    )
    inputs_embeds = torch.zeros((2, 2), dtype=torch.bfloat16)
    next_prompt_embeds = torch.zeros((1, 2), dtype=torch.bfloat16)

    assembler.stage_prompt_window(_view(0, 2, 4), inputs_embeds, next_prompt_embeds)

    assert inputs_embeds.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert next_prompt_embeds.tolist() == [[5.0, 6.0]]
    assert stage.consumed == 2


def test_align_consumed_rewinds_and_skips() -> None:
    stage = _PromptStage(torch.arange(1, 9, dtype=torch.bfloat16), 2)
    stage.consumed = 3
    assembler = LwdCloudInputAssembler(
        device=torch.device("cpu"),
        hidden_size=2,
        max_num_tokens=4,
        enable_lookahead=False,
        prompt_stage=stage,
    )

    assembler.align_consumed(_view(1, 1, 4))
    assert stage.consumed == 1
    assembler.align_consumed(_view(3, 1, 4))
    assert stage.consumed == 3


def test_stage_position_mismatch_fails_fast() -> None:
    stage = _PromptStage(torch.arange(1, 9, dtype=torch.bfloat16), 2)
    stage.consumed = 1
    assembler = LwdCloudInputAssembler(
        device=torch.device("cpu"),
        hidden_size=2,
        max_num_tokens=4,
        enable_lookahead=False,
        prompt_stage=stage,
    )

    try:
        assembler.stage_prompt_window(
            _view(0, 2, 4),
            torch.zeros((2, 2), dtype=torch.bfloat16),
            torch.zeros((1, 2), dtype=torch.bfloat16),
        )
    except RuntimeError as exc:
        assert "stage position mismatch" in str(exc)
    else:
        raise AssertionError("missing stage-position failure")


def test_lookahead_uses_req_state_slot_and_rejects_missing_row() -> None:
    stage = _PromptStage(torch.arange(1, 9, dtype=torch.bfloat16), 2)
    assembler = LwdCloudInputAssembler(
        device=torch.device("cpu"),
        hidden_size=2,
        max_num_tokens=4,
        enable_lookahead=True,
        prompt_stage=stage,
    )
    next_prompt_embeds = torch.full((3, 2), 99, dtype=torch.bfloat16)
    view = _view(0, 2, 4)._replace(idx_mapping_np=np.array([2]))

    assembler.stage_prompt_window(
        view,
        torch.zeros((2, 2), dtype=torch.bfloat16),
        next_prompt_embeds,
    )

    assert next_prompt_embeds[2].tolist() == [5.0, 6.0]

    assembler.stage_prompt_window(
        _view(4, 1, 4)._replace(idx_mapping_np=np.array([2])),
        torch.zeros((1, 2), dtype=torch.bfloat16),
        next_prompt_embeds,
    )
    assert next_prompt_embeds[2].tolist() == [0.0, 0.0]

    stage.recv = 2
    try:
        assembler.stage_prompt_window(
            _view(2, 1, 4),
            torch.zeros((1, 2), dtype=torch.bfloat16),
            next_prompt_embeds,
        )
    except RuntimeError as exc:
        assert "missing prompt lookahead" in str(exc)
    else:
        raise AssertionError("missing lookahead failure")


def test_align_consumed_does_not_skip_past_prompt() -> None:
    stage = _PromptStage(torch.arange(1, 9, dtype=torch.bfloat16), 2)
    stage.consumed = 4
    assembler = LwdCloudInputAssembler(
        device=torch.device("cpu"),
        hidden_size=2,
        max_num_tokens=4,
        enable_lookahead=False,
        prompt_stage=stage,
    )

    assembler.align_consumed(_view(6, 1, 4))

    assert stage.consumed == 4
