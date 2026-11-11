# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import NamedTuple, Protocol

import numpy as np
import torch


class LwdCloudBatchView(NamedTuple):
    req_ids: list[str]
    idx_mapping_np: np.ndarray
    num_scheduled_tokens: np.ndarray
    num_computed_tokens_np: np.ndarray
    prefill_len_np: np.ndarray
    query_start_loc_np: np.ndarray


class LwdPromptWindow(NamedTuple):
    req_id: str
    req_index: int
    start: int
    rows: int


class LwdPromptStageRequest(Protocol):
    host: torch.Tensor


class LwdPromptStage(Protocol):
    def take_req(self, req_id: str) -> LwdPromptStageRequest | None: ...

    def recv_rows(self, req_id: str) -> int: ...

    def consumed_rows(self, req_id: str) -> int: ...

    def segment_for(self, req_id: str, n_rows: int) -> tuple[int, int, int]: ...

    def consume(self, req_id: str, n_rows: int) -> None: ...

    def rewind_consumed(self, req_id: str, to_rows: int) -> bool: ...

    def skip_consumed(self, req_id: str, to_rows: int) -> bool: ...


class LwdStagedWindow(NamedTuple):
    device_payload: torch.Tensor
    window_index: int | None
    plan: list[LwdPromptWindow]


class LwdCloudInputAssembler:
    def __init__(
        self,
        *,
        device: torch.device,
        hidden_size: int,
        max_num_tokens: int,
        enable_lookahead: bool,
        prompt_stage: LwdPromptStage | None = None,
    ) -> None:
        self.device = device
        self.hidden_size = hidden_size
        self.max_num_tokens = max_num_tokens
        self.enable_lookahead = enable_lookahead
        self.prompt_stage = prompt_stage
        self._copy_stream: torch.cuda.Stream | None = None
        self._window_device: list[torch.Tensor] | None = None
        self._window_flip = 0
        self._window_used_events: list[torch.cuda.Event | None] = [None, None]

    def bind_prompt_stage(self, prompt_stage: LwdPromptStage | None) -> None:
        self.prompt_stage = prompt_stage

    def align_consumed(self, view: LwdCloudBatchView) -> None:
        if self.prompt_stage is None:
            return
        for req_id, computed in zip(view.req_ids, view.num_computed_tokens_np):
            consumed = self.prompt_stage.consumed_rows(req_id)
            new_computed = int(computed)
            if consumed > new_computed and not self.prompt_stage.rewind_consumed(
                req_id, new_computed
            ):
                raise RuntimeError(f"[LWD] cannot rewind {req_id} to {new_computed}")
            recv_rows = self.prompt_stage.recv_rows(req_id)
            if consumed < new_computed and consumed < recv_rows:
                skip_to = min(new_computed, recv_rows)
                if not self.prompt_stage.skip_consumed(req_id, skip_to):
                    raise RuntimeError(f"[LWD] cannot skip {req_id} to {skip_to}")

    def stage_prompt_window(
        self,
        view: LwdCloudBatchView,
        inputs_embeds: torch.Tensor,
        next_prompt_embeds: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> None:
        del input_ids
        for req_state_idx in view.idx_mapping_np:
            if req_state_idx >= 0:
                next_prompt_embeds[int(req_state_idx)].zero_()
        plan = self._prompt_plan(view)
        if not plan:
            return
        staged = self._harvest_window(plan, next_prompt_embeds, view)
        self._inject_window(staged, view, inputs_embeds)

    def _prompt_plan(self, view: LwdCloudBatchView) -> list[LwdPromptWindow]:
        plan: list[LwdPromptWindow] = []
        for index, req_id in enumerate(view.req_ids):
            if view.idx_mapping_np[index] < 0:
                continue
            start = int(view.num_computed_tokens_np[index])
            prompt_len = int(view.prefill_len_np[index])
            scheduled = int(view.num_scheduled_tokens[index])
            rows = min(scheduled, prompt_len - start)
            if rows > 0:
                plan.append(LwdPromptWindow(req_id, index, start, rows))
        return plan

    def _harvest_window(
        self,
        plan: list[LwdPromptWindow],
        next_prompt_embeds: torch.Tensor,
        view: LwdCloudBatchView,
    ) -> LwdStagedWindow:
        if self.prompt_stage is None:
            raise RuntimeError("[LWD] prompt stage is required for remote embeddings")
        numel = sum(window.rows for window in plan) * self._elements_per_row()
        device_payload, window_index = self._window_buffer(numel)
        self._stage_window(plan, device_payload, next_prompt_embeds, view)
        return LwdStagedWindow(device_payload, window_index, plan)

    def _elements_per_row(self) -> int:
        return self.hidden_size

    def _window_buffer(self, numel: int) -> tuple[torch.Tensor, int]:
        if self._window_device is None:
            capacity = max(numel, self.max_num_tokens * self._elements_per_row())
            self._window_device = [
                torch.empty(capacity, dtype=torch.bfloat16, device=self.device)
                for _ in range(2)
            ]
        if self._window_device[0].numel() < numel:
            self._window_device = [
                torch.empty(numel, dtype=torch.bfloat16, device=self.device)
                for _ in range(2)
            ]
        index = self._window_flip
        self._window_flip ^= 1
        return self._window_device[index][:numel], index

    def _stage_window(
        self,
        plan: list[LwdPromptWindow],
        device_payload: torch.Tensor,
        next_prompt_embeds: torch.Tensor,
        view: LwdCloudBatchView,
    ) -> None:
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(self.device)
        used_event = self._window_used_events[self._window_flip ^ 1]
        with torch.cuda.stream(self._copy_stream):
            if used_event is not None:
                self._copy_stream.wait_event(used_event)
            embed_offset = 0
            for window in plan:
                self._stage_one_request(window, device_payload, embed_offset)
                embed_offset += window.rows * self.hidden_size
            self._stage_window_extras(plan, device_payload, embed_offset)
            self._stage_lookahead(plan, next_prompt_embeds, view)
            ready_event = torch.cuda.Event()
            ready_event.record(self._copy_stream)
        torch.cuda.current_stream().wait_event(ready_event)
        assert self.prompt_stage is not None
        for window in plan:
            self.prompt_stage.consume(window.req_id, window.rows)

    def _stage_one_request(
        self,
        window: LwdPromptWindow,
        device_payload: torch.Tensor,
        embed_offset: int,
    ) -> None:
        assert self.prompt_stage is not None
        consumed = self.prompt_stage.consumed_rows(window.req_id)
        if consumed != window.start:
            raise RuntimeError(
                f"[LWD] stage position mismatch req={window.req_id}: "
                f"consumed={consumed}, window_start={window.start}"
            )
        host_start, numel, _ = self.prompt_stage.segment_for(window.req_id, window.rows)
        if numel != window.rows * self.hidden_size:
            raise RuntimeError(f"[LWD] invalid host window for {window.req_id}")
        request = self.prompt_stage.take_req(window.req_id)
        assert request is not None
        device_payload[embed_offset : embed_offset + numel].copy_(
            request.host[host_start : host_start + numel], non_blocking=True
        )

    def _stage_window_extras(
        self,
        plan: list[LwdPromptWindow],
        device_payload: torch.Tensor,
        embed_offset: int,
    ) -> None:
        del plan, device_payload, embed_offset

    def _stage_lookahead(
        self,
        plan: list[LwdPromptWindow],
        next_prompt_embeds: torch.Tensor,
        view: LwdCloudBatchView,
    ) -> None:
        if not self.enable_lookahead:
            return
        assert self.prompt_stage is not None
        for window in plan:
            req_state_idx = int(view.idx_mapping_np[window.req_index])
            next_row = window.start + window.rows
            if next_row >= int(view.prefill_len_np[window.req_index]):
                continue
            if next_row >= int(self.prompt_stage.recv_rows(window.req_id)):
                raise RuntimeError(
                    f"[LWD] missing prompt lookahead for {window.req_id} "
                    f"at row {next_row}"
                )
            request = self.prompt_stage.take_req(window.req_id)
            assert request is not None
            host_start = next_row * self.hidden_size
            next_prompt_embeds[req_state_idx].copy_(
                request.host[host_start : host_start + self.hidden_size],
                non_blocking=True,
            )

    def _inject_window(
        self,
        staged: LwdStagedWindow,
        view: LwdCloudBatchView,
        inputs_embeds: torch.Tensor,
    ) -> None:
        total_rows = sum(window.rows for window in staged.plan)
        embeds = staged.device_payload[: total_rows * self.hidden_size].view(
            -1, self.hidden_size
        )
        row = 0
        for window in staged.plan:
            output = int(view.query_start_loc_np[window.req_index])
            inputs_embeds[output : output + window.rows].copy_(
                embeds[row : row + window.rows], non_blocking=True
            )
            row += window.rows
        if staged.window_index is not None:
            used_event = torch.cuda.Event()
            used_event.record()
            self._window_used_events[staged.window_index] = used_event
        else:
            staged.device_payload.record_stream(torch.cuda.current_stream())
        self._inject_window_extras(staged, view)

    def _inject_window_extras(
        self,
        staged: LwdStagedWindow,
        view: LwdCloudBatchView,
    ) -> None:
        del staged, view
