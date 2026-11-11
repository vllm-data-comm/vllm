# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import numpy as np
import torch


class LwdPinnedStaging:
    def __init__(
        self,
        *,
        dtype: torch.dtype,
        min_capacity: int,
        depth: int = 4,
        device: str | torch.device | None = None,
    ) -> None:
        self._dtype = dtype
        self._min_capacity = min_capacity
        self._depth = depth
        self._device = device
        self._pinned_slots: list[torch.Tensor] | None = None
        self._device_slots: list[torch.Tensor] | None = None
        self._index = 0

    def _ensure(self, n: int) -> None:
        if self._pinned_slots is not None and self._pinned_slots[0].numel() >= n:
            return
        capacity = max(n, self._min_capacity)
        self._pinned_slots = [
            torch.empty(capacity, dtype=self._dtype, pin_memory=True)
            for _ in range(self._depth)
        ]
        self._device_slots = (
            None
            if self._device is None
            else [
                torch.empty(capacity, dtype=self._dtype, device=self._device)
                for _ in range(self._depth)
            ]
        )
        self._index = 0

    def _rotate(self) -> int:
        index = self._index
        self._index = (index + 1) % self._depth
        return index

    def stage_to_device(self, src: Sequence[int] | np.ndarray, n: int) -> torch.Tensor:
        self._ensure(n)
        assert self._pinned_slots is not None
        index = self._rotate()
        stage = self._pinned_slots[index]
        stage.numpy()[:n] = src
        assert self._device_slots is not None
        device = self._device_slots[index]
        device[:n].copy_(stage[:n], non_blocking=True)
        return device[:n]

    def next_pinned(self, n: int) -> torch.Tensor:
        self._ensure(n)
        assert self._pinned_slots is not None
        return self._pinned_slots[self._rotate()]
