# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.worker.gpu.sample.penalties import PenaltiesState


class LwdCloudPenaltiesState(PenaltiesState):
    def apply_staged_writes(self) -> None:
        super().apply_staged_writes()
        self.prompt_bin_mask.zero_()
