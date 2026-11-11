# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


class LwdTokenRankEncoder:
    def __init__(self) -> None:
        self.vocab_ids: torch.Tensor | None = None

    def encode(self, logits: torch.Tensor, selected_ids: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.shape[-1]
        if (
            self.vocab_ids is None
            or self.vocab_ids.numel() != vocab_size
            or self.vocab_ids.device != logits.device
        ):
            self.vocab_ids = torch.arange(vocab_size, device=logits.device)
        selected_ids = selected_ids.long()
        valid = selected_ids != -1
        safe_ids = selected_ids.masked_fill(~valid, 0)
        selected_logits = logits.gather(1, safe_ids.unsqueeze(1))
        preceding = (logits > selected_logits) | (
            (logits == selected_logits) & (self.vocab_ids < safe_ids.unsqueeze(1))
        )
        ranks = preceding.sum(dim=1, dtype=torch.int32)
        return ranks.masked_fill(~valid, 0)
