# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from vllm.v1.worker.gpu.lwd.pinned_staging import LwdPinnedStaging
from vllm.v1.worker.gpu.lwd.token_rank import LwdTokenRankEncoder


class LwdDownBatch(NamedTuple):
    sample_hidden_states: torch.Tensor
    logits: torch.Tensor
    sampled_token_ids: torch.Tensor
    req_ids: list[str]
    valid_rows: np.ndarray
    num_draft_tokens_per_req: np.ndarray | None = None


class LwdDownPayload(NamedTuple):
    hidden: torch.Tensor
    meta_dev: torch.Tensor
    seg_lens_host: list[int] | None
    ready_event: torch.cuda.Event
    req_ids: list[str]


class LwdDownSelection(NamedTuple):
    hidden_packet: torch.Tensor
    ranks: list[torch.Tensor]
    counts_dev: torch.Tensor | None
    seg_lens_host: list[int] | None


@dataclass
class LwdDownPayloadBuilder:
    device: torch.device

    def __post_init__(self) -> None:
        self._rank_encoder = LwdTokenRankEncoder()
        self._index_staging = LwdPinnedStaging(
            dtype=torch.int64,
            min_capacity=1024,
            device=self.device,
        )

    def build(self, batch: LwdDownBatch) -> LwdDownPayload | None:
        sampled = batch.sampled_token_ids
        if sampled.dim() != 2:
            return None
        valid = batch.valid_rows[: len(batch.req_ids)].astype(bool)
        if batch.num_draft_tokens_per_req is None:
            selection = self._pack_plain_rows(batch, valid)
        else:
            selection = self._pack_spec_rows(batch, valid)
        if selection is None:
            return None
        hidden_packet, ranks_list, counts_dev, seg_lens_host = selection
        if seg_lens_host is not None:
            assert counts_dev is not None
            meta_dev = torch.cat(ranks_list + [counts_dev])
        else:
            n_req = int(valid.sum())
            counts_dev = torch.ones(n_req, dtype=torch.int32, device=self.device)
            meta_dev = torch.cat(ranks_list + [counts_dev, counts_dev])
        ready_event = torch.cuda.Event()
        ready_event.record()
        req_ids = [req_id for req_id, keep in zip(batch.req_ids, valid) if keep]
        return LwdDownPayload(
            hidden_packet,
            meta_dev,
            seg_lens_host,
            ready_event,
            req_ids,
        )

    def _pack_plain_rows(
        self, batch: LwdDownBatch, valid: np.ndarray
    ) -> LwdDownSelection | None:
        idx = np.nonzero(valid)[0].tolist()
        if not idx:
            return None
        full_cover = bool(valid.all())
        logits = batch.logits if full_cover else batch.logits[idx]
        sampled = batch.sampled_token_ids
        sampled = sampled[:, 0] if full_cover else sampled[idx][:, 0]
        ranks = [self._rank_encoder.encode(logits, sampled)]
        hidden_packet = torch.cat([batch.sample_hidden_states[i : i + 1] for i in idx])
        return LwdDownSelection(hidden_packet, ranks, None, None)

    def _pack_spec_rows(
        self, batch: LwdDownBatch, valid: np.ndarray
    ) -> LwdDownSelection | None:
        assert batch.num_draft_tokens_per_req is not None
        seg_lens_np = batch.num_draft_tokens_per_req.astype(np.int32) + 1
        seg_starts = np.cumsum(seg_lens_np) - seg_lens_np
        total_rows = batch.sample_hidden_states.shape[0]
        rows_per_req = np.minimum(seg_lens_np, np.maximum(total_rows - seg_starts, 0))
        valid_indices = np.nonzero(valid & (rows_per_req >= 1))[0].astype(np.int64)
        if valid_indices.size == 0:
            return None
        valid_lens = rows_per_req[valid_indices].astype(np.int64)
        starts = seg_starts[valid_indices].astype(np.int64)
        n_rows = int(valid_lens.sum())
        req_idx_np = np.repeat(valid_indices, valid_lens)
        pos_idx_np = np.arange(n_rows, dtype=np.int64) - np.repeat(
            np.cumsum(valid_lens) - valid_lens, valid_lens
        )
        row_sel_np = np.repeat(starts, valid_lens) + pos_idx_np
        indices = np.concatenate(
            [req_idx_np, pos_idx_np, row_sel_np, valid_indices, valid_lens]
        )
        buf = self._index_staging.stage_to_device(indices, indices.size)
        n_valid = valid_indices.size
        req_d, pos_d = buf[:n_rows], buf[n_rows : 2 * n_rows]
        sel_d = buf[2 * n_rows : 3 * n_rows]
        vidx_d = buf[3 * n_rows : 3 * n_rows + n_valid]
        vlen_d = buf[3 * n_rows + n_valid :]
        sampled = batch.sampled_token_ids
        sampled_rows = sampled[req_d, pos_d]
        if n_rows == total_rows:
            logits = batch.logits[:n_rows]
            hidden_packet = batch.sample_hidden_states[:n_rows]
        else:
            logits = batch.logits[sel_d]
            hidden_packet = batch.sample_hidden_states[sel_d]
        ranks = [self._rank_encoder.encode(logits, sampled_rows)]
        counts_all = (sampled != -1).sum(dim=1)
        counts_dev = torch.minimum(counts_all[vidx_d], vlen_d).to(torch.int32)
        return LwdDownSelection(
            hidden_packet,
            ranks,
            counts_dev,
            valid_lens.tolist(),
        )
