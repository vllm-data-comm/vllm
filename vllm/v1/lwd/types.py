# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch and result metadata shared by the LWD control and data planes."""

import enum
from dataclasses import dataclass

import msgspec

from vllm.config.lwd.topology import LwdConnectionKey


class LwdBatchType(enum.Enum):
    LWD_EMBED = "lwd_embed"
    LWD_UNEMBED = "lwd_unembed"


# Tagged payloads let the executor decode the union without guessing its type.
class LwdEmbedBatch(msgspec.Struct, tag=True):  # type: ignore[call-arg]
    req_ids: list[str] = msgspec.field(default_factory=list)
    token_ids: list[list[int]] = msgspec.field(default_factory=list)
    # Absolute prompt offsets survive chunking and prefix-cache hits.
    prompt_offsets: list[int] = msgspec.field(default_factory=list)
    has_mrope: list[bool] = msgspec.field(default_factory=list)
    mrope_positions: list[bytes] = msgspec.field(default_factory=list)


class LwdUnembedBatch(msgspec.Struct, tag=True):  # type: ignore[call-arg]
    req_ids: list[str] = msgspec.field(default_factory=list)
    num_accept_tokens: list[int] = msgspec.field(default_factory=list)
    recv_num_elements: int = 0
    out_token_idxs: list[list[int]] = msgspec.field(default_factory=list)
    top_id_ths: list[list[int]] = msgspec.field(default_factory=list)
    token_ids: list[list[int]] = msgspec.field(default_factory=list)


@dataclass
class LwdBatch:
    batch_type: LwdBatchType
    seqno: int
    batch_meta: LwdEmbedBatch | LwdUnembedBatch | None
    connection_key: LwdConnectionKey | None = None


@dataclass
class LwdC2eMeta:
    # Receive sizing comes from the actual tensor, not draft acceptance counts.
    hidden_num_elements: int
    top_id_ths: list[list[int]]
    num_accepted_tokens: list[int]
    req_ids: list[str]
    down_seqno: int = -1
    connection_key: LwdConnectionKey | None = None
