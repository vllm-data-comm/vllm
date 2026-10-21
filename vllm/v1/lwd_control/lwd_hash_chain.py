# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-hash chain for the embeds-only cloud request (mm parity).

The PR #3 review dropped the hash chain from the cloud engine pending a
separate prefix-caching design, and froze the old identifiers out of that
file. The 2026-10-20 multi-request run proved the gap fatal on its own:
with no hasher attached, the cloud request's hash list stays empty and
the first full block crashes cache_full_blocks with IndexError. This
module restores the chain OUTSIDE the frozen file so the freeze stands:

- first build uses the edge-computed wire chain (hashes over the REAL
  token ids; the cloud's local fallback hashes zero placeholders and can
  never match another request's entries);
- continuation, missing or length-mismatched chains fall back to the
  standard local hasher (fail-open), which also keeps the list long
  enough for the block indexing;
- the cloud's prompt_embeds is a meta tensor (no storage; the migration
  switched from mm's zero buffers to meta to keep 35MB+ payloads off the
  MQ overflow path). The fallback hasher reads embeds data via
  _gen_prompt_embeds_extra_hash_keys, which raises NotImplementedError on
  a meta tensor. The fix: pre-fill the per-block embeds-hash cache for
  every block the fallback is about to hash, so the lazy path hits the
  cache and never touches the meta tensor.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm import VllmConfig
    from vllm.v1.core.kv_cache_utils import KVCacheConfig
    from vllm.v1.request import Request


def lwd_cloud_block_hasher(
    vllm_config: "VllmConfig",
    kv_cache_config: "KVCacheConfig",
    wire_hashes: list[bytes],
) -> Callable[["Request"], list[Any]] | None:
    """Build the per-request hasher closure; None disables hashing."""
    if not vllm_config.cache_config.enable_prefix_caching:
        return None
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.kv_cache_utils import (
        get_request_block_hasher,
        init_none_hash,
        resolve_kv_cache_block_sizes,
    )

    _, hash_block_size = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
    caching_hash_fn = get_hash_fn_by_name(
        vllm_config.cache_config.prefix_caching_hash_algo
    )
    init_none_hash(caching_hash_fn)
    fallback = get_request_block_hasher(hash_block_size, caching_hash_fn)
    wire = list(wire_hashes)

    def _prefill_meta_embeds_cache(request: "Request") -> None:
        """Pre-fill the per-block embeds-hash cache for every block the
        fallback hasher is about to hash.

        The fallback iterates from len(block_hashes) * hash_block_size
        while start + hash_block_size <= num_tokens; mirror that walk and
        seed the cache dict, so the lazy path in
        _gen_prompt_embeds_extra_hash_keys hits the cache and never calls
        .cpu() on the meta tensor. The hash value salts on request_id:
        different requests never share embeds hashes, preventing false
        prefix-cache hits (the meta tensor carries no content to hash).
        """
        start = len(request.block_hashes) * hash_block_size
        while start + hash_block_size <= request.num_tokens:
            key = (start, start + hash_block_size)
            if key not in request._prompt_embeds_per_block_hashes:
                request._prompt_embeds_per_block_hashes[key] = hashlib.sha256(
                    f"lwd-meta:{request.request_id}:{start}".encode()
                ).digest()
            start += hash_block_size

    def hasher(request: "Request") -> list[Any]:
        if len(request.block_hashes) == 0 and request.num_output_tokens == 0:
            if len(wire) == request.num_prompt_tokens // hash_block_size:
                return wire
        _prefill_meta_embeds_cache(request)
        return fallback(request)

    return hasher
