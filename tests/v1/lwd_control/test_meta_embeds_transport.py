# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Transport gate for the meta placeholder prompt embeds.

The LWD cloud attaches prompt embeds as meta tensors (shape only, no
storage), so the placeholder rides the multiproc MQ inline pickle path
instead of the overflow slow channel. These tests are the go/no-go gate
for that design: they pin how a meta tensor behaves under PEP-574 pickle
(the exact serialization MessageQueue.enqueue uses) and under roundtrip.

Runs anywhere torch is importable; skipped on the lightweight env and
re-run on the NPU env (python>=3.10 + torch).
"""

from __future__ import annotations

import pickle

import pytest

torch = pytest.importorskip(
    "torch", reason="re-run on NPU env: python>=3.10 + torch"
)

# Match the placeholder geometry of a ~35MB production buffer:
# 3456 prompt tokens x 5120 hidden, bf16.
NUM_TOKENS, HIDDEN = 3456, 5120


def _make_placeholder() -> "torch.Tensor":
    return torch.empty(
        NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="meta"
    )


def _pickle_with_oob(obj: object) -> tuple[bytes, list[bytes]]:
    """Pickle exactly like shm_broadcast.MessageQueue.enqueue does:
    PEP-574 protocol 5 with a buffer_callback collecting out-of-band
    storage segments."""
    oob: list[bytes] = []

    def callback(buf: "pickle.PickleBuffer") -> bool:
        raw = buf.raw()
        oob.append(raw)
        return False

    main = pickle.dumps(obj, protocol=5, buffer_callback=callback)
    return main, oob


def test_meta_placeholder_has_no_oob_storage() -> None:
    """A meta tensor must produce no >=1MiB out-of-band segment: with an
    empty storage there is nothing for the MQ overflow branch to carry,
    so the message stays on the SHM inline fast path."""
    _, oob = _pickle_with_oob(_make_placeholder())
    for segment in oob:
        assert len(segment) < 1024 * 1024, (
            f"meta tensor leaked a {len(segment)}-byte OOB segment"
        )


def test_meta_placeholder_pickle_overhead_is_tiny() -> None:
    """The meta placeholder must add only header-level bytes to the main
    pickle stream (threshold 1KiB), unlike a materialized zeros buffer
    which adds its full storage."""
    base_main, _ = _pickle_with_oob({"req": "x"})
    placeholder_main, _ = _pickle_with_oob(
        {"req": "x", "embeds": _make_placeholder()}
    )
    overhead = len(placeholder_main) - len(base_main)
    assert overhead < 1024, f"meta placeholder added {overhead} bytes"


def test_meta_placeholder_roundtrip_keeps_metadata() -> None:
    """After pickle roundtrip (engine -> worker hop) the tensor must stay
    a meta tensor with identical shape/dtype: the worker-side wash keys
    off is_meta, so this property is load-bearing."""
    main, oob = _pickle_with_oob(_make_placeholder())
    restored = pickle.loads(main, buffers=oob)
    assert restored.is_meta is True
    assert tuple(restored.shape) == (NUM_TOKENS, HIDDEN)
    assert restored.dtype == torch.bfloat16


def test_meta_nbytes_reports_full_shape_size() -> None:
    """Trap pin: nbytes is derived from numel * element_size even though
    the storage is empty -- a 35MB-scale number. Do not infer transport
    cost (or log sizes) from nbytes on a meta tensor."""
    placeholder = _make_placeholder()
    assert placeholder.nbytes == NUM_TOKENS * HIDDEN * 2
