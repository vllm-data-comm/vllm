# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import msgspec
import pytest
from lwd_config_under_test import topology


@pytest.fixture
def batch_types(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # Load the real transport types without importing GPU runtime fixtures.
    monkeypatch.setitem(sys.modules, "vllm.config.lwd.topology", topology)
    path = Path(__file__).resolve().parents[3] / "vllm/v1/lwd/types.py"
    spec = importlib.util.spec_from_file_location("lwd_batch_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("is_embed", [True, False])
def test_batch_handoff_preserves_direction_identity_and_row_metadata(
    batch_types: ModuleType, is_embed: bool
) -> None:
    t = batch_types
    meta = (
        t.LwdEmbedBatch(
            req_ids=["r"],
            token_ids=[[5, 9]],
            prompt_offsets=[8192],
            has_mrope=[True],
            mrope_positions=[b"positions"],
        )
        if is_embed
        else t.LwdUnembedBatch(
            req_ids=["r"],
            num_accept_tokens=[0],
            recv_num_elements=128,
            top_id_ths=[[2]],
            out_token_idxs=[[9]],
        )
    )
    original = t.LwdBatch(
        t.LwdBatchType.LWD_EMBED if is_embed else t.LwdBatchType.LWD_UNEMBED,
        17,
        meta,
        topology.LwdConnectionKey(2, 3, 1),
    )
    encoded = msgspec.msgpack.encode(original)
    assert msgspec.msgpack.decode(encoded, type=t.LwdBatch) == original
    malformed = msgspec.msgpack.decode(encoded)
    malformed["batch_meta"]["type"] = "unknown"
    with pytest.raises(msgspec.ValidationError):
        msgspec.msgpack.decode(msgspec.msgpack.encode(malformed), type=t.LwdBatch)


def test_result_handoff_keeps_tensor_size_separate_from_draft_acceptance(
    batch_types: ModuleType,
) -> None:
    meta = batch_types.LwdC2eMeta(
        hidden_num_elements=128,
        top_id_ths=[[2]],
        num_accepted_tokens=[0],
        req_ids=["r"],
        down_seqno=8,
        connection_key=topology.LwdConnectionKey(2, 3, 1),
    )
    assert (
        msgspec.msgpack.decode(
            msgspec.msgpack.encode(meta),
            type=batch_types.LwdC2eMeta,
        )
        == meta
    )
