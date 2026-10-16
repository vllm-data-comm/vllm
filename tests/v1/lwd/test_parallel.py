# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check PR #33's rank layout and local runtime world without NPU hardware."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.distributed import parallel_state as ps
from vllm.distributed.lwd import get_lwd_distributed_args


def config_for(role: str, edge: int = 1, cloud: int = 8) -> SimpleNamespace:
    link = SimpleNamespace(
        edge=SimpleNamespace(addr="76.76.26.18", ranks=tuple(range(edge))),
        cloud=SimpleNamespace(ranks=tuple(range(edge, edge + cloud))),
    )
    return SimpleNamespace(
        lwd_config=SimpleNamespace(
            role=role,
            connections=(link,),
            topology=SimpleNamespace(
                deployment=SimpleNamespace(
                    scene="single_instance",
                    hccl_world_size=edge + cloud,
                )
            ),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=edge if role == "edge" else cloud,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            nnodes=1,
            enable_elastic_ep=False,
            use_all2all=False,
            enable_eplb=False,
        ),
        engram_config=None,
        model_config=SimpleNamespace(is_moe=False),
    )


@pytest.mark.parametrize(
    "role,rank,global_rank", [("edge", 0, 0), ("cloud", 0, 1), ("cloud", 7, 8)]
)
def test_both_sides_share_bootstrap_address_but_keep_local_device_rank(
    role: str, rank: int, global_rank: int
) -> None:
    assert get_lwd_distributed_args(config_for(role), rank) == (
        9,
        global_rank,
        "tcp://76.76.26.18:29600",
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("tensor_parallel_size", 2),
        ("pipeline_parallel_size", 2),
        ("data_parallel_size", 2),
    ],
)
def test_mismatched_parallel_layout_fails_before_rendezvous(
    field: str, value: int
) -> None:
    config = config_for("edge")
    setattr(config.parallel_config, field, value)
    with pytest.raises(ValueError, match="no-PP requires"):
        get_lwd_distributed_args(config, 0)


@pytest.mark.parametrize("rank", [0, 1, 2, 5])
def test_asymmetric_tp_and_singleton_pp_preserve_bootstrap_for_channels(
    monkeypatch: pytest.MonkeyPatch, rank: int
) -> None:
    config = config_for("edge" if rank < 2 else "cloud", 2, 4)
    bootstrap = SimpleNamespace(
        world_size=6, local_rank=rank if rank < 2 else rank - 2, ranks=list(range(6))
    )
    monkeypatch.setattr(ps, "_WORLD", bootstrap)
    for name in (
        "_TP",
        "_ETP",
        "_PP",
        "_DP",
        "_PCP",
        "_DCP",
        "_EP",
        "_EPLB",
        "_LWD_BOOTSTRAP_WORLD",
        "_NODE_COUNT",
    ):
        monkeypatch.setattr(ps, name, None)
    calls = {}

    def create(
        groups: list[list[int]], local_rank: int, backend: str, **kwargs: object
    ) -> SimpleNamespace:
        calls[kwargs["group_name"]] = groups
        ranks = next(group for group in groups if rank in group)
        return SimpleNamespace(ranks=ranks, cpu_group=object(), destroy=Mock())

    monkeypatch.setattr(ps, "init_model_parallel_group", create)
    monkeypatch.setattr(ps, "_node_count", lambda group: 1)
    ps._initialize_lwd_model_parallel(config, "gloo")
    assert calls["tp"] == [[0, 1], [2, 3, 4, 5]]
    assert all(
        calls[name] == [[i] for i in range(6)] for name in ("pp", "dp", "pcp", "dcp")
    )
    assert ps.get_lwd_bootstrap_world() is bootstrap
    assert ps.get_world_group() is ps.get_tp_group()
    tp = ps.get_tp_group()
    ps.destroy_model_parallel()
    tp.destroy.assert_called_once()
    assert ps._WORLD is None
