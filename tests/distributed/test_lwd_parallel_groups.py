# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import pytest

from vllm.config.lwd import LwdTopology
from vllm.distributed.lwd import LwdChannelEndpoints, LwdParallelGroupsSpec


def _instance(identity: int, ranks: list[list[int]], cloud: bool) -> dict[str, Any]:
    return {
        "id": identity,
        "dp": [
            {
                "dp_idx": index,
                "addr": f"10.{int(cloud)}.0.{identity + 1}",
                "ranks": group,
                **({"ctrl_port": 5550 + index} if cloud else {}),
            }
            for index, group in enumerate(ranks)
        ],
    }


def _topology(
    scene: str,
    edges: list[list[list[int]]],
    clouds: list[list[list[int]]],
    links: list[tuple[int, int]],
) -> LwdTopology:
    world_size = len(
        {rank for dps in (*edges, *clouds) for ranks in dps for rank in ranks}
    )
    return LwdTopology.from_dict(
        {
            "deployment": {
                "mode": 0,
                "scene": scene,
                "hccl_world_size": world_size,
                "edges_num": len(edges),
                "clouds_num": len(clouds),
            },
            "feature_ctrl": {"enable_early_recv": False, "enable_scramble": False},
            "edges": [_instance(i, dps, cloud=False) for i, dps in enumerate(edges)],
            "clouds": [_instance(i, dps, cloud=True) for i, dps in enumerate(clouds)],
            "instance_links": [{"edge": e, "cloud": c} for e, c in links],
        }
    )


def test_single_instance_groups_each_side_and_pairs_the_first_ranks() -> None:
    spec = LwdParallelGroupsSpec.from_topology(
        _topology("single_instance", [[[0, 1]]], [[[2, 3, 4, 5]]], [(0, 0)])
    )

    assert spec.world_size == 6
    assert spec.tp_groups == ((0, 1), (2, 3, 4, 5))
    assert spec.singleton_groups == ((0,), (1,), (2,), (3,), (4,), (5,))
    assert spec.channel_endpoints == (LwdChannelEndpoints(0, 2),)
    assert spec.tp_group_of(4) == (2, 3, 4, 5)
    assert spec.channel_endpoints_of(2) == LwdChannelEndpoints(0, 2)
    assert spec.channel_endpoints_of(1) is None


def test_dps_sharing_edge_ranks_are_rejected() -> None:
    topology = _topology("single_instance", [[[0], [0]]], [[[1, 2], [3, 4]]], [(0, 0)])

    with pytest.raises(NotImplementedError, match="sharing ranks"):
        LwdParallelGroupsSpec.from_topology(topology)


def test_ranks_holding_several_connections_are_rejected() -> None:
    topology = _topology(
        "lwd_cluster",
        [[[0]], [[1]]],
        [[[2]], [[3]]],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    )

    with pytest.raises(NotImplementedError, match=r"ranks \[0, 1, 2, 3\]"):
        LwdParallelGroupsSpec.from_topology(topology)


def test_rank_outside_the_world_has_no_tp_group() -> None:
    spec = LwdParallelGroupsSpec.from_topology(
        _topology("single_instance", [[[0]]], [[[1]]], [(0, 0)])
    )

    with pytest.raises(ValueError, match="not in"):
        spec.tp_group_of(2)
