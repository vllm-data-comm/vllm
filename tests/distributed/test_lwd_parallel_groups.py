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
            "feature_ctrl": {
                "enable_early_recv": False,
                "scramble": {
                    "enable": False,
                    "device_id": None,
                    "custom_factor": None,
                },
            },
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
    assert spec.channel_endpoints == (LwdChannelEndpoints(0, 0, 2),)
    assert spec.tp_group_of(4) == (2, 3, 4, 5)
    assert spec.channel_endpoints_of(2) == (LwdChannelEndpoints(0, 0, 2),)
    assert spec.channel_endpoints_of(1) == ()


def test_one_dp_keeps_singleton_dp_groups_and_ep_equal_to_tp() -> None:
    """At one DP the new groups must stay what a centralized pp=1
    deployment has, or every existing deployment changes behaviour."""
    spec = LwdParallelGroupsSpec.from_topology(
        _topology("single_instance", [[[0, 1]]], [[[2, 3, 4, 5]]], [(0, 0)])
    )

    assert spec.dp_groups == spec.singleton_groups
    assert spec.ep_groups == spec.tp_groups


def test_several_cloud_dps_span_the_instance_for_ep_and_zip_for_dp() -> None:
    """The MoE all2all rides the EP group, so it has to cover every cloud
    rank; a group per TP group would route each token to one replica's
    experts only, which produces garbage rather than an error."""
    spec = LwdParallelGroupsSpec.from_topology(
        _topology("single_instance", [[[0], [0]]], [[[1, 2], [3, 4]]], [(0, 0)])
    )

    assert spec.world_size == 5
    assert spec.tp_groups == ((0,), (1, 2), (3, 4))
    # Edge one group; cloud spans both of its DPs.
    assert spec.ep_groups == ((0,), (1, 2, 3, 4))
    # Replicas of one TP position, plus the edge on its own.
    assert spec.dp_groups == ((0,), (1, 3), (2, 4))
    # The shared edge rank holds one channel pair per cloud DP.
    assert spec.channel_endpoints_of(0) == (
        LwdChannelEndpoints(0, 0, 1),
        LwdChannelEndpoints(1, 0, 3),
    )
    assert spec.channel_endpoints_of(3) == (LwdChannelEndpoints(1, 0, 3),)


def test_rank_outside_the_world_has_no_tp_group() -> None:
    spec = LwdParallelGroupsSpec.from_topology(
        _topology("single_instance", [[[0]]], [[[1]]], [(0, 0)])
    )

    with pytest.raises(ValueError, match="not in"):
        spec.tp_group_of(2)
