# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Edge engine sharing declared by the topology ``api`` endpoints.

An edge that declares ``api.rpc_port`` accepts the frontends of every edge whose
``api.attach`` equals that endpoint; the primary engine owns the connections of
all of them.
"""

import pytest
from lwd_config_under_test import LwdConfig, LwdTopology

_ADDR = "10.0.0.1"
_ENDPOINT = f"{_ADDR}:8100"


def host_cluster_payload(api: dict[int, dict] | None = None) -> dict:
    edges = []
    for identity in range(3):
        edge: dict = {
            "id": identity,
            "dp": [{"dp_idx": 0, "addr": _ADDR}],
            "coordination": {"enabled": True, "placement_only": True},
        }
        if api and identity in api:
            edge["api"] = api[identity]
        edges.append(edge)
    clouds = [
        {
            "id": identity,
            "dp": [
                {
                    "dp_idx": 0,
                    "addr": f"10.1.0.{identity + 1}",
                    "ctrl_port": 5558,
                    "data_port": 7000,
                    "coord_port": 8100 + identity,
                    "coord_url": f"http://10.1.0.{identity + 1}:{8100 + identity}",
                }
            ],
            "coordination": {"enabled": True, "listen_host": "0.0.0.0"},
        }
        for identity in range(2)
    ]
    return {
        "deployment": {
            "mode": 0,
            "scene": "lwd_cluster",
            "edges_num": 3,
            "clouds_num": 2,
            "data_plane_comm": "host",
            "data_plane_host": {"down_buffer_size": 32},
        },
        "feature_ctrl": {"enable_early_recv": False},
        "edges": edges,
        "clouds": clouds,
        "instance_links": [
            {"edge": edge, "cloud": cloud} for edge in range(3) for cloud in range(2)
        ],
    }


def shared_payload() -> dict:
    return host_cluster_payload({0: {"rpc_port": _ENDPOINT}, 1: {"attach": _ENDPOINT}})


def test_frontend_attaches_to_the_endpoint_owner() -> None:
    topology = LwdTopology.from_dict(shared_payload())

    assert topology.engine_primary("edge", 1) == 0
    assert topology.engine_members("edge", 0) == (0, 1)
    assert topology.engine_members("edge", 1) == (0, 1)
    assert topology.engine_members("edge", 2) == (2,)
    assert topology.engine_members("cloud", 0) == (0,)


def test_shared_engine_owns_every_member_connection() -> None:
    topology = LwdTopology.from_dict(shared_payload())
    shared = LwdConfig.from_topology("topology.yaml", "edge", 0, topology)
    frontend = LwdConfig.from_topology("topology.yaml", "edge", 1, topology)
    standalone = LwdConfig.from_topology("topology.yaml", "edge", 2, topology)

    # Two edges x two clouds for the shared engine, two links for the lone edge.
    assert len(shared.connections) == 4
    assert len(frontend.connections) == 4
    assert len(standalone.connections) == 2
    assert {c.connection_id.edge_id for c in shared.connections} == {0, 1}
    assert {c.connection_id.edge_id for c in standalone.connections} == {2}


def test_rpc_port_must_use_the_instance_address() -> None:
    payload = shared_payload()
    payload["edges"][0]["api"]["rpc_port"] = "10.0.0.2:8100"

    with pytest.raises(ValueError, match="must use its own address"):
        LwdTopology.from_dict(payload)


def test_api_needs_exactly_one_endpoint() -> None:
    payload = host_cluster_payload(
        {0: {"rpc_port": _ENDPOINT}, 1: {"rpc_port": _ENDPOINT, "attach": _ENDPOINT}}
    )

    with pytest.raises(ValueError, match="exactly one of rpc_port or attach"):
        LwdTopology.from_dict(payload)


def test_attach_endpoint_must_be_declared_by_a_primary() -> None:
    payload = host_cluster_payload(
        {0: {"rpc_port": _ENDPOINT}, 1: {"attach": "10.0.0.9:8100"}}
    )

    with pytest.raises(ValueError, match="which no edge declares as api.rpc_port"):
        LwdTopology.from_dict(payload)


def test_cloud_cannot_declare_api() -> None:
    payload = shared_payload()
    payload["clouds"][0]["api"] = {"rpc_port": "10.1.0.1:8100"}

    with pytest.raises(ValueError, match="applies to edge instances only"):
        LwdTopology.from_dict(payload)


def test_device_independent_engines_cannot_declare_api() -> None:
    payload = shared_payload()
    payload["deployment"]["data_plane_comm"] = "device"
    payload["deployment"]["hccl_world_size"] = 5
    payload["deployment"]["data_plane_host"] = None
    for identity, edge in enumerate(payload["edges"]):
        edge["dp"][0]["ranks"] = [identity]
    for index, cloud in enumerate(payload["clouds"]):
        cloud["dp"][0]["ranks"] = [3 + index]

    with pytest.raises(ValueError, match="edge_share"):
        LwdTopology.from_dict(payload)


def test_edge_share_declares_the_frontend_endpoints() -> None:
    payload = shared_payload()
    payload["deployment"]["scene"] = "edge_share"
    payload["deployment"]["edges_num"] = 2
    payload["edges"] = payload["edges"][:2]
    payload["instance_links"] = [{"edge": 0, "cloud": 0}, {"edge": 1, "cloud": 1}]

    topology = LwdTopology.from_dict(payload)

    assert topology.engine_primary("edge", 1) == 0
    assert topology.rpc_port_owner(_ENDPOINT) == 0


def test_yaml_key_without_space_after_the_colon_is_explained() -> None:
    payload = host_cluster_payload({0: {'rpc_port:"10.0.0.1:8100"': None}})

    with pytest.raises(ValueError, match="no space after the colon"):
        LwdTopology.from_dict(payload)
