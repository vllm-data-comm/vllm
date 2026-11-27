# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The attach endpoints live in the topology for both data planes."""

import copy
from pathlib import Path

import pytest
import yaml
from lwd_config_under_test import LwdConfig, LwdTopology, resolve_lwd_config
from sample_topologies import PAYLOADS
from test_engine_sharing import host_cluster_payload

_ADDR = "10.0.0.1"
_ENDPOINT = f"{_ADDR}:8100"


def _write(tmp_path: Path, api: dict[int, dict]) -> Path:
    path = tmp_path / "topology.yaml"
    path.write_text(yaml.safe_dump(host_cluster_payload(api)), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "value",
    [
        "25100",
        "10.0.0.1:0",
        "10.0.0.1:65536",
        "hostname:25100",
        "0.0.0.0:25100",
        "224.0.0.1:25100",
        "::1:25100",
        "",
        25100,
    ],
)
def test_api_rejects_invalid_or_ambiguous_addresses(value: str | int) -> None:
    with pytest.raises(ValueError):
        LwdTopology.from_dict(host_cluster_payload({0: {"rpc_port": value}}))


def test_api_is_projected_into_the_serve_view(tmp_path: Path) -> None:
    path = _write(tmp_path, {0: {"rpc_port": _ENDPOINT}, 1: {"attach": _ENDPOINT}})

    primary = LwdConfig.from_dict({"path": str(path), "role": "edge", "instance_id": 0})
    assert primary.api.rpc_port == _ENDPOINT
    assert primary.api.attach is None and primary.api.primary is None

    frontend = LwdConfig.from_dict(
        {"path": str(path), "role": "edge", "instance_id": 1}
    )
    assert frontend.api.attach == _ENDPOINT
    assert frontend.api.rpc_port is None
    assert frontend.api.primary == 0

    standalone = LwdConfig.from_dict(
        {"path": str(path), "role": "edge", "instance_id": 2}
    )
    assert not standalone.api.requested


def test_api_requires_a_shared_engine(tmp_path: Path) -> None:
    path = _write(tmp_path, {2: {"rpc_port": _ENDPOINT}})

    with pytest.raises(ValueError, match="shared by several instances"):
        LwdConfig.from_dict({"path": str(path), "role": "edge", "instance_id": 2})


def test_resolve_lwd_config_without_topology_has_no_api() -> None:
    assert resolve_lwd_config({}) is None


def test_device_edge_share_projects_the_same_serve_view(tmp_path: Path) -> None:
    payload = copy.deepcopy(PAYLOADS["s2_edge_share_1dp.yaml"])
    payload["edges"][0]["api"] = {"rpc_port": _ENDPOINT}
    payload["edges"][1]["api"] = {"attach": _ENDPOINT}
    path = tmp_path / "topology.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    primary = LwdConfig.from_dict({"path": str(path), "role": "edge", "instance_id": 0})
    assert primary.api.rpc_port == _ENDPOINT

    member = LwdConfig.from_dict({"path": str(path), "role": "edge", "instance_id": 1})
    assert member.api.attach == _ENDPOINT
    assert member.api.primary == 0


def _device_payload() -> dict:
    edges = [
        {
            "id": identity,
            "dp": [{"dp_idx": 0, "addr": "10.0.0.1", "ranks": [identity]}],
            "coordination": {"enabled": True, "placement_only": True},
        }
        for identity in range(3)
    ]
    clouds = [
        {
            "id": identity,
            "dp": [
                {
                    "dp_idx": 0,
                    "addr": f"10.1.0.{identity + 1}",
                    "ctrl_port": 5558,
                    "ranks": [3 + 4 * identity + offset for offset in range(4)],
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
            "data_plane_comm": "device",
            "hccl_world_size": 11,
        },
        "feature_ctrl": {"enable_early_recv": False},
        "edges": edges,
        "clouds": clouds,
        "instance_links": [
            {"edge": edge, "cloud": cloud} for edge in range(3) for cloud in range(2)
        ],
    }


def test_device_independent_engines_cannot_declare_api() -> None:
    payload = _device_payload()
    payload["edges"][0]["api"] = {"rpc_port": _ENDPOINT}

    with pytest.raises(ValueError, match="edge_share"):
        LwdTopology.from_dict(payload)
