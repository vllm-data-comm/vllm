# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frontend attach endpoints are declared in the topology yaml.

There is no CLI form: ``--api-server-rpc-port`` / ``--api-server-attach``
are gone. ``edges[].api.rpc_port`` names the engine primary, ``api.attach``
sends that instance's frontend to the primary, and omitting the block keeps
an independent engine.
"""

from pathlib import Path

import pytest
import yaml
from lwd_config_under_test import (
    LwdAPIConfig,
    LwdConfig,
    LwdTopology,
    resolve_lwd_config,
)
from sample_topologies import host_cluster_payload, shared_payload

_ADDR = "10.0.0.1"
_ENDPOINT = f"{_ADDR}:8100"


def _write(tmp_path: Path, api: dict[int, dict[str, str]] | None) -> Path:
    path = tmp_path / "topology.yaml"
    path.write_text(yaml.safe_dump(host_cluster_payload(api)), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "value",
    [
        "25100",
        "10.0.0.1:0",
        "10.0.0.1:65536",
        "hostname:8100",
        "0.0.0.0:8100",
        "224.0.0.1:8100",
        "::1:8100",
        "",
        25100,
    ],
)
def test_api_rejects_invalid_or_ambiguous_addresses(value: str | int) -> None:
    with pytest.raises(ValueError):
        LwdTopology.from_dict(host_cluster_payload({0: {"rpc_port": value}}))


def test_api_needs_exactly_one_endpoint() -> None:
    for block in ({}, {"rpc_port": _ENDPOINT, "attach": _ENDPOINT}):
        with pytest.raises(ValueError, match="exactly one"):
            LwdTopology.from_dict(host_cluster_payload({0: block}))


def test_api_is_edge_only() -> None:
    payload = host_cluster_payload({0: {"rpc_port": _ENDPOINT}})
    payload["clouds"][0]["api"] = {"rpc_port": "10.1.0.1:8100"}
    with pytest.raises(ValueError, match="edge instances only"):
        LwdTopology.from_dict(payload)


def test_rpc_port_must_use_its_own_address() -> None:
    payload = host_cluster_payload({0: {"rpc_port": "10.9.9.9:8100"}})
    with pytest.raises(ValueError, match="own address"):
        LwdTopology.from_dict(payload)


def test_rpc_port_owner_is_unique() -> None:
    with pytest.raises(ValueError, match="both edge"):
        LwdTopology.from_dict(
            host_cluster_payload(
                {0: {"rpc_port": _ENDPOINT}, 2: {"rpc_port": _ENDPOINT}}
            )
        )


def test_attach_must_resolve_to_a_declared_rpc_port() -> None:
    with pytest.raises(ValueError, match="no edge declares"):
        LwdTopology.from_dict(host_cluster_payload({1: {"attach": _ENDPOINT}}))


def test_attach_endpoint_shape_is_checked() -> None:
    with pytest.raises(ValueError):
        LwdTopology.from_dict(
            host_cluster_payload({0: {"rpc_port": _ENDPOINT}, 1: {"attach": "bad"}})
        )


def test_api_is_projected_into_the_serve_view(tmp_path: Path) -> None:
    path = _write(tmp_path, {0: {"rpc_port": _ENDPOINT}, 1: {"attach": _ENDPOINT}})

    primary = LwdConfig.from_dict({"path": str(path), "role": "edge", "instance_id": 0})
    assert primary.api.rpc_port == _ENDPOINT
    assert primary.api.attach is None
    assert primary.api.primary == 0

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


def test_no_api_block_means_independent_engines(tmp_path: Path) -> None:
    path = _write(tmp_path, None)
    topology = LwdTopology.from_file(str(path))
    assert topology.engine_members("edge", 0) == (0,)
    assert topology.engine_members("edge", 1) == (1,)
    assert topology.engine_members("edge", 2) == (2,)
    for identity in range(3):
        config = LwdConfig.from_topology(str(path), "edge", identity, topology)
        assert not config.api.requested
        assert len(config.connections) == 2


def test_frontend_attaches_to_the_endpoint_owner() -> None:
    topology = LwdTopology.from_dict(shared_payload())
    assert topology.rpc_port_owner(_ENDPOINT) == 0
    assert topology.rpc_port_owner("10.0.0.1:9999") is None
    assert topology.engine_primary("edge", 1) == 0
    assert topology.engine_members("edge", 0) == (0, 1)
    assert topology.engine_members("edge", 1) == (0, 1)
    assert topology.engine_members("edge", 2) == (2,)
    assert topology.engine_members("cloud", 0) == (0,)


def test_shared_engine_owns_every_member_connection(tmp_path: Path) -> None:
    path = _write(tmp_path, {0: {"rpc_port": _ENDPOINT}, 1: {"attach": _ENDPOINT}})
    topology = LwdTopology.from_file(str(path))
    shared = LwdConfig.from_topology(str(path), "edge", 0, topology)
    frontend = LwdConfig.from_topology(str(path), "edge", 1, topology)
    standalone = LwdConfig.from_topology(str(path), "edge", 2, topology)

    assert len(shared.connections) == 4
    assert len(frontend.connections) == 4
    assert len(standalone.connections) == 2
    assert {c.connection_id.edge_id for c in shared.connections} == {0, 1}
    assert {c.connection_id.edge_id for c in standalone.connections} == {2}


def test_frontend_digest_is_stable_across_api_blocks() -> None:
    with_api = LwdTopology.from_dict(shared_payload())
    without_api = LwdTopology.from_dict(host_cluster_payload())
    assert with_api.frontend_digest == without_api.frontend_digest
    assert with_api.digest != without_api.digest


def test_resolve_lwd_config_without_topology_has_no_api() -> None:
    assert resolve_lwd_config({}) is None
    assert resolve_lwd_config({"other": 1}) is None


def test_api_config_requires_an_edge_instance(tmp_path: Path) -> None:
    path = _write(tmp_path, {0: {"rpc_port": _ENDPOINT}, 1: {"attach": _ENDPOINT}})
    api = LwdAPIConfig(rpc_port=_ENDPOINT)
    with pytest.raises(ValueError, match="edge instances only"):
        api.validate_topology(
            LwdConfig.from_dict({"path": str(path), "role": "cloud", "instance_id": 0})
        )
