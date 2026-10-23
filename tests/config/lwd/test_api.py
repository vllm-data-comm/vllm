# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from pathlib import Path

import pytest
from lwd_config_under_test import LwdAPIConfig, LwdConfig, resolve_lwd_config


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
@pytest.mark.parametrize("field", ["api_server_rpc_port", "api_server_attach"])
def test_api_endpoints_reject_invalid_or_ambiguous_addresses(
    value: str | int, field: str, lwd_yaml_dir: Path
) -> None:
    with pytest.raises(ValueError):
        LwdConfig.from_dict(
            {
                "path": str(lwd_yaml_dir / "s2_edge_share_1dp.yaml"),
                "role": "edge",
                "instance_id": 0 if field == "api_server_rpc_port" else 1,
            },
            api=LwdAPIConfig(**{field: value}),
        )


def test_api_modes_are_exclusive_and_allow_bracketed_ipv6(lwd_yaml_dir: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_lwd_config({}, api=LwdAPIConfig("10.0.0.1:25100", "10.0.0.1:25100"))
    config = LwdConfig.from_dict(
        {
            "path": str(lwd_yaml_dir / "s2_edge_share_1dp.yaml"),
            "role": "edge",
            "instance_id": 0,
        },
        api=LwdAPIConfig(api_server_rpc_port="[::1]:25100"),
    )
    assert config.api.api_server_rpc_port == "[::1]:25100"


@pytest.mark.parametrize(
    "role,identity,flag,frontend,match",
    [
        ("cloud", 0, "api_server_rpc_port", {}, "edge instances only"),
        ("edge", 1, "api_server_rpc_port", {}, "instance_id=0"),
        ("edge", 0, "api_server_attach", {}, "non-primary"),
        ("edge", 0, "api_server_rpc_port", {"headless": True}, "HTTP frontend"),
        ("edge", 0, "api_server_rpc_port", {"grpc": True}, "HTTP frontend"),
    ],
)
def test_api_launch_rejects_incompatible_identity_and_frontend(
    role: str,
    identity: int,
    flag: str,
    frontend: dict[str, bool],
    match: str,
    lwd_yaml_dir: Path,
) -> None:
    args = argparse.Namespace(
        **{flag: "10.0.0.1:25100"},
        **frontend,
        additional_config={
            "lwd_config": {
                "path": str(lwd_yaml_dir / "s2_edge_share_1dp.yaml"),
                "role": role,
                "instance_id": identity,
            }
        },
    )
    with pytest.raises(ValueError, match=match):
        resolve_lwd_config(
            args.additional_config, api=LwdAPIConfig.from_namespace(args)
        )


def test_api_requires_edge_share_topology_but_is_optional_for_centralized(
    lwd_yaml_dir: Path,
) -> None:
    LwdAPIConfig().validate_topology(None)
    assert not LwdAPIConfig.from_namespace(argparse.Namespace()).requested
    api = LwdAPIConfig(api_server_rpc_port="10.0.0.1:25100")
    with pytest.raises(ValueError, match="require lwd_config.path"):
        resolve_lwd_config(
            {},
            api=LwdAPIConfig.from_namespace(
                argparse.Namespace(api_server_rpc_port="10.0.0.1:25100")
            ),
        )
    with pytest.raises(ValueError, match="require lwd_config.path"):
        api.validate_topology(None)
    with pytest.raises(ValueError, match="scene=edge_share"):
        LwdConfig.from_dict(
            {
                "path": str(lwd_yaml_dir / "s3_cloud_share_1dp.yaml"),
                "role": "edge",
                "instance_id": 0,
            },
            api=api,
        )
