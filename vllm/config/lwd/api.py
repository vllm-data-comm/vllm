# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration-only API attachment; no service or routing implementation."""

import argparse
from dataclasses import dataclass
from ipaddress import ip_address
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .entry import LwdConfig


def _endpoint(value: str | None, flag: str) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ValueError(f"{flag} requires IP:port")
    host, sep, port = value.rpartition(":")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    elif ":" in host:
        raise ValueError(f"{flag}: IPv6 must use [IP]:port")
    if not sep or not port.isascii() or not port.isdecimal():
        raise ValueError(f"{flag} requires IP:port")
    if not 1 <= int(port) <= 65535:
        raise ValueError(f"{flag}: port must be in [1, 65535]")
    try:
        addr = ip_address(host)
    except ValueError as exc:
        raise ValueError(f"{flag}: invalid IP {host!r}") from exc
    if addr.is_unspecified or addr.is_multicast:
        raise ValueError(f"{flag}: use a concrete unicast IP")


@dataclass(frozen=True)
class LwdAPIConfig:
    """PR #42 API endpoint metadata forwarded with the topology snapshot."""

    api_server_rpc_port: str | None = None
    """Primary endpoint as IP:port, despite the historical option name."""
    api_server_attach: str | None = None
    """Primary endpoint referenced by a non-primary edge frontend."""

    @property
    def requested(self) -> bool:
        return (
            self.api_server_rpc_port is not None or self.api_server_attach is not None
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("LWD API configuration")
        for flag, action in (
            ("rpc-port", "start an RPC listener"),
            ("attach", "attach an existing engine"),
        ):
            group.add_argument(
                f"--api-server-{flag}",
                metavar="IP:port",
                help=f"Primary edge endpoint metadata; does not {action}.",
            )

    @staticmethod
    def validate_frontend(args: Any, *, headless: bool = False) -> None:
        requested = any(
            getattr(args, name, None) is not None
            for name in ("api_server_rpc_port", "api_server_attach")
        )
        if requested and (
            headless or getattr(args, "headless", False) or getattr(args, "grpc", False)
        ):
            raise ValueError("API options require the HTTP frontend")

    @classmethod
    def from_namespace(cls, args: Any, *, headless: bool = False) -> "LwdAPIConfig":
        cls.validate_frontend(args, headless=headless)
        return cls(
            api_server_rpc_port=getattr(args, "api_server_rpc_port", None),
            api_server_attach=getattr(args, "api_server_attach", None),
        )

    def validate_topology(self, lwd: "LwdConfig | None") -> None:
        if not self.requested:
            return
        if self.api_server_rpc_port is not None and self.api_server_attach is not None:
            raise ValueError(
                "--api-server-rpc-port and --api-server-attach "
                "are mutually exclusive"
            )
        _endpoint(self.api_server_rpc_port, "--api-server-rpc-port")
        _endpoint(self.api_server_attach, "--api-server-attach")
        if lwd is None:
            raise ValueError("API options require lwd_config.path")
        if lwd.role != "edge":
            raise ValueError("API options apply to edge instances only")
        if self.api_server_rpc_port is not None and lwd.instance_id != 0:
            raise ValueError("--api-server-rpc-port requires edge instance_id=0")
        if self.api_server_attach is not None and lwd.instance_id == 0:
            raise ValueError("--api-server-attach requires a non-primary edge")
        if lwd.topology.deployment.scene != "edge_share":
            raise ValueError("API attachment metadata requires scene=edge_share")
