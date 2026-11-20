# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolved frontend attach endpoints of one LWD instance.

The topology declares them (``edges[].api``) for both data planes; this view is
what the serve path consumes, so the whole frontend attachment is configured in
the yaml file. There is no command-line form: ``--api-server-rpc-port`` and
``--api-server-attach`` are gone, and device-comm deployments declare the same
``api`` block as host-comm ones.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .topology import parse_endpoint

if TYPE_CHECKING:
    from .entry import LwdConfig
    from .topology import LwdInstance, LwdTopology


@dataclass(frozen=True)
class LwdAPIConfig:
    """Attach endpoints of the selected instance, parsed from the topology."""

    rpc_port: str | None = None
    """This engine primary's own registration endpoint, as IP:port."""
    attach: str | None = None
    """Primary endpoint this instance's frontend registers with, as IP:port."""
    primary: int | None = None
    """Instance id that declares `attach` as its endpoint."""

    @property
    def requested(self) -> bool:
        return self.rpc_port is not None or self.attach is not None

    @classmethod
    def from_instance(
        cls, instance: "LwdInstance", role: str, topology: "LwdTopology"
    ) -> "LwdAPIConfig":
        """Project the instance's yaml `api` block into the serve-path view."""
        declared = instance.api
        if declared is None:
            return cls()
        if role != "edge":
            raise ValueError("[LWD] api applies to edge instances only")
        if declared.rpc_port is not None:
            host, port = parse_endpoint(declared.rpc_port, "api.rpc_port")
            return cls(rpc_port=f"{host}:{port}")
        assert declared.attach is not None
        host, port = parse_endpoint(declared.attach, "api.attach")
        return cls(
            attach=f"{host}:{port}",
            primary=topology.rpc_port_owner(declared.attach),
        )

    def validate_topology(self, lwd: "LwdConfig") -> None:
        """Check the endpoint against the engine group it belongs to."""
        if not self.requested:
            return
        if lwd.role != "edge":
            raise ValueError("[LWD] api applies to edge instances only")
        members = lwd.topology.engine_members("edge", lwd.instance_id)
        if len(members) == 1:
            raise ValueError(
                "[LWD] api requires an edge engine shared by several instances; "
                "declare edges[].api.rpc_port on the primary and api.attach on "
                "every instance its engine should serve"
            )
        primary = min(members)
        if self.rpc_port is not None and lwd.instance_id != primary:
            raise ValueError(
                f"[LWD] api.rpc_port requires the engine primary instance {primary}"
            )
        if self.attach is not None and lwd.instance_id == primary:
            raise ValueError("[LWD] api.attach requires a non-primary edge")
