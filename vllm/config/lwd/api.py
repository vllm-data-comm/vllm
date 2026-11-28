# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolved frontend attach endpoints of one LWD instance.

The topology declares them (``edges[].api``) for both data planes; this view
is what the serve path consumes, so the whole frontend attachment is
configured in the yaml file. There is no command-line form:
``--api-server-rpc-port`` and ``--api-server-attach`` are gone.
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
    """Instance id that declares `attach` as its `api.rpc_port`."""

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
            return cls(rpc_port=f"{host}:{port}", primary=instance.id)
        assert declared.attach is not None
        host, port = parse_endpoint(declared.attach, "api.attach")
        return cls(
            attach=f"{host}:{port}",
            primary=topology.rpc_port_owner(declared.attach),
        )

    def validate_topology(self, lwd: "LwdConfig") -> None:
        """Check that attachment is declared on an edge instance.

        Endpoint shape, ownership and scene compatibility are validated once
        by ``LwdTopology._validate_api`` when the yaml is parsed.
        """
        if not self.requested:
            return
        if lwd.role != "edge":
            raise ValueError("[LWD] api applies to edge instances only")
