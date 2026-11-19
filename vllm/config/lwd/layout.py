# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Facts derived from the LWD topology and the role of this engine.

The topology file keeps both data-plane layouts in one schema; this module
projects the parsed topology into the one plane it actually uses and derives
everything the launch must agree with. Downstream code reads these facts
instead of walking the topology for itself.

Design: docs/design/lwd_config_plan.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.logger import init_logger

from .topology import (
    LwdConnectionId,
    LwdDPConnection,
    LwdHostDataPlane,
    LwdQosDataPlane,
    LwdTopology,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.lwd import LwdConfig

logger = init_logger(__name__)


@dataclass(frozen=True)
class LwdHostEndpoint:
    """Cloud DP endpoint of one connection on the host data plane."""

    addr: str
    data_port: int | None


@dataclass(frozen=True)
class LwdDevicePlane:
    """The device plane: one HCCL world shared by every instance."""

    hccl_world_size: int
    instance_dp_ranks: dict[tuple[str, int], tuple[tuple[int, ...], ...]]
    """``(role, instance_id)`` -> one rank block per DP of that instance."""

    @classmethod
    def from_topology(cls, topology: LwdTopology) -> LwdDevicePlane:
        """Validate and project the device layout.

        Raises:
            ValueError: The device layout of the topology is inconsistent.
        """
        deployment = topology.deployment
        if deployment.data_plane_host is not None:
            raise ValueError(
                "[LWD] data_plane_host is host-comm-only and must be "
                "omitted when data_plane_comm=device"
            )
        world_size = deployment.hccl_world_size
        if world_size is None:
            raise ValueError(
                "[LWD] hccl_world_size is required when "
                "data_plane_comm=device (the edge and cloud share one world)"
            )
        for instance in (*topology.edges, *topology.clouds):
            for dp in instance.dp:
                if not dp.ranks:
                    raise ValueError(
                        "[LWD] ranks are required on every DP when "
                        "data_plane_comm=device"
                    )
        for edge in topology.edges:
            if len({(dp.addr, dp.ranks) for dp in edge.dp}) != 1:
                raise ValueError(
                    "[LWD] Edge DPs must share their instance's ranks/address"
                )
        for cloud in topology.clouds:
            if len({len(dp.ranks) for dp in cloud.dp}) != 1:
                raise ValueError(
                    "[LWD] Cloud instance must split its ranks equally among DPs"
                )
        # Edge DPs reuse an allocation; edge_share also reuses it across
        # instances.
        edges = topology.edges
        if deployment.scene == "edge_share":
            if len({(edge.dp[0].addr, edge.dp[0].ranks) for edge in edges}) != 1:
                raise ValueError(
                    "[LWD] edge_share instances must share edge ranks/address"
                )
            edges = edges[:1]
        allocations = [edge.dp[0] for edge in edges]
        allocations.extend(dp for cloud in topology.clouds for dp in cloud.dp)
        ranks: set[int] = set()
        for dp in allocations:
            if len(set(dp.ranks)) != len(dp.ranks) or ranks.intersection(dp.ranks):
                raise ValueError(
                    "[LWD] Separate allocations require distinct ranks; "
                    "duplicate or overlapping ranks"
                )
            ranks.update(dp.ranks)
        if ranks != set(range(world_size)):
            raise ValueError(
                "[LWD] Unique ranks must cover [0, hccl_world_size) without gaps"
            )
        instance_dp_ranks = {
            ("edge", edge.id): tuple(dp.ranks for dp in edge.dp)
            for edge in topology.edges
        }
        instance_dp_ranks.update(
            {
                ("cloud", cloud.id): tuple(dp.ranks for dp in cloud.dp)
                for cloud in topology.clouds
            }
        )
        return cls(hccl_world_size=world_size, instance_dp_ranks=instance_dp_ranks)

    def dp_ranks_of(self, role: str, instance_id: int) -> tuple[tuple[int, ...], ...]:
        """The rank blocks of every DP of one instance, by DP index."""
        return self.instance_dp_ranks[(role, instance_id)]

    def validate_launch(self, vllm_config: VllmConfig) -> None:
        """Warn about switches the device plane accepts but ignores."""
        lwd_config = vllm_config.lwd_config
        assert lwd_config is not None
        qos = lwd_config.topology.qos
        if qos is not None and qos.data_plane is not None:
            # Host-plane settings are injected into UCX at link init; the
            # device plane has no such knob, so honour the config without
            # rejecting a topology that works today.
            logger.warning(
                "[LWD] qos.data_plane is host-comm-only and is ignored on "
                "the device data plane"
            )


@dataclass(frozen=True)
class LwdHostPlane:
    """The host plane: separate instances, one listener per cloud DP."""

    endpoints: dict[LwdConnectionId, LwdHostEndpoint]
    buffers: LwdHostDataPlane
    qos: LwdQosDataPlane

    @classmethod
    def from_topology(cls, topology: LwdTopology) -> LwdHostPlane:
        """Validate and project the host layout.

        Raises:
            ValueError: The host layout of the topology is inconsistent.
        """
        for instance in (*topology.edges, *topology.clouds):
            if any(dp.ranks for dp in instance.dp):
                raise ValueError(
                    "[LWD] ranks must be omitted when "
                    "data_plane_comm=host (no global comm domain; "
                    "links are established from addr/data_port)"
                )
        if topology.deployment.hccl_world_size is not None:
            raise ValueError(
                "[LWD] hccl_world_size must be omitted when "
                "data_plane_comm=host (the edge and cloud are "
                "separate worlds; their sizes come from their own "
                "launches)"
            )
        for instance in topology.edges:
            if any(dp.data_port is not None for dp in instance.dp):
                raise ValueError(
                    "[LWD] data_port is cloud-only (edges connect out and never listen)"
                )
        by_addr: dict[str, list[int]] = {}
        for instance in topology.clouds:
            for dp in instance.dp:
                if dp.data_port is None:
                    raise ValueError(
                        "[LWD] data_port is required on every cloud DP "
                        "when data_plane_comm=host (ctrl_port is "
                        "control-plane only)"
                    )
                by_addr.setdefault(dp.addr, []).append(dp.data_port)
        for addr, ports in by_addr.items():
            if len(ports) != len(set(ports)):
                raise ValueError(f"[LWD] duplicate data_port on {addr}")
        endpoints = {
            connection.connection_id: LwdHostEndpoint(
                connection.cloud.addr, connection.cloud.data_port
            )
            for edge in topology.edges
            for connection in topology.connections("edge", edge.id)
        }
        qos = topology.qos
        return cls(
            endpoints=endpoints,
            buffers=topology.deployment.data_plane_host or LwdHostDataPlane(),
            qos=qos.data_plane
            if qos is not None and qos.data_plane
            else LwdQosDataPlane(),
        )

    def endpoint_of(self, connection_id: LwdConnectionId) -> LwdHostEndpoint:
        """The cloud endpoint one connection dials."""
        return self.endpoints[connection_id]

    def validate_launch(self, vllm_config: VllmConfig) -> None:
        """Nothing to add at launch: the host rules live in the topology."""


LwdPlane = LwdDevicePlane | LwdHostPlane


def lwd_plane_of(topology: LwdTopology) -> LwdPlane:
    """Project the topology onto the data plane it uses.

    The one place that dispatches on ``data_plane_comm``: every other reader
    consumes the result instead of asking which plane is configured.
    """
    if topology.is_host_comm:
        return LwdHostPlane.from_topology(topology)
    return LwdDevicePlane.from_topology(topology)


@dataclass(frozen=True)
class LwdParallelExpectation:
    """What this engine must declare for the parallel sizes.

    ``tensor_parallel_size`` is ``None`` when the plane does not constrain it
    (host comm: the instance world is the launch's own, not the topology's).
    """

    data_parallel_size: int
    tensor_parallel_size: int | None


@dataclass(frozen=True)
class LwdInstancePlan:
    """Everything one process derives from the topology before it starts."""

    role: str
    instance_id: int
    dp_count: int
    connections: tuple[LwdDPConnection, ...]
    plane: LwdPlane
    expectation: LwdParallelExpectation
    world_size: int | None = None
    rank_blocks: tuple[tuple[int, ...], ...] | None = None

    @classmethod
    def from_config(cls, lwd_config: LwdConfig) -> LwdInstancePlan:
        """Derive the plan of the instance this process runs."""
        plane = lwd_plane_of(lwd_config.topology)
        instance = lwd_config.instance
        dp_count = len(instance.dp)
        world_size: int | None = None
        rank_blocks: tuple[tuple[int, ...], ...] | None = None
        tensor_parallel_size: int | None = None
        if isinstance(plane, LwdDevicePlane):
            world_size = plane.hccl_world_size
            rank_blocks = plane.dp_ranks_of(lwd_config.role, lwd_config.instance_id)
            tensor_parallel_size = len(rank_blocks[0])
        # One edge process serves every DP connection of its instance; a cloud
        # engine owns one DP. The rule does not depend on the plane.
        data_parallel_size = 1 if lwd_config.role == "edge" else dp_count
        return cls(
            role=lwd_config.role,
            instance_id=lwd_config.instance_id,
            dp_count=dp_count,
            connections=lwd_config.connections,
            plane=plane,
            expectation=LwdParallelExpectation(
                data_parallel_size=data_parallel_size,
                tensor_parallel_size=tensor_parallel_size,
            ),
            world_size=world_size,
            rank_blocks=rank_blocks,
        )

    @property
    def is_host(self) -> bool:
        """True on the host plane (separate instance worlds)."""
        return isinstance(self.plane, LwdHostPlane)

    def check_parallel(self, parallel_config: object) -> None:
        """Compare the launched TP size with the rank block of each DP.

        Raises:
            ValueError: The launched tensor parallel size does not match.
        """
        expected = self.expectation.tensor_parallel_size
        if expected is None:
            return
        actual = parallel_config.tensor_parallel_size  # type: ignore[attr-defined]
        if actual != expected:
            raise ValueError(
                f"[LWD] tensor parallel size {actual} "
                f"!= {expected} ranks per DP of {self.role} instance "
                f"{self.instance_id}"
            )

    def check_features(self, vllm_config: VllmConfig) -> None:
        """Reject feature switches the data plane cannot honour.

        Raises:
            ValueError: A plane-independent feature combination is invalid.
        """
        lwd_config = vllm_config.lwd_config
        assert lwd_config is not None
        scramble = lwd_config.topology.feature_ctrl.scramble
        if scramble.enable and (not scramble.device_id or not scramble.custom_factor):
            raise ValueError(
                "[LWD] feature_ctrl.scramble.enable requires both "
                "device_id and custom_factor"
            )
