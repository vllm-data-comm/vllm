# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank groups of the HCCL world shared by the LWD edge and cloud instances.

Design: docs/design/lwd_data_plane.md.
"""

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.config.lwd import LwdTopology

if TYPE_CHECKING:
    from vllm.config import VllmConfig

LwdRankGroup = tuple[int, ...]


@dataclass(frozen=True)
class LwdChannelEndpoints:
    """Global ranks that hold the data-plane channels of one DP connection."""

    edge_rank: int
    cloud_rank: int


@dataclass(frozen=True)
class LwdParallelGroupsSpec:
    """Rank groups that every process of the shared world builds.

    Each edge instance and each cloud DP runs like a centralized pp=1
    deployment on its own ranks: its TP group is its rank allocation and all
    other model-parallel groups are singletons. Each DP connection adds one
    edge-cloud rank pair for the data-plane channels.

    Built by `from_topology`; every process derives the same spec from the
    same topology, so group creation (a world collective) runs in the same
    order everywhere.
    """

    world_size: int
    tp_groups: tuple[LwdRankGroup, ...]
    """One group per edge instance DP and cloud instance DP, by first rank."""
    channel_endpoints: tuple[LwdChannelEndpoints, ...]
    """One pair per DP connection, by edge rank."""

    @classmethod
    def from_topology(cls, topology: LwdTopology) -> "LwdParallelGroupsSpec":
        """Derive the groups, or raise before any process creates one.

        Raises:
            NotImplementedError: A rank is shared by several DPs, or holds the
                channels of several DP connections.
        """
        # 1. A DP's ranks are its TP group; one rank is one process, so reject sharing.
        allocations = [
            dp.ranks
            for instance in (*topology.edges, *topology.clouds)
            for dp in instance.dp
        ]
        if len(set(allocations)) != len(allocations):
            # TODO(lwd): host shared edge ranks once each DP connection gets
            # its own communication domain instead of one shared world.
            raise NotImplementedError(
                "[LWD] DPs sharing ranks cannot run in one shared HCCL world "
                f"(scene={topology.deployment.scene}): one rank is one process"
            )
        # 2. Endpoints are the first ranks; the cloud one broadcasts to its TP group.
        connections = [
            connection
            for edge in topology.edges
            for connection in topology.connections("edge", edge.id)
        ]
        channel_endpoints = sorted(
            (
                LwdChannelEndpoints(
                    connection.edge.ranks[0], connection.cloud_leader_rank
                )
                for connection in connections
            ),
            key=lambda endpoints: (endpoints.edge_rank, endpoints.cloud_rank),
        )
        # 3. One UP/DOWN pair per rank, so reject ranks in several connections.
        endpoint_uses = Counter(
            rank
            for endpoints in channel_endpoints
            for rank in (endpoints.edge_rank, endpoints.cloud_rank)
        )
        overloaded = sorted(rank for rank, uses in endpoint_uses.items() if uses > 1)
        if overloaded:
            # TODO(lwd): key the data-plane channels by DP connection.
            raise NotImplementedError(
                f"[LWD] ranks {overloaded} hold the channels of several DP "
                "connections; the data plane supports one per rank "
                f"(scene={topology.deployment.scene})"
            )
        # 4. Sorted groups make every process create them in the same order.
        return cls(
            world_size=topology.deployment.hccl_world_size,
            tp_groups=tuple(sorted(allocations)),
            channel_endpoints=tuple(channel_endpoints),
        )

    @property
    def singleton_groups(self) -> tuple[LwdRankGroup, ...]:
        return tuple((rank,) for rank in range(self.world_size))

    def tp_group_of(self, rank: int) -> LwdRankGroup:
        for group in self.tp_groups:
            if rank in group:
                return group
        raise ValueError(f"[LWD] rank {rank} is not in [0, {self.world_size})")

    def channel_endpoints_of(self, rank: int) -> LwdChannelEndpoints | None:
        """The channel pair this rank holds; None for non-endpoint ranks."""
        for endpoints in self.channel_endpoints:
            if rank in (endpoints.edge_rank, endpoints.cloud_rank):
                return endpoints
        return None


def check_lwd_parallel_config(vllm_config: "VllmConfig") -> None:
    """Reject at config time what the shared HCCL world cannot host.

    Every process reads the same topology, so both sides fail at startup,
    before any worker joins the world.

    Raises:
        NotImplementedError: A parallel feature or the topology is unsupported.
        ValueError: The tensor parallel size differs from the ranks of each DP
            of the local instance.
    """
    lwd_config = vllm_config.lwd_config
    assert lwd_config is not None
    parallel_config = vllm_config.parallel_config
    unsupported = [
        name
        for name, enabled in (
            ("pipeline parallelism", parallel_config.pipeline_parallel_size > 1),
            (
                "prefill context parallelism",
                parallel_config.prefill_context_parallel_size > 1,
            ),
            (
                "decode context parallelism",
                parallel_config.decode_context_parallel_size > 1,
            ),
            ("elastic EP", parallel_config.enable_elastic_ep),
            ("Engram", vllm_config.engram_config is not None),
        )
        if enabled
    ]
    if unsupported:
        raise NotImplementedError(
            f"[LWD] not supported in edge-cloud mode: {', '.join(unsupported)}"
        )
    # Raises for topologies the shared HCCL world cannot host.
    LwdParallelGroupsSpec.from_topology(lwd_config.topology)
    # The topology gives every DP of an instance the same rank count.
    dp_ranks = lwd_config.instance.dp[0].ranks
    if parallel_config.tensor_parallel_size != len(dp_ranks):
        raise ValueError(
            f"[LWD] tensor parallel size {parallel_config.tensor_parallel_size} "
            f"!= {len(dp_ranks)} ranks per DP of {lwd_config.role} instance "
            f"{lwd_config.instance_id}"
        )
