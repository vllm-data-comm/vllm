# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank groups of the HCCL world shared by the LWD edge and cloud instances.

Design: docs/design/lwd_data_plane.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.config.engram import model_has_engram_layers
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.lwd import LwdTopology

logger = init_logger(__name__)

LwdRankGroup = tuple[int, ...]

_LWD_ENGRAM_ARCHITECTURES = ("DeepseekV41ForCausalLM",)
"""Architectures whose Engram the edge hashes and the cloud looks up."""


@dataclass(frozen=True)
class LwdChannelEndpoints:
    """Global ranks that hold the data-plane channels of one DP connection."""

    dp_idx: int
    edge_rank: int
    cloud_rank: int


@dataclass(frozen=True)
class LwdParallelGroupsSpec:
    """Rank groups that every process of the shared world builds.

    Each edge instance and each cloud DP runs like a centralized pp=1
    deployment on its own ranks: its TP group is its rank allocation, and the
    PP, DCP and PCP groups are singletons. An instance whose DPs span several
    rank allocations adds two groups that span them: the DP group joins the
    replicas of one TP position, and the EP group covers the whole instance,
    so that experts shard over DP x TP as they do in a centralized
    deployment. Each DP connection adds one edge-cloud rank pair for the
    data-plane channels.

    Built by `from_topology`; every process derives the same spec from the
    same topology, so group creation (a world collective) runs in the same
    order everywhere.
    """

    world_size: int
    tp_groups: tuple[LwdRankGroup, ...]
    """One group per edge instance DP and cloud instance DP, by first rank."""
    dp_groups: tuple[LwdRankGroup, ...]
    """One group per TP position of an instance, joining its DP replicas.

    Covers every rank: a DP of its own is a singleton. Sorted by first rank.
    """
    ep_groups: tuple[LwdRankGroup, ...]
    """One group per instance, over all of its ranks, by first rank.

    The MoE all2all rides this group, so it has to span the DP replicas that
    hold the other expert shards. Ascend builds its MC2 group over the same
    ranks.
    """
    channel_endpoints: tuple[LwdChannelEndpoints, ...]
    """One pair per DP connection, by edge rank."""

    @classmethod
    def from_topology(cls, topology: LwdTopology) -> LwdParallelGroupsSpec:
        """Derive the groups, or raise before any process creates one.

        Raises:
            NotImplementedError: A rank is shared by several DPs of different
                instances.
        """
        instances = [
            (instance, [dp.ranks for dp in instance.dp])
            for instance in (*topology.edges, *topology.clouds)
        ]
        # 1. A DP's ranks are its TP group. Edge DPs of one instance share an
        #    allocation (one edge process serves every cloud DP), so only
        #    count each allocation once; `_validate_ranks` already rejects
        #    allocations overlapping across instances.
        allocations = sorted({ranks for _, group in instances for ranks in group})
        # 2. Endpoints are the first ranks; the cloud one broadcasts to its TP group.
        connections = [
            connection
            for edge in topology.edges
            for connection in topology.connections("edge", edge.id)
        ]
        channel_endpoints = sorted(
            (
                LwdChannelEndpoints(
                    connection.dp_idx,
                    connection.edge.ranks[0],
                    connection.cloud_leader_rank,
                )
                for connection in connections
            ),
            key=lambda endpoints: (endpoints.edge_rank, endpoints.cloud_rank),
        )
        # 3. DP groups join the replicas of one TP position; EP groups span a
        #    whole instance. Both are singletons at one DP, which is what the
        #    native groups are in a centralized pp=1 deployment.
        dp_groups: list[LwdRankGroup] = []
        ep_groups: list[LwdRankGroup] = []
        for _, group in instances:
            distinct = sorted(set(group))
            ep_groups.append(
                tuple(sorted(rank for ranks in distinct for rank in ranks))
            )
            dp_groups.extend(tuple(sorted(replicas)) for replicas in zip(*distinct))
        # 4. Sorted groups make every process create them in the same order.
        return cls(
            world_size=topology.deployment.hccl_world_size,
            tp_groups=tuple(allocations),
            dp_groups=tuple(sorted(dp_groups)),
            ep_groups=tuple(sorted(ep_groups)),
            channel_endpoints=tuple(channel_endpoints),
        )

    @property
    def singleton_groups(self) -> tuple[LwdRankGroup, ...]:
        return tuple((rank,) for rank in range(self.world_size))

    def tp_group_of(self, rank: int) -> LwdRankGroup:
        return self._group_of(self.tp_groups, rank, "TP")

    def dp_group_of(self, rank: int) -> LwdRankGroup:
        return self._group_of(self.dp_groups, rank, "DP")

    def ep_group_of(self, rank: int) -> LwdRankGroup:
        return self._group_of(self.ep_groups, rank, "EP")

    def _group_of(
        self, groups: tuple[LwdRankGroup, ...], rank: int, kind: str
    ) -> LwdRankGroup:
        for group in groups:
            if rank in group:
                return group
        raise ValueError(
            f"[LWD] rank {rank} is in no {kind} group of [0, {self.world_size})"
        )

    def channel_endpoints_of(self, rank: int) -> tuple[LwdChannelEndpoints, ...]:
        """The channel pairs this rank holds, by DP index; empty for none.

        A shared edge rank holds one pair per cloud DP it serves.
        """
        return tuple(
            endpoints
            for endpoints in self.channel_endpoints
            if rank in (endpoints.edge_rank, endpoints.cloud_rank)
        )


def lwd_check_parallel_config(vllm_config: VllmConfig) -> None:
    """Reject at config time what the shared HCCL world cannot host.

    Every process reads the same topology, so both sides fail at startup,
    before any worker joins the world.

    Raises:
        NotImplementedError: A parallel feature, the topology or the Engram
            placement is unsupported.
        ValueError: The tensor parallel size differs from the ranks of each DP
            of the local instance, or the edge model kept n-gram layers.
    """
    lwd_config = vllm_config.lwd_config
    assert lwd_config is not None
    parallel_config = vllm_config.parallel_config
    engram_config = vllm_config.engram_config
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
            (
                "Engram dp_shared_memory",
                engram_config is not None and engram_config.dp_shared_memory,
            ),
            (
                "Engram embedding_across_dp",
                engram_config is not None and engram_config.embedding_across_dp,
            ),
        )
        if enabled
    ]
    if unsupported:
        raise NotImplementedError(
            f"[LWD] not supported in edge-cloud mode: {', '.join(unsupported)}"
        )
    _lwd_check_engram_layers(vllm_config)
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


def _lwd_check_engram_layers(vllm_config: VllmConfig) -> None:
    """Engram tables live on the cloud, for the architectures LWD hashes.

    Reads the model, not `engram_config`: a platform may resolve the Engram
    config after this check (Ascend does, in `check_and_update_config`).

    Raises:
        ValueError: The edge model kept its n-gram layers through the edge
            projection.
        NotImplementedError: The cloud model has n-gram layers of an
            architecture LWD does not hash.
    """
    model_config = vllm_config.model_config
    if model_config is None or not model_has_engram_layers(model_config):
        return
    lwd_config = vllm_config.lwd_config
    assert lwd_config is not None
    architecture = model_config.architecture
    if lwd_config.role == "edge":
        raise ValueError(
            f"[LWD] the edge {architecture} model keeps its n-gram embedding "
            "layers; the edge projection clears them unless hf_overrides is a "
            "dict that updates a sub-config: pass flat keys or a callable"
        )
    if architecture not in _LWD_ENGRAM_ARCHITECTURES:
        raise NotImplementedError(
            f"[LWD] Engram of {architecture} is not supported in edge-cloud "
            f"mode; supported: {_LWD_ENGRAM_ARCHITECTURES}"
        )
    logger.info("[LWD] cloud hosts the Engram tables of %s", architecture)
