# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LWD (layerwise disaggregated) edge-cloud parallel groups."""

from typing import TYPE_CHECKING

from vllm import envs

from .lwd_parallel_groups import (
    LwdChannelEndpoints,
    LwdParallelGroupsSpec,
    LwdRankGroup,
    lwd_check_parallel_config,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.lwd import LwdConfig

__all__ = [
    "LwdChannelEndpoints",
    "LwdParallelGroupsSpec",
    "LwdRankGroup",
    "get_lwd_distributed_args",
    "get_lwd_init_method",
    "lwd_check_parallel_config",
]


def get_lwd_init_method(config: "LwdConfig") -> str:
    """Wire store both sides rendezvous on: hosted where global rank 0 lives.

    Plain ``tcp://`` URL; the topology validator already guarantees the
    address is a concrete unicast IP, so no extra formatting is needed.
    """
    try:
        port = envs.VLLM_LWD_WIRE_STORE_PORT
    except ValueError as exc:
        raise ValueError("[LWD] VLLM_LWD_WIRE_STORE_PORT must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("[LWD] VLLM_LWD_WIRE_STORE_PORT must be in [1, 65535]")
    root = next(
        dp
        for connection in config.connections
        for dp in (connection.edge, connection.cloud)
        if 0 in dp.ranks
    )
    return f"tcp://{root.addr}:{port}"


def get_lwd_distributed_args(
    config: "VllmConfig", tp_rank: int
) -> tuple[int, int, str]:
    """Reinterpret the incoming rank as the instance-local TP rank.

    Returns ``(world_size, global_rank, init_method)`` for joining the one
    HCCL world the topology describes: the executor numbers ranks per serve
    process, while the shared world needs the YAML-allocated global ranks.
    """
    lwd_config = config.lwd_config
    parallel_config = config.parallel_config
    assert lwd_config is not None
    if lwd_config.topology.deployment.scene != "single_instance":
        raise ValueError("[LWD] no-PP bootstrap requires single_instance")

    dp_idx = parallel_config.data_parallel_index
    ranks = lwd_config.instance.dp[dp_idx].ranks
    if parallel_config.tensor_parallel_size != len(ranks):
        raise ValueError("[LWD] local TP size must match YAML ranks")
    if not 0 <= tp_rank < len(ranks):
        raise ValueError(
            f"[LWD] tp_rank {tp_rank} outside the {len(ranks)} YAML ranks"
            f" of this instance"
        )

    return (
        lwd_config.topology.deployment.hccl_world_size,
        ranks[tp_rank],
        get_lwd_init_method(lwd_config),
    )
