# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YAML adapter for the shared bootstrap world from LWD PR #33."""

from typing import TYPE_CHECKING

from vllm.utils.network_utils import get_distributed_init_method

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.lwd import LwdConfig

LWD_WIRE_STORE_PORT = 29600


def get_lwd_rank_groups(config: "LwdConfig") -> list[list[int]]:
    if (
        config.topology.deployment.scene != "single_instance"
        or len(config.connections) != 1
    ):
        raise ValueError("[LWD] no-PP bootstrap requires single_instance DP=1")
    connection = config.connections[0]
    groups = [list(connection.edge.ranks), list(connection.cloud.ranks)]
    if groups[0] + groups[1] != list(range(config.topology.deployment.hccl_world_size)):
        raise ValueError("[LWD] bootstrap ranks must be contiguous, edge first")
    return groups


def get_lwd_init_method(config: "LwdConfig") -> str:
    return get_distributed_init_method(
        config.connections[0].edge.addr, LWD_WIRE_STORE_PORT
    )


def get_lwd_distributed_args(config: "VllmConfig", rank: int) -> tuple[int, int, str]:
    lwd, parallel = config.lwd_config, config.parallel_config
    assert lwd is not None
    groups = get_lwd_rank_groups(lwd)
    ranks = groups[0 if lwd.role == "edge" else 1]
    if (
        parallel.tensor_parallel_size != len(ranks)
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.nnodes != 1
        or parallel.enable_elastic_ep
    ):
        raise ValueError("[LWD] no-PP requires local TP=YAML ranks, PP=DP=PCP=DCP=1")
    if not 0 <= rank < len(ranks):
        raise ValueError("[LWD] local worker rank is outside its YAML allocation")
    return sum(map(len, groups)), ranks[rank], get_lwd_init_method(lwd)
