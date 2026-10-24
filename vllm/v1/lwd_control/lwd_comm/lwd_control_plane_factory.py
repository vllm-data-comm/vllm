# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the control plane of one DP rank from `VllmConfig.lwd_config`.

Every cloud DP rank binds its topology `ctrl_port` on all interfaces; every
edge DP rank connects to the cloud DP rank of the same index.
"""

from vllm.config.lwd import LwdConfig
from vllm.config.lwd.topology import LwdDPConnection
from vllm.utils.network_utils import is_valid_ipv6_address, make_zmq_path
from vllm.v1.lwd_control.lwd_comm.lwd_control_plane import (
    LwdCloudControlPlane,
    LwdCloudControlPlaneSpec,
    LwdControlPlane,
    LwdEdgeControlPlane,
    LwdEdgeControlPlaneSpec,
)

# The topology address may be unreachable from inside the host (NAT,
# containers), so the cloud binds every interface of the address family.
_LWD_IPV4_ANY = "0.0.0.0"
_LWD_IPV6_ANY = "::"


def lwd_create_control_plane(
    lwd_config: LwdConfig, dp_rank: int, dp_size: int
) -> LwdControlPlane:
    """Create, not start, one control plane of this engine.

    Args:
        lwd_config: `VllmConfig.lwd_config` of this instance.
        dp_rank: Cloud DP index this link serves. On a cloud that is its own
            `ParallelConfig.data_parallel_index`; on an edge, which holds one
            link per cloud DP, the caller iterates over them.
        dp_size: Global DP size. Non-MoE engine processes reset
            `ParallelConfig.data_parallel_size` to 1 before the engine is
            built, so read it before that.

    Raises:
        ValueError: `dp_size` or `dp_rank` does not match the topology.
    """
    _lwd_validate_dp(lwd_config, dp_rank, dp_size)
    if lwd_config.role == "cloud":
        return LwdCloudControlPlane(_lwd_cloud_spec(lwd_config, dp_rank, dp_size))
    return LwdEdgeControlPlane(_lwd_edge_spec(lwd_config, dp_rank, dp_size))


def _lwd_validate_dp(lwd_config: LwdConfig, dp_rank: int, dp_size: int) -> None:
    """Check the link index against the topology.

    The DP count that matters is the cloud's: it is the number of links, and
    on a cloud it also equals its own vLLM `data_parallel_size`. The edge
    runs a single process whatever that count is, so its vLLM DP size is 1
    and is deliberately not compared.
    """
    topology_dp_size = lwd_config.cloud_dp_size
    if lwd_config.role == "cloud" and dp_size != topology_dp_size:
        raise ValueError(
            f"[LWD] vLLM data parallel size {dp_size} != topology DP count "
            f"{topology_dp_size} of cloud instance {lwd_config.instance_id}"
        )
    # Checked before the topology is indexed by dp_rank; the spec checks it
    # again only for callers that build a spec by hand.
    if not 0 <= dp_rank < topology_dp_size:
        raise ValueError(
            f"[LWD] dp_rank {dp_rank} not in [0, {topology_dp_size})"
        )


def _lwd_connection(lwd_config: LwdConfig, dp_rank: int) -> LwdDPConnection:
    """The DP connection of one link, by cloud DP index.

    Connections are zipped by DP index, so a cloud DP has exactly one and a
    shared edge has one per cloud DP. More than one match would mean two
    clouds at the same index, which the topology does not express.
    """
    connections = [c for c in lwd_config.connections if c.dp_idx == dp_rank]
    if len(connections) != 1:
        raise NotImplementedError(
            f"[LWD] {lwd_config.role} instance {lwd_config.instance_id} "
            f"dp{dp_rank} has {len(connections)} DP connections; one link "
            f"serves one (scene={lwd_config.topology.deployment.scene})"
        )
    return connections[0]


def _lwd_cloud_spec(
    lwd_config: LwdConfig, dp_rank: int, dp_size: int
) -> LwdCloudControlPlaneSpec:
    _lwd_connection(lwd_config, dp_rank)
    local = lwd_config.instance.dp[dp_rank]
    assert local.ctrl_port is not None, "topology requires ctrl_port on clouds"
    bind_host = _LWD_IPV6_ANY if is_valid_ipv6_address(local.addr) else _LWD_IPV4_ANY
    return LwdCloudControlPlaneSpec(
        instance_id=lwd_config.instance_id,
        dp_rank=dp_rank,
        dp_size=dp_size,
        bind_endpoint=make_zmq_path("tcp", bind_host, local.ctrl_port),
        config=lwd_config.topology.control_plane,
    )


def _lwd_edge_spec(
    lwd_config: LwdConfig, dp_rank: int, dp_size: int
) -> LwdEdgeControlPlaneSpec:
    """Spec of the edge link to cloud DP `dp_rank`.

    `dp_rank` names the cloud side of the link, not the edge process: the
    cloud handshake checks it against its own rank, so the edge has to
    introduce itself as "the peer of cloud DP k" on link k. `dp_size` is
    likewise the cloud's DP count, which both sides compare.
    """
    cloud = _lwd_connection(lwd_config, dp_rank).cloud
    assert cloud.ctrl_port is not None, "topology requires ctrl_port on clouds"
    return LwdEdgeControlPlaneSpec(
        instance_id=lwd_config.instance_id,
        dp_rank=dp_rank,
        dp_size=dp_size,
        cloud_endpoint=make_zmq_path("tcp", cloud.addr, cloud.ctrl_port),
        config=lwd_config.topology.control_plane,
    )
