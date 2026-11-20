# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A host engine that serves several edge instances keeps one link per connection."""

import pytest

from vllm.config.lwd.topology import LwdConnectionId, LwdControlPlaneConfig
from vllm.v1.lwd_control.lwd_comm.lwd_control_plane import (
    LwdEdgeControlPlane,
    LwdEdgeControlPlaneSpec,
)
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import (
    LWD_LINK_PROTOCOL_VERSION,
    LwdLinkAccept,
    LwdPeer,
)

# Pure control-plane objects: no device is touched, so the global device
# teardown neither can nor needs to run after these tests.
pytestmark = pytest.mark.skip_global_cleanup

CLOUD0_DP0 = LwdPeer(0, 0)
# Edge 0 and edge 1 both reach cloud 0 dp 0; edge 0 also reaches cloud 1 dp 0.
CONNECTIONS = (
    LwdConnectionId(0, 0, 0),
    LwdConnectionId(0, 1, 0),
    LwdConnectionId(1, 0, 0),
)


class _Listener:
    """Records peer registrations; the plane calls nothing else here."""

    def __init__(self) -> None:
        self.peers: list[LwdPeer] = []

    def on_peer_up(self, peer: LwdPeer) -> None:
        self.peers.append(peer)

    def on_message(self, peer: LwdPeer, payload: bytes, connection=None) -> None:
        return None

    def on_fatal(self, reason: str) -> None:
        return None


def _plane() -> LwdEdgeControlPlane:
    spec = LwdEdgeControlPlaneSpec(
        instance_id=0,
        dp_rank=0,
        dp_size=1,
        cloud_endpoints=("tcp://127.0.0.1:5550",) * 3,
        connection_ids=CONNECTIONS,
        cloud_dp_sizes=(1, 1, 1),
        config=LwdControlPlaneConfig(),
    )
    plane = LwdEdgeControlPlane(spec)
    plane._listener = _Listener()
    return plane


def _accept(plane: LwdEdgeControlPlane, connection_id: LwdConnectionId):
    hello = plane._handshakes[connection_id].hello
    return LwdLinkAccept(
        link_version=LWD_LINK_PROTOCOL_VERSION,
        instance_id=connection_id.cloud_id,
        dp_rank=hello.dp_rank,
        dp_size=hello.dp_size,
        session_id=hello.session_id,
    )


def test_accept_completes_only_the_link_it_arrived_on() -> None:
    """Both links reach cloud 0 dp 0; one accept must not answer both."""
    plane = _plane()
    answered, other = CONNECTIONS[0], CONNECTIONS[2]
    plane._on_link_message(None, None, _accept(plane, answered), 0)
    assert plane._handshakes[answered].completed
    assert not plane._handshakes[other].completed


def test_outbound_identity_follows_the_connection() -> None:
    """The peer alone cannot tell the two connections to cloud 0 dp 0 apart."""
    plane = _plane()
    assert plane._outbound_identity(CLOUD0_DP0, 0) == b"0"
    assert plane._outbound_identity(CLOUD0_DP0, 2) == b"2"
    assert plane._outbound_identity(CLOUD0_DP0) == b"0"


def test_link_of_maps_every_connection() -> None:
    plane = _plane()
    for index, connection_id in enumerate(CONNECTIONS):
        assert plane.link_of(connection_id) == index
