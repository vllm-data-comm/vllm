# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Socket-free handshake state machines of the LWD control link.

The edge registers with `LwdLinkHello` until the cloud answers with
`LwdLinkAccept` or `LwdLinkReject`. A cloud DP rank serves one edge for now.
"""

from dataclasses import dataclass

from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import (
    LWD_LINK_PROTOCOL_VERSION,
    LwdLinkAccept,
    LwdLinkHello,
    LwdLinkMessage,
    LwdLinkReject,
    LwdPeer,
    lwd_edge_identity,
)
from vllm.v1.lwd_control.lwd_comm.lwd_messages import LWD_MESSAGE_VERSION


class LwdHandshakeError(RuntimeError):
    """The edge-cloud handshake failed; the edge must not serve."""


class LwdEdgeHandshake:
    def __init__(self, hello: LwdLinkHello, hello_retry_interval_s: float) -> None:
        self.hello = hello
        self._hello_retry_interval_s = hello_retry_interval_s
        self._last_hello_at: float | None = None
        self.cloud: LwdPeer | None = None

    @property
    def completed(self) -> bool:
        return self.cloud is not None

    def should_send_hello(self, now: float) -> bool:
        """True until accepted, at most once per retry interval."""
        if self.completed:
            return False
        return (
            self._last_hello_at is None
            or now - self._last_hello_at >= self._hello_retry_interval_s
        )

    def on_hello_sent(self, now: float) -> None:
        self._last_hello_at = now

    def on_reply(self, msg: LwdLinkMessage) -> None:
        """Consume a cloud reply; repeated accepts of retried hellos are no-ops.

        Raises:
            LwdHandshakeError: Rejected, or an accept that does not answer
                this hello.
        """
        if isinstance(msg, LwdLinkReject):
            raise LwdHandshakeError(f"cloud rejected the hello: {msg.reason}")
        if not isinstance(msg, LwdLinkAccept):
            raise LwdHandshakeError(f"unexpected {type(msg).__name__} from the cloud")
        expected = (
            LWD_LINK_PROTOCOL_VERSION,
            self.hello.dp_rank,
            self.hello.dp_size,
            self.hello.session_id,
        )
        actual = (msg.link_version, msg.dp_rank, msg.dp_size, msg.session_id)
        if actual != expected:
            raise LwdHandshakeError(
                f"accept (link_version, dp_rank, dp_size, session) = {actual} "
                f"does not answer hello {expected}"
            )
        if self.cloud is None:
            self.cloud = LwdPeer(msg.instance_id, msg.dp_rank)


@dataclass(frozen=True)
class LwdCloudHandshakeDecision:
    reply: LwdLinkAccept | LwdLinkReject
    registered: LwdPeer | None = None
    """Set when this hello registered a new edge."""


@dataclass(frozen=True)
class _LwdEdgeSession:
    peer: LwdPeer
    identity: bytes
    session_id: str


class LwdCloudHandshake:
    """Registers edges on one cloud DP rank and maps them to ZMQ identities."""

    def __init__(self, instance_id: int, dp_rank: int, dp_size: int) -> None:
        self._instance_id = instance_id
        self._dp_rank = dp_rank
        self._dp_size = dp_size
        # TODO(lwd): turn the single slot into a session table to serve
        # several edges per cloud DP rank, issue #8.
        self._session: _LwdEdgeSession | None = None

    def peer_of(self, identity: bytes) -> LwdPeer | None:
        session = self._session
        if session is None or session.identity != identity:
            return None
        return session.peer

    def identity_of(self, peer: LwdPeer) -> bytes | None:
        session = self._session
        if session is None or session.peer != peer:
            return None
        return session.identity

    def on_hello(
        self, identity: bytes, hello: LwdLinkHello
    ) -> LwdCloudHandshakeDecision:
        reason = self._mismatch_reason(identity, hello)
        if reason is None:
            reason = self._occupied_reason(hello)
        if reason is not None:
            return LwdCloudHandshakeDecision(LwdLinkReject(reason))
        if self._session is not None:
            # Retried hello of the registered session.
            return LwdCloudHandshakeDecision(self._accept(hello))
        peer = LwdPeer(hello.instance_id, hello.dp_rank)
        self._session = _LwdEdgeSession(peer, identity, hello.session_id)
        return LwdCloudHandshakeDecision(self._accept(hello), registered=peer)

    def _mismatch_reason(self, identity: bytes, hello: LwdLinkHello) -> str | None:
        if hello.link_version != LWD_LINK_PROTOCOL_VERSION:
            return (
                f"link protocol version {hello.link_version} != cloud "
                f"{LWD_LINK_PROTOCOL_VERSION}"
            )
        if hello.message_version != LWD_MESSAGE_VERSION:
            return (
                f"message version {hello.message_version} != cloud "
                f"{LWD_MESSAGE_VERSION}"
            )
        if hello.dp_rank != self._dp_rank:
            return (
                f"edge dp_rank {hello.dp_rank} reached cloud dp_rank "
                f"{self._dp_rank}; check the control ports"
            )
        if hello.dp_size != self._dp_size:
            return f"edge dp_size {hello.dp_size} != cloud dp_size {self._dp_size}"
        expected = lwd_edge_identity(hello.instance_id, hello.dp_rank, hello.session_id)
        if identity != expected:
            return f"identity {identity!r} != expected {expected!r}"
        return None

    def _occupied_reason(self, hello: LwdLinkHello) -> str | None:
        session = self._session
        if session is None:
            return None
        if session.peer.instance_id != hello.instance_id:
            return (
                f"cloud dp{self._dp_rank} already serves edge instance "
                f"{session.peer.instance_id}; one edge per cloud DP rank is "
                "supported"
            )
        if session.session_id != hello.session_id:
            # TODO(lwd): once peer loss is detected, reject only while the
            # registered session is alive and replace it otherwise, issue #8.
            return (
                f"an edge with the same configuration (instance "
                f"{hello.instance_id}, dp{hello.dp_rank}) is already registered "
                f"on cloud {self._instance_id} dp{self._dp_rank} as session "
                f"{session.session_id}; this process is session "
                f"{hello.session_id}. Either a duplicate edge process uses the "
                "same instance_id (fix the deployment), or the edge restarted "
                "(restart the cloud as well)"
            )
        return None

    def _accept(self, hello: LwdLinkHello) -> LwdLinkAccept:
        return LwdLinkAccept(
            link_version=LWD_LINK_PROTOCOL_VERSION,
            instance_id=self._instance_id,
            dp_rank=self._dp_rank,
            dp_size=self._dp_size,
            session_id=hello.session_id,
        )
