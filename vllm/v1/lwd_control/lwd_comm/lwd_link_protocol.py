# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire format of the LWD control link, independent of business messages.

ZMQ delivers each multipart message whole and in order, and on the cloud
ROUTER it prepends the sender's identity frame on receive and routes by it
on send. Everything else on the wire is defined here:

    edge DEALER:   [kind, body]
    cloud ROUTER:  [identity, kind, body]

`kind` is one byte. `LINK` frames carry a handshake message (hello, accept,
reject) and are decoded by the link. `PAYLOAD` frames carry a business
message that the link hands to the caller as bytes, so the link never needs
to know business message types. The transport drops messages with the wrong
number of frames or an unknown `kind`.

Example, an edge registering then sending a request:

    edge sends     [0x00, LwdLinkHello]
    cloud receives [b"lwd-edge0-dp0-<session>", 0x00, LwdLinkHello]
    edge sends     [0x01, encoded LwdRequestNotify]
    cloud receives [b"lwd-edge0-dp0-<session>", 0x01, encoded LwdRequestNotify]
"""

import enum
from typing import TypeAlias

import msgspec

LWD_LINK_PROTOCOL_VERSION = 1


class LwdFrameKind(enum.Enum):
    LINK = b"\x00"
    PAYLOAD = b"\x01"


class LwdPeer(msgspec.Struct, frozen=True, array_like=True, gc=False):  # type: ignore[call-arg]
    """One DP rank of an LWD engine instance, used as the address of a peer.

    Carries no role: on an edge it names a cloud, on a cloud an edge.
    """

    instance_id: int
    dp_rank: int


def lwd_edge_identity(instance_id: int, dp_rank: int, session_id: str) -> bytes:
    """ZMQ routing identity of one edge engine process.

    The session id keeps two processes configured with the same instance and
    rank apart at the ZMQ layer, so the later one reaches the handshake and is
    rejected instead of `ROUTER_HANDOVER` disconnecting the working one. A
    reconnect of the same process reuses the identity and is handed over.
    """
    return f"lwd-edge{instance_id}-dp{dp_rank}-{session_id}".encode()


# Link messages use map encoding (no array_like), so a peer speaking another
# link version still decodes and gets an explicit version reject.
class LwdLinkHello(msgspec.Struct, tag=True, gc=False):  # type: ignore[call-arg]
    """Edge -> cloud. Registers the edge; retried until answered."""

    link_version: int
    message_version: int
    instance_id: int
    dp_rank: int
    dp_size: int
    session_id: str
    """Unique per edge engine process; also part of its ZMQ identity."""


class LwdLinkAccept(msgspec.Struct, tag=True, gc=False):  # type: ignore[call-arg]
    """Cloud -> edge. Accepts the hello of `session_id`."""

    link_version: int
    instance_id: int
    dp_rank: int
    dp_size: int
    session_id: str


class LwdLinkReject(msgspec.Struct, tag=True, gc=False):  # type: ignore[call-arg]
    """Cloud -> edge. Refuses a hello; the edge must fail fast."""

    reason: str


LwdLinkMessage: TypeAlias = LwdLinkHello | LwdLinkAccept | LwdLinkReject

_LWD_LINK_DECODER = msgspec.msgpack.Decoder(LwdLinkMessage)
_LWD_PEER_DECODER = msgspec.msgpack.Decoder(LwdPeer)


def lwd_encode_link_message(msg: LwdLinkMessage) -> bytes:
    return msgspec.msgpack.encode(msg)


def lwd_decode_link_message(data: bytes) -> LwdLinkMessage:
    """Raises msgspec.DecodeError or msgspec.ValidationError on a bad frame."""
    return _LWD_LINK_DECODER.decode(data)


def lwd_encode_peer(peer: LwdPeer) -> bytes:
    return msgspec.msgpack.encode(peer)


def lwd_decode_peer(data: bytes) -> LwdPeer:
    return _LWD_PEER_DECODER.decode(data)
