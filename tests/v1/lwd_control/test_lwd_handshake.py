# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.lwd_control.lwd_comm.lwd_handshake import (
    LwdCloudHandshake,
    LwdEdgeHandshake,
    LwdHandshakeError,
)
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import (
    LWD_LINK_PROTOCOL_VERSION,
    LwdLinkAccept,
    LwdLinkHello,
    LwdLinkReject,
    lwd_edge_identity,
)
from vllm.v1.lwd_control.lwd_comm.lwd_messages import LWD_MESSAGE_VERSION


def _hello(runner_kind: str) -> LwdLinkHello:
    return LwdLinkHello(
        link_version=LWD_LINK_PROTOCOL_VERSION,
        message_version=LWD_MESSAGE_VERSION,
        instance_id=3,
        dp_rank=0,
        dp_size=1,
        session_id="session-a",
        runner_kind=runner_kind,
    )


def test_cloud_handshake_accepts_matching_runner_kind() -> None:
    handshake = LwdCloudHandshake(7, 0, 1, runner_kind="v2")
    hello = _hello("v2")

    decision = handshake.on_hello(
        lwd_edge_identity(hello.instance_id, hello.dp_rank, hello.session_id),
        hello,
    )

    assert isinstance(decision.reply, LwdLinkAccept)
    assert decision.reply.runner_kind == "v2"
    assert decision.registered is not None


def test_cloud_handshake_rejects_mismatched_runner_kind() -> None:
    handshake = LwdCloudHandshake(7, 0, 1, runner_kind="v2")
    hello = _hello("v1")

    decision = handshake.on_hello(
        lwd_edge_identity(hello.instance_id, hello.dp_rank, hello.session_id),
        hello,
    )

    assert isinstance(decision.reply, LwdLinkReject)
    assert "v1" in decision.reply.reason
    assert "v2" in decision.reply.reason
    assert decision.registered is None


def test_edge_handshake_rejects_mismatched_runner_kind() -> None:
    handshake = LwdEdgeHandshake(_hello("v1"), 0.1)

    with pytest.raises(LwdHandshakeError, match="v2.*v1"):
        handshake.on_reply(
            LwdLinkAccept(
                link_version=LWD_LINK_PROTOCOL_VERSION,
                instance_id=7,
                dp_rank=0,
                dp_size=1,
                session_id="session-a",
                runner_kind="v2",
            )
        )


def test_legacy_messages_default_to_v1() -> None:
    hello = LwdLinkHello(
        link_version=LWD_LINK_PROTOCOL_VERSION,
        message_version=LWD_MESSAGE_VERSION,
        instance_id=3,
        dp_rank=0,
        dp_size=1,
        session_id="session-a",
    )

    assert hello.runner_kind == "v1"
