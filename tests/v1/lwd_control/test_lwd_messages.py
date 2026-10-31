# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LWD_MESSAGE_VERSION,
    LwdRequestNotify,
    lwd_decode_edge_to_cloud,
    lwd_encode_message,
)


def test_request_notify_round_trips_full_prompt_mrope_positions() -> None:
    assert LWD_MESSAGE_VERSION == 2
    payload = bytes(range(24))
    decoded = lwd_decode_edge_to_cloud(
        lwd_encode_message(
            LwdRequestNotify(
                request_id="r1",
                num_prompt_tokens=1,
                mrope_positions=payload,
            )
        )
    )
    assert isinstance(decoded, LwdRequestNotify)
    assert decoded.mrope_positions == payload
