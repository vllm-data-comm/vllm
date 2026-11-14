# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The simulated model: the real Qwen3-0.6B config, no weights, fixed outputs.

The config sets everything the flow depends on: the hidden size gives the
UP/DOWN transfer sizes, the layer and KV head counts the cloud's KV blocks.
"""

from pathlib import Path

LWD_SIM_MODEL_DIR = Path(__file__).parent / "qwen3_0_6b"
LWD_SIM_HIDDEN_SIZE = 1024
"""`hidden_size` of the config."""

LWD_SIM_SAMPLED_TOKEN_ID = 100
"""Token of every simulated sample; any id but the EOS keeps requests going."""

_LWD_SIM_RANK_MODULUS = 7


def lwd_sim_token_rank(down_seqno: int, row: int) -> int:
    """Sampled-token rank the cloud reports for `row` of DOWN step
    `down_seqno`; it varies per row so the copies the edge receives can be
    matched to the cloud's."""
    return (down_seqno + row) % _LWD_SIM_RANK_MODULUS
