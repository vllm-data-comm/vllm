# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Log the resolved LWD metadata without reading files or starting execution."""

import json
from dataclasses import asdict

from vllm.lwd_logger import lwd_logger

from .entry import LwdConfig


def log_lwd_config(config: LwdConfig) -> None:
    """Emit one startup record with all fields, defaults and selected connections."""
    snapshot = {
        "metadata_only": True,
        "lwd_config": asdict(config),
        "selected_instance": asdict(config.instance),
        "connections": [
            {**asdict(link), "cloud_leader_rank": link.cloud_leader_rank}
            for link in config.connections
        ],
        "cloud_cards": {
            str(cloud.id): sum(len(dp.ranks) for dp in cloud.dp)
            for cloud in config.topology.clouds
        },
    }
    # Serialize the parsed snapshot only; tenant_key_file remains a path, not content.
    lwd_logger.info("[config] resolved=%s", json.dumps(snapshot, ensure_ascii=False))
