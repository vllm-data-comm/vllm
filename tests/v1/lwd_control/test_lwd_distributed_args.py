# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Table-driven tests for the LWD wire-world bootstrap mapping.

``get_lwd_distributed_args`` is the seam that lets each side's executor keep
its per-process rank numbering while every worker joins the one HCCL world
the topology describes. Mocked SimpleNamespace inputs, no torch needed;
on torch-less hosts the conftest vllm stub plus a ``vllm.distributed``
namespace stub keep the import light.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

if importlib.util.find_spec("torch") is None:
    # Mirror the conftest vllm stub (the real vllm/__init__ pulls torch) and
    # additionally namespace-stub vllm.distributed so its torch-heavy package
    # __init__ (parallel_state) is skipped; vllm.distributed.lwd itself is
    # stdlib-only. vllm.envs uses 3.10-only annotations, so on old hosts a
    # one-entry stub with identical getenv semantics stands in for it.
    # On torch hosts everything imports natively.
    if "vllm" not in sys.modules:
        _vllm = types.ModuleType("vllm")
        _vllm.__path__ = [str(REPO_ROOT / "vllm")]
        sys.modules["vllm"] = _vllm
    if "vllm.distributed" not in sys.modules:
        _dist = types.ModuleType("vllm.distributed")
        _dist.__path__ = [str(REPO_ROOT / "vllm" / "distributed")]
        sys.modules["vllm.distributed"] = _dist
        sys.modules["vllm"].distributed = _dist
    if "vllm.envs" not in sys.modules:
        import os

        _envs = types.ModuleType("vllm.envs")

        def _envs_getattr(name: str) -> int:
            if name == "VLLM_LWD_WIRE_STORE_PORT":
                return int(os.getenv("VLLM_LWD_WIRE_STORE_PORT", "29600"))
            raise AttributeError(f"stub vllm.envs has no attribute {name!r}")

        _envs.__getattr__ = _envs_getattr
        sys.modules["vllm.envs"] = _envs
        sys.modules["vllm"].envs = _envs

from vllm.distributed.lwd import get_lwd_distributed_args, get_lwd_init_method

EDGE_IP = "76.76.26.18"
CLOUD_IP = "76.76.26.234"
EDGE_RANKS = (0,)
CLOUD_RANKS = tuple(range(1, 9))


def config_for(role: str) -> SimpleNamespace:
    """1-edge + 8-cloud single_instance layout as the YAML would describe it."""
    ranks = EDGE_RANKS if role == "edge" else CLOUD_RANKS
    return SimpleNamespace(
        lwd_config=SimpleNamespace(
            connections=(
                SimpleNamespace(
                    edge=SimpleNamespace(addr=EDGE_IP, ranks=EDGE_RANKS),
                    cloud=SimpleNamespace(addr=CLOUD_IP, ranks=CLOUD_RANKS),
                ),
            ),
            instance=SimpleNamespace(
                dp=(SimpleNamespace(addr=EDGE_IP if role == "edge" else CLOUD_IP,
                                    ranks=ranks),)
            ),
            topology=SimpleNamespace(
                deployment=SimpleNamespace(
                    scene="single_instance", hccl_world_size=9
                )
            ),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=len(ranks),
            data_parallel_index=0,
        ),
    )


def _clear_port_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_LWD_WIRE_STORE_PORT", raising=False)


@pytest.mark.parametrize(
    "role,tp_rank,global_rank",
    [("edge", 0, 0), ("cloud", 0, 1), ("cloud", 7, 8)],
)
def test_tp_rank_maps_to_yaml_global_rank_in_one_world(
    role: str, tp_rank: int, global_rank: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_port_env(monkeypatch)
    assert get_lwd_distributed_args(config_for(role), tp_rank) == (
        9,
        global_rank,
        f"tcp://{EDGE_IP}:29600",
    )


def test_wire_store_port_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_port_env(monkeypatch)
    monkeypatch.setenv("VLLM_LWD_WIRE_STORE_PORT", "29700")
    assert get_lwd_init_method(config_for("cloud").lwd_config) == (
        f"tcp://{EDGE_IP}:29700"
    )


@pytest.mark.parametrize("port", ["0", "-1", "65536", "invalid", ""])
def test_wire_store_port_env_rejected(
    port: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_port_env(monkeypatch)
    monkeypatch.setenv("VLLM_LWD_WIRE_STORE_PORT", port)
    with pytest.raises(ValueError, match=r"VLLM_LWD_WIRE_STORE_PORT"):
        get_lwd_init_method(config_for("edge").lwd_config)


def test_non_single_instance_scene_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_port_env(monkeypatch)
    config = config_for("cloud")
    config.lwd_config.topology.deployment.scene = "cloud_share"
    with pytest.raises(ValueError, match="single_instance"):
        get_lwd_distributed_args(config, 0)


def test_tp_size_mismatch_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_port_env(monkeypatch)
    config = config_for("cloud")
    config.parallel_config.tensor_parallel_size = 4
    with pytest.raises(ValueError, match="TP size must match YAML ranks"):
        get_lwd_distributed_args(config, 0)


def test_tp_rank_outside_yaml_ranks_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_port_env(monkeypatch)
    with pytest.raises(ValueError, match="outside"):
        get_lwd_distributed_args(config_for("cloud"), 8)
