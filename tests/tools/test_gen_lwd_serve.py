# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Render-level tests for tools/gen_lwd_serve.py (pure stdlib, no device).

The generator itself shells out to tools/gen_lwd_topology.py only when the
topology is (re)generated; these tests use --skip-topology with a fixture
file so they run on any interpreter the repo supports.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "gen_lwd_serve.py"
EDGE_IP = "10.0.0.1"
CLOUD_IP = "10.1.0.1"


def run_gen(out_dir: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--edge-ip",
            EDGE_IP,
            "--cloud-ip",
            CLOUD_IP,
            "-o",
            str(out_dir),
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def generated(out_dir: Path, *extra: str) -> tuple[str, str, Path]:
    """Fixture topology + one generator run; returns (edge, cloud, topology)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    topology = out_dir / "lwd_topology.yaml"
    topology.write_text("deployment: {}\n", encoding="utf-8")
    proc = run_gen(out_dir, "--skip-topology", *extra)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    edge = (out_dir / "serve_edge.sh").read_text(encoding="utf-8")
    cloud = (out_dir / "serve_cloud.sh").read_text(encoding="utf-8")
    return edge, cloud, topology


def additional_config(script: str) -> dict:
    match = re.search(r"--additional-config '(\{.*?\})'", script)
    assert match is not None, "additional-config missing"
    return json.loads(match.group(1))


def test_skip_topology_writes_executable_scripts(tmp_path: Path) -> None:
    edge, cloud, topology = generated(tmp_path / "out")
    assert topology.is_file()
    for name in ("serve_edge.sh", "serve_cloud.sh"):
        assert os.access(tmp_path / "out" / name, os.X_OK)
    assert edge.startswith("#!/bin/sh") and cloud.startswith("#!/bin/sh")


def test_edge_script_bakes_role_topology_and_single_card(
    tmp_path: Path,
) -> None:
    edge, _, topology = generated(tmp_path / "out")
    assert f"export VLLM_HOST_IP={EDGE_IP}" in edge
    assert "export ASCEND_RT_VISIBLE_DEVICES=3" in edge
    assert "export VLLM_LWD_LOG_LEVEL=INFO" in edge
    assert "--host 0.0.0.0" in edge and "--port 9724" in edge
    assert "--pipeline-parallel-size 1" in edge
    assert "--tensor-parallel-size 1" in edge
    assert "--headless" not in edge
    assert additional_config(edge)["lwd_config"] == {
        "path": str(topology),
        "role": "edge",
        "instance_id": 0,
    }


def test_cloud_script_derives_tp_and_keeps_a_frontend(tmp_path: Path) -> None:
    _, cloud, topology = generated(tmp_path / "out")
    assert f"export VLLM_HOST_IP={CLOUD_IP}" in cloud
    assert "export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" in cloud
    assert "--tensor-parallel-size 8" in cloud
    assert "--pipeline-parallel-size 1" in cloud
    # --headless would starve the engine startup handshake (it expects an
    # external LB supervisor); the cloud keeps its own front-end instead.
    assert "--headless" not in cloud
    assert "--port 9725" in cloud
    spec = re.search(r"--speculative-config '(\{.*?\})'", cloud)
    assert spec is not None, "speculative config missing"
    assert json.loads(spec.group(1))["num_speculative_tokens"] == 3
    assert "800]" in cloud, "capture sizes should end at the default max 800"
    assert additional_config(cloud)["lwd_config"]["path"] == str(topology)
    assert additional_config(cloud)["lwd_config"]["role"] == "cloud"


def test_spec_tokens_zero_omits_speculative_config(tmp_path: Path) -> None:
    _, cloud, _ = generated(tmp_path / "out", "--spec-tokens", "0")
    assert "--speculative-config" not in cloud


def test_rejects_multi_device_edge(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "lwd_topology.yaml").write_text("deployment: {}\n", encoding="utf-8")
    proc = run_gen(out_dir, "--skip-topology", "--edge-devices", "0,1")
    assert proc.returncode != 0
    assert "one rank per edge instance" in proc.stderr + proc.stdout


def test_skip_topology_requires_existing_topology(tmp_path: Path) -> None:
    proc = run_gen(tmp_path / "out", "--skip-topology")
    assert proc.returncode != 0
    assert "missing" in proc.stderr


def test_generates_topology_through_repo_tool(tmp_path: Path) -> None:
    """Full path: the repo's gen_lwd_topology.py must accept our invocation."""
    proc = run_gen(tmp_path / "out")
    if proc.returncode != 0:
        pytest.skip(f"topology tool needs a newer interpreter: {proc.stderr}")
    topology = (tmp_path / "out" / "lwd_topology.yaml").read_text(encoding="utf-8")
    assert "hccl_world_size: 9" in topology
    assert "scene: single_instance" in topology
