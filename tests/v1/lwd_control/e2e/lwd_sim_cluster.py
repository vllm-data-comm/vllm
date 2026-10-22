# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runs one LWD edge-cloud pair on CPU and collects its trace.

Each instance runs in its own driver process (driver -> EngineCore ->
worker), started the way a deployment starts them, with the stand-in workers
of `lwd_sim_workers`. The edge's topology points its control link at a tap in
the test process that relays every frame to the cloud and records it; the
cloud's topology carries the cloud's own port.

The cloud engine imports vllm-ascend's early-receive module, so the drivers
need the vllm-ascend source tree named by `VLLM_LWD_ASCEND_PATH`.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgspec
import yaml
import zmq

from tests.v1.lwd_control.e2e.lwd_sim_model import LWD_SIM_MODEL_DIR
from tests.v1.lwd_control.e2e.lwd_sim_trace import (
    LWD_SIM_TRACE_DIR_ENV,
    LwdSimTrace,
    LwdSimTraceWriter,
)
from vllm.utils.network_utils import get_open_ports_list
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import (
    LwdFrameKind,
    lwd_decode_link_message,
)
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    lwd_decode_cloud_to_edge,
    lwd_decode_edge_to_cloud,
)

LWD_ASCEND_PATH_ENV = "VLLM_LWD_ASCEND_PATH"
_LWD_ASCEND_MARKER = "vllm_ascend/worker/lwd_cloud_model/lwd_pre_recv.py"
_REPO_ROOT = Path(__file__).resolve().parents[4]
_WORKERS_MODULE = "tests.v1.lwd_control.e2e.lwd_sim_workers"
_DRIVER_MODULE = "tests.v1.lwd_control.e2e.lwd_sim_driver"
_LOCALHOST = "127.0.0.1"
_MAX_MODEL_LEN = 2048
_HELLO_RETRY_INTERVAL_S = 0.2
_HANDSHAKE_TIMEOUT_S = 120.0
_EDGE_TIMEOUT_S = 420.0
_CLOUD_EXIT_TIMEOUT_S = 30.0
_TAP_POLL_MS = 50
_TAP_JOIN_TIMEOUT_S = 5.0
_LOG_TAIL_LINES = 80
_TRACE_TAIL_EVENTS = 200


def lwd_ascend_path() -> Path | None:
    """The vllm-ascend source tree, or None when the variable is unset.

    Raises:
        ValueError: The variable names a directory without the early-receive
            module.
    """
    raw = os.environ.get(LWD_ASCEND_PATH_ENV)
    if not raw:
        return None
    path = Path(raw).resolve()
    if not (path / _LWD_ASCEND_MARKER).is_file():
        raise ValueError(f"{LWD_ASCEND_PATH_ENV}={raw} has no {_LWD_ASCEND_MARKER}")
    return path


@dataclass(frozen=True)
class LwdSimRequest:
    prompt_len: int
    max_tokens: int


@dataclass(frozen=True)
class LwdSimLimits:
    """Scheduling limits the edge checks against the cloud's at startup."""

    max_num_batched_tokens: int = 256
    max_num_seqs: int = 4


@dataclass
class LwdSimRun:
    returncodes: dict[str, int | None]
    """Exit code per role; None when the role was killed on timeout."""
    cloud_alive_after_edge: bool
    edge_results: list[dict[str, Any]]
    trace: LwdSimTrace
    log_paths: dict[str, Path] = field(default_factory=dict)

    def log(self, role: str) -> str:
        return self.log_paths[role].read_text(errors="replace")

    def describe(self) -> str:
        """Exit codes, log tails and trace tail, for assertion messages."""
        parts = [f"returncodes={self.returncodes}"]
        for role in self.log_paths:
            tail = self.log(role).splitlines()[-_LOG_TAIL_LINES:]
            parts.append(f"--- {role} log tail ---\n" + "\n".join(tail))
        events = self.trace.dump().splitlines()[-_TRACE_TAIL_EVENTS:]
        parts.append("--- trace tail ---\n" + "\n".join(events))
        return "\n".join(parts)


class LwdWireTap:
    """Relays the control link of one edge to the cloud, recording each frame.

    The upstream DEALER takes the edge's ZMQ identity, so the cloud handshake
    sees the edge itself, and sets IMMEDIATE as the edge does: a frame the
    cloud cannot take yet (a hello before it listens) is dropped and recorded
    as undelivered, and the edge retries it.
    """

    def __init__(
        self, listen_port: int, cloud_endpoint: str, trace: LwdSimTraceWriter
    ) -> None:
        self._listen_endpoint = f"tcp://{_LOCALHOST}:{listen_port}"
        self._cloud_endpoint = cloud_endpoint
        self._trace = trace
        self._stopped = threading.Event()
        self._bound = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="lwd-wire-tap")

    def __enter__(self) -> LwdWireTap:
        self._thread.start()
        self._bound.wait()
        if self._error is not None:
            raise self._error
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stopped.set()
        self._thread.join(_TAP_JOIN_TIMEOUT_S)
        if self._error is not None:
            raise RuntimeError("the control-link tap failed") from self._error

    def _run(self) -> None:
        context = zmq.Context()
        front = context.socket(zmq.ROUTER)
        front.setsockopt(zmq.LINGER, 0)
        upstream: zmq.Socket | None = None
        try:
            front.bind(self._listen_endpoint)
            self._bound.set()
            poller = zmq.Poller()
            poller.register(front, zmq.POLLIN)
            identity = b""
            while not self._stopped.is_set():
                for socket, _ in poller.poll(_TAP_POLL_MS):
                    if socket is front:
                        sender, kind, body = front.recv_multipart()
                        if upstream is None:
                            identity = sender
                            upstream = self._connect_upstream(context, identity)
                            poller.register(upstream, zmq.POLLIN)
                        assert sender == identity, "one edge per cloud DP rank"
                        self._relay(
                            "e2c",
                            kind,
                            body,
                            self._make_sender(upstream, [kind, body]),
                        )
                    else:
                        assert upstream is not None
                        kind, body = upstream.recv_multipart()
                        self._relay(
                            "c2e",
                            kind,
                            body,
                            self._make_sender(front, [identity, kind, body]),
                        )
        except BaseException as e:
            self._error = e
            self._bound.set()
        finally:
            if upstream is not None:
                upstream.close()
            front.close()
            context.term()

    @staticmethod
    def _make_sender(socket: zmq.Socket, frames: list[bytes]) -> Callable[[], None]:
        """A zero-argument send of `frames` on `socket`, failing with
        zmq.Again when the peer cannot take it."""
        return lambda: socket.send_multipart(frames, zmq.NOBLOCK)

    def _connect_upstream(self, context: zmq.Context, identity: bytes) -> zmq.Socket:
        upstream = context.socket(zmq.DEALER)
        upstream.setsockopt(zmq.ROUTING_ID, identity)
        upstream.setsockopt(zmq.IMMEDIATE, 1)
        upstream.setsockopt(zmq.LINGER, 0)
        upstream.connect(self._cloud_endpoint)
        return upstream

    def _relay(
        self, direction: str, kind: bytes, body: bytes, send: Callable[[], None]
    ) -> None:
        # Stamped before sending: the receiver may react before the record.
        t = time.monotonic_ns()
        try:
            send()
            delivered = True
        except zmq.Again:
            delivered = False
        self._trace.record(
            "frame",
            t=t,
            direction=direction,
            delivered=delivered,
            **_lwd_describe_frame(direction, kind, body),
        )


def _lwd_describe_frame(direction: str, kind: bytes, body: bytes) -> dict[str, Any]:
    if kind == LwdFrameKind.LINK.value:
        msg: Any = lwd_decode_link_message(body)
        kind_name = "link"
    else:
        decode = (
            lwd_decode_edge_to_cloud if direction == "e2c" else lwd_decode_cloud_to_edge
        )
        msg = decode(body)
        kind_name = "payload"
    return {"kind": kind_name, "msg": type(msg).__name__, **msgspec.to_builtins(msg)}


def _lwd_topology(ctrl_port: int) -> dict[str, Any]:
    """single_instance with one rank per instance: edge rank 0, cloud rank 1."""
    return {
        "deployment": {
            "mode": 0,
            "scene": "single_instance",
            "hccl_world_size": 2,
            "edges_num": 1,
            "clouds_num": 1,
        },
        "feature_ctrl": {"enable_early_recv": True},
        "control_plane": {
            "hello_retry_interval_s": _HELLO_RETRY_INTERVAL_S,
            "handshake_timeout_s": _HANDSHAKE_TIMEOUT_S,
        },
        "edges": [{"id": 0, "dp": [{"dp_idx": 0, "addr": _LOCALHOST, "ranks": [0]}]}],
        "clouds": [
            {
                "id": 0,
                "dp": [
                    {
                        "dp_idx": 0,
                        "addr": _LOCALHOST,
                        "ranks": [1],
                        "ctrl_port": ctrl_port,
                    }
                ],
            }
        ],
        "instance_links": [{"edge": 0, "cloud": 0}],
    }


def _lwd_engine_args(
    role: str, topology_path: Path, limits: LwdSimLimits, master_port: int
) -> dict[str, Any]:
    worker = "LwdSimEdgeWorker" if role == "edge" else "LwdSimCloudWorker"
    return {
        "model": str(LWD_SIM_MODEL_DIR),
        "skip_tokenizer_init": True,
        "max_model_len": _MAX_MODEL_LEN,
        "max_num_batched_tokens": limits.max_num_batched_tokens,
        "max_num_seqs": limits.max_num_seqs,
        # The cloud scheduler does not support prefix caching.
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "disable_log_stats": True,
        "worker_cls": f"{_WORKERS_MODULE}.{worker}",
        # Edge rank 0 hosts the store of the shared world.
        "master_addr": _LOCALHOST,
        "master_port": master_port,
        "additional_config": {
            "lwd_config": {"path": str(topology_path), "role": role, "instance_id": 0}
        },
    }


def _lwd_driver_env(ascend_path: Path, trace_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    python_path = [str(_REPO_ROOT), str(ascend_path), env.get("PYTHONPATH", "")]
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(p for p in python_path if p),
            "VLLM_TARGET_DEVICE": "cpu",
            # An installed vllm-ascend must not take over the platform.
            "VLLM_PLUGINS": "",
            "VLLM_LWD_LOG_LEVEL": "INFO",
            LWD_SIM_TRACE_DIR_ENV: str(trace_dir),
        }
    )
    return env


def _lwd_start_driver(
    role: str, spec: dict[str, Any], work_dir: Path, env: dict[str, str]
) -> tuple[subprocess.Popen[bytes], Path]:
    spec_path = work_dir / f"{role}.json"
    spec_path.write_text(json.dumps(spec))
    log_path = work_dir / f"{role}.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", _DRIVER_MODULE, str(spec_path)],
            cwd=_REPO_ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            # Own process group, so a hang can be killed with its engine and
            # workers.
            start_new_session=True,
        )
    return process, log_path


def _lwd_wait(process: subprocess.Popen[bytes], timeout_s: float) -> int | None:
    try:
        return process.wait(timeout_s)
    except subprocess.TimeoutExpired:
        return None


def _lwd_kill_group(process: subprocess.Popen[bytes]) -> None:
    if sys.platform != "win32":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def lwd_run_sim(
    work_dir: Path,
    requests: list[LwdSimRequest],
    edge_limits: LwdSimLimits | None = None,
    cloud_limits: LwdSimLimits | None = None,
) -> LwdSimRun:
    """Start the cloud and the edge, run `requests` on the edge, stop both."""
    ascend_path = lwd_ascend_path()
    assert ascend_path is not None, f"{LWD_ASCEND_PATH_ENV} is not set"
    trace_dir = work_dir / "trace"
    trace_dir.mkdir()
    tap_port, cloud_port, master_port = get_open_ports_list(3)
    limits = {
        "edge": edge_limits or LwdSimLimits(),
        "cloud": cloud_limits or LwdSimLimits(),
    }
    ctrl_ports = {"edge": tap_port, "cloud": cloud_port}
    env = _lwd_driver_env(ascend_path, trace_dir)
    result_path = work_dir / "edge_results.json"
    specs = {}
    for role in ("cloud", "edge"):
        topology_path = work_dir / f"{role}_topology.yaml"
        topology_path.write_text(yaml.safe_dump(_lwd_topology(ctrl_ports[role])))
        specs[role] = {
            "role": role,
            "engine_args": _lwd_engine_args(
                role, topology_path, limits[role], master_port
            ),
            "requests": [vars(request) for request in requests],
            "result_path": str(result_path),
        }

    tap_trace = LwdSimTraceWriter("tap", str(trace_dir))
    cloud_endpoint = f"tcp://{_LOCALHOST}:{cloud_port}"
    processes: dict[str, subprocess.Popen[bytes]] = {}
    log_paths: dict[str, Path] = {}
    try:
        with LwdWireTap(tap_port, cloud_endpoint, tap_trace):
            for role in ("cloud", "edge"):
                processes[role], log_paths[role] = _lwd_start_driver(
                    role, specs[role], work_dir, env
                )
            edge_returncode = _lwd_wait(processes["edge"], _EDGE_TIMEOUT_S)
            cloud = processes["cloud"]
            cloud_alive_after_edge = cloud.poll() is None
            assert cloud.stdin is not None
            cloud.stdin.close()
            cloud_returncode = _lwd_wait(cloud, _CLOUD_EXIT_TIMEOUT_S)
    finally:
        for process in processes.values():
            _lwd_kill_group(process)
        tap_trace.close()

    edge_results = json.loads(result_path.read_text()) if result_path.exists() else []
    return LwdSimRun(
        returncodes={"edge": edge_returncode, "cloud": cloud_returncode},
        cloud_alive_after_edge=cloud_alive_after_edge,
        edge_results=edge_results,
        trace=LwdSimTrace(trace_dir),
        log_paths=log_paths,
    )
