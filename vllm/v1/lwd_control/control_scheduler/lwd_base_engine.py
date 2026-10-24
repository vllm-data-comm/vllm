# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Common base of the LWD edge and cloud engine cores: scheduler injection,
the control plane of the DP rank and the hand-over of its IO thread events to
the engine thread."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import msgspec

from vllm.lwd_logger import lwd_logger
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.lwd_control.lwd_comm.lwd_control_plane import (
    LwdControlPlane,
    LwdControlSender,
)
from vllm.v1.lwd_control.lwd_comm.lwd_control_plane_factory import (
    lwd_create_control_plane,
)
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import LwdPeer
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LwdMessage,
    lwd_encode_message,
)


@dataclass(frozen=True)
class _LwdPeerUp:
    """`WAKEUP` payload: `peer` completed the control plane handshake."""

    peer: LwdPeer


@dataclass(frozen=True)
class _LwdControlPlaneFailed:
    """`EXECUTOR_FAILED` payload naming the control plane as the cause."""

    reason: str


class _LwdEngineListener:
    """Control plane callbacks (IO thread), forwarded to the engine."""

    def __init__(self, engine: LwdBaseEngineCore) -> None:
        self._engine = engine

    def on_peer_up(self, peer: LwdPeer) -> None:
        # The sender belongs to the engine thread, so registration is handled
        # there; input_queue FIFO puts it before any request from `peer`.
        self._engine.input_queue.put_nowait(
            (EngineCoreRequestType.WAKEUP, _LwdPeerUp(peer))
        )

    def on_message(self, peer: LwdPeer, payload: bytes) -> None:
        try:
            self._engine._lwd_on_payload(peer, payload)
        except Exception as e:
            # Keep the IO thread alive; the engine fails on its own thread.
            lwd_logger.exception("[comm] handling a payload from %s failed", peer)
            self._engine._lwd_fail(f"handling a payload from {peer}: {e!r}")

    def on_fatal(self, reason: str) -> None:
        # The plane has logged the crash.
        self._engine._lwd_fail(reason)


class LwdBaseEngineCore(EngineCoreProc):
    """EngineCoreProc of an LWD edge or cloud.

    Subclasses declare `lwd_scheduler_cls`, injected before `super()`, which
    builds the scheduler from `scheduler_cls` once. Upstream disables chunked
    prefill for deployments without KV cache groups, while LWD edge chunks are
    pure scheduling bookkeeping, so it is restored for LWD engines afterwards.
    `_lwd_assemble` then sets up the role and starts the control plane; READY
    has already reached the frontend by then.

    This class is the single-DP base. The seams live in the methods below, so
    a DP variant pairs the same ones with `DPEngineCoreProc` by listing this
    class first in its bases: every override here delegates through `super()`,
    which then reaches the DP implementation rather than the plain one.
    """

    lwd_scheduler_cls: type

    def __init__(self, *args: Any, lwd_dp_size: int, **kwargs: Any) -> None:
        self._lwd_dp_size = lwd_dp_size
        # One plane and sender per link this process holds. A cloud process
        # holds exactly one (its own DP); an edge process holds one per
        # cloud DP it serves. Indexed by the link's cloud DP index.
        self._lwd_control_planes: list[LwdControlPlane] = []
        self._lwd_senders: list[LwdControlSender] = []
        # has_work state at the last transition log; None = not logged yet.
        self._lwd_last_has_work_state: tuple[bool, bool, bool] | None = None
        vllm_config = kwargs["vllm_config"]
        scheduler_config = vllm_config.scheduler_config
        scheduler_config.scheduler_cls = self.lwd_scheduler_cls
        chunked_prefill_wanted = scheduler_config.enable_chunked_prefill
        super().__init__(*args, **kwargs)
        if not scheduler_config.enable_chunked_prefill and chunked_prefill_wanted:
            scheduler_config.enable_chunked_prefill = True
        self.lwd_config = vllm_config.lwd_config
        try:
            self._lwd_assemble()
        except BaseException:
            # run_engine_core never gets this engine to shut down, so release
            # the executor and the control plane before failing startup.
            self.shutdown()
            raise

    def _lwd_assemble(self) -> None:
        """Role setup once the engine is built, including the control plane."""
        raise NotImplementedError

    def _lwd_start_control_plane(self) -> LwdControlPlane:
        """Start this DP rank's single control plane; the calling (engine)
        thread owns its sender. Cloud roles, which have one link."""
        dp_rank = self.vllm_config.parallel_config.data_parallel_index
        return self._lwd_start_control_planes(dp_ranks=[dp_rank])[0]

    def _lwd_start_control_planes(
        self, dp_ranks: list[int] | None = None
    ) -> list[LwdControlPlane]:
        """Start one control plane per link; the engine thread owns the
        senders.

        Each link gets its own plane (own ZMQ context, IO thread and
        socket), so a stalled peer can only back up its own outbound queue.
        `dp_ranks` defaults to every cloud DP, which is what a shared edge
        process serves.
        """
        if dp_ranks is None:
            dp_ranks = list(range(self.lwd_config.cloud_dp_size))
        # Both sides of a link compare the cloud's DP count: it is the
        # number of links, and the handshake checks it. A cloud passes its
        # own vLLM DP size, which the factory also checks against the
        # topology; an edge has no such local value (its own DP size is
        # always 1) and takes the count from the topology.
        link_dp_size = (
            self._lwd_dp_size
            if self.lwd_config.role == "cloud"
            else self.lwd_config.cloud_dp_size
        )
        for dp_rank in dp_ranks:
            plane = lwd_create_control_plane(
                self.lwd_config, dp_rank, link_dp_size
            )
            self._lwd_control_planes.append(plane)
            self._lwd_senders.append(plane.start(_LwdEngineListener(self)))
            lwd_logger.info(
                "[comm] control plane dp%d/%d started on %s",
                dp_rank,
                self._lwd_dp_size,
                plane.endpoint,
            )
        return self._lwd_control_planes

    def _lwd_send(self, peer: LwdPeer, message: LwdMessage, link: int = 0) -> bool:
        """Queue `message` for `peer` on `link`; False when its outbound
        queue is full."""
        assert self._lwd_senders, "control plane not started"
        return self._lwd_senders[link].send(peer, lwd_encode_message(message))

    def _lwd_fail(self, reason: str) -> None:
        """Any thread: fail the engine on the engine thread, where
        run_engine_core logs the failure. Callers log the cause."""
        self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, _LwdControlPlaneFailed(reason))
        )

    # ------------------------------------------------------------------ #
    # IO thread message entry                                            #
    # ------------------------------------------------------------------ #
    def _lwd_on_payload(self, peer: LwdPeer, payload: bytes) -> None:
        """Common entry of received payloads: `_lwd_dispatch_message` handles
        them; malformed payloads and unknown types are dropped here with a
        warning."""
        try:
            msg = self._lwd_decode(payload)
        except (msgspec.DecodeError, msgspec.ValidationError) as e:
            lwd_logger.warning("[comm] drop malformed payload from %s: %s", peer, e)
            return
        if not self._lwd_dispatch_message(peer, msg):
            lwd_logger.warning(
                "[comm] drop unexpected frame %r from %s", type(msg), peer
            )

    def _lwd_decode(self, payload: bytes) -> LwdMessage:
        """Decode a payload sent by the peer role."""
        raise NotImplementedError

    def _lwd_dispatch_message(self, peer: LwdPeer, msg: LwdMessage) -> bool:
        """IO thread; must not block. True = handled, False = unknown."""
        raise NotImplementedError

    def _lwd_handle_peer_registered(self, peer: LwdPeer) -> None:
        """Engine thread: `peer` completed the handshake."""

    # ------------------------------------------------------------------ #
    # Engine interface overrides                                         #
    # ------------------------------------------------------------------ #
    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        if request_type == EngineCoreRequestType.WAKEUP and isinstance(
            request, _LwdPeerUp
        ):
            self._lwd_handle_peer_registered(request.peer)
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED and isinstance(
            request, _LwdControlPlaneFailed
        ):
            raise RuntimeError(f"[LWD] control plane failed: {request.reason}")
        else:
            super()._handle_client_request(request_type, request)

    def has_work(self) -> bool:
        """Native three-source check, logging its transitions (lwd_logger,
        level VLLM_LWD_LOG_LEVEL, silent by default)."""
        state = (
            self.engines_running,
            self.scheduler.has_requests(),
            bool(self.batch_queue),
        )
        if state != self._lwd_last_has_work_state:
            lwd_logger.info(
                "[has_work] engines_running=%s has_requests=%s batch_queue=%s",
                *state,
            )
            self._lwd_last_has_work_state = state
        return any(state)

    def shutdown(self) -> None:
        # Also reached after a failed assembly, before any plane exists.
        for plane in self._lwd_control_planes:
            plane.shutdown()
        super().shutdown()
