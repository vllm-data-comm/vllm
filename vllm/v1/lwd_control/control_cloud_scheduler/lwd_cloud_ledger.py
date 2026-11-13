# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request lifecycle bookkeeping of the cloud engine.

One :class:`LwdCloudLedger` owns every per-request and per-connection map the
cloud control plane keeps between a RequestNotify and the final C2E flush:
canonical routes, prompt lengths, prefix nonces, seqno tracking, retirement
tombstones and failed connections. The engine keeps only transport and
scheduling wiring; request bookkeeping transitions go through the ledger.

The ledger borrows the engine's scheduler because retirement must stay in
lockstep with scheduling state: arrival rows are dropped with the request,
and a tombstone outlives its request only while a pending C2E still needs
the original route.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.config.lwd import LwdConnectionId
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import LwdPeer

if TYPE_CHECKING:
    from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_scheduler import (
        LwdCloudScheduler,
    )

_LWD_RETIRED_ROUTE_CAP = 4096


class LwdCloudLedger:
    """Routes, nonces, seqnos and tombstones of one cloud engine."""

    def __init__(self, scheduler: LwdCloudScheduler) -> None:
        self._scheduler = scheduler
        self.prompt_lens: dict[str, int] = {}
        """Recorded RequestNotify prompt lengths, keyed by canonical id."""
        self.retired_prompt_lens: dict[str, int] = {}
        """Prompt lengths preserved across abort retirement for late hints."""
        self.request_nonces: dict[str, int] = {}
        """Prefix nonce of every live request generation."""
        self.retired_nonces: dict[str, int] = {}
        """Nonce kept until the retirement tombstone drains."""
        self.retired_req_ids: set[str] = set()
        """Requests retired while seqnos or pending C2E still reference them."""
        self.request_routes: dict[str, LwdPeer] = {}
        """Canonical request id -> originating edge peer."""
        self.retired_routes: dict[str, LwdPeer] = {}
        """Bounded route tombstones for DOWN metadata trailing retirement."""
        self.request_seqnos: dict[str, set[int]] = {}
        """Request id -> seqnos still pending on its connection."""
        self.pending_seqnos: dict[LwdConnectionId, dict[int, set[str]]] = {}
        """Connection -> seqno -> request ids of that announced range."""
        self.latest_seqnos: dict[LwdConnectionId, int] = {}
        """Newest accepted RangeNotify seqno per connection."""
        self.failed_connections: set[LwdConnectionId] = set()
        """Connections whose early-recv reported a failure."""
        self.prefix_hits: dict[str, int] = {}
        """Admitted prefix hit tokens per request (shared with the scheduler)."""

    def track_range(
        self,
        connection_id: LwdConnectionId,
        seqno: int,
        request_ids: list[str],
    ) -> None:
        """Accept one RangeNotify seqno and map it to its request ids."""
        self.latest_seqnos[connection_id] = seqno
        self.pending_seqnos.setdefault(connection_id, {})[seqno] = set(request_ids)
        for request_id in request_ids:
            self.request_seqnos.setdefault(request_id, set()).add(seqno)

    def complete_seqno(self, connection_id: LwdConnectionId, seqno: int) -> None:
        """Release one drained seqno; retire requests it was holding alive."""
        pending_by_seqno = self.pending_seqnos.get(connection_id)
        if pending_by_seqno is None:
            return
        request_ids = pending_by_seqno.pop(seqno, None)
        if not pending_by_seqno:
            self.pending_seqnos.pop(connection_id, None)
        if request_ids is None:
            return
        for request_id in request_ids:
            pending = self.request_seqnos.get(request_id)
            if pending is not None:
                pending.discard(seqno)
            if pending:
                continue
            self.request_seqnos.pop(request_id, None)
            if request_id in self.retired_req_ids:
                self.clear_retired(request_id)

    def retire_request(self, request_id: str, preserve_prompt: bool = False) -> None:
        """Retire one request until its seqnos and C2E batches have drained."""
        self._scheduler.lwd_recv_rows.pop(request_id, None)
        self.retired_nonces[request_id] = self.request_nonces.get(request_id, 0)
        if not self.request_seqnos.get(request_id):
            self.clear_retired(request_id)
            return
        self.retired_req_ids.add(request_id)
        if preserve_prompt:
            prompt_len = self.prompt_lens.get(request_id)
            if prompt_len is not None:
                self.retired_prompt_lens[request_id] = prompt_len

    def clear_retired(self, request_id: str) -> None:
        """Drop the retirement tombstone once nothing needs the route."""
        if any(request_id in notify.req_ids for notify in self._scheduler.pending_c2e):
            # A final or zero-accepted notification still needs the original
            # route. Keep the tombstone until post_step has flushed it.
            self.retired_req_ids.add(request_id)
            return
        route = self.request_routes.pop(request_id, None)
        if route is not None:
            # DOWN metadata can trail native request retirement by one or more
            # asynchronous steps. Keep a bounded tombstone for those batches.
            self.retired_routes[request_id] = route
            while len(self.retired_routes) > _LWD_RETIRED_ROUTE_CAP:
                self.retired_routes.pop(next(iter(self.retired_routes)))
        self.retired_req_ids.discard(request_id)
        self.retired_prompt_lens.pop(request_id, None)
        self.retired_nonces.pop(request_id, None)
        self.request_nonces.pop(request_id, None)
        self.request_seqnos.pop(request_id, None)
        self.prompt_lens.pop(request_id, None)
