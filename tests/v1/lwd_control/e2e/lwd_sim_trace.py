# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Event trace shared by the processes of the LWD flow simulation.

Each process appends JSON lines to its own file in the directory named by
`VLLM_LWD_SIM_TRACE_DIR`. Timestamps come from the host-wide monotonic clock,
so events of different processes can be ordered when one causes the other.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LWD_SIM_TRACE_DIR_ENV = "VLLM_LWD_SIM_TRACE_DIR"


class LwdSimTraceWriter:
    """Appends the events of one process; safe to share between threads."""

    def __init__(self, source: str, directory: str | None = None) -> None:
        directory = directory or os.environ[LWD_SIM_TRACE_DIR_ENV]
        self._source = source
        self._file = open(Path(directory) / f"{source}.jsonl", "a")  # noqa: SIM115
        self._lock = threading.Lock()

    def record(self, event: str, t: int | None = None, **fields: Any) -> None:
        """`t` defaults to now; pass the time an event was caused at when the
        record is only written after the effect may have started."""
        t = time.monotonic_ns() if t is None else t
        line = json.dumps({"t": t, "src": self._source, "event": event, **fields})
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            self._file.close()


@dataclass(frozen=True)
class LwdSimEvent:
    t: int
    src: str
    event: str
    fields: dict[str, Any]

    def __getitem__(self, name: str) -> Any:
        return self.fields[name]


class LwdSimTrace:
    """All recorded events of one run, in timestamp order."""

    def __init__(self, directory: Path) -> None:
        events = []
        for path in sorted(directory.glob("*.jsonl")):
            for line in path.read_text().splitlines():
                raw = json.loads(line)
                events.append(
                    LwdSimEvent(raw.pop("t"), raw.pop("src"), raw.pop("event"), raw)
                )
        self.events = sorted(events, key=lambda event: event.t)

    def select(self, src: str, event: str) -> list[LwdSimEvent]:
        return [e for e in self.events if e.src == src and e.event == event]

    def frames(self, direction: str, kind: str) -> list[LwdSimEvent]:
        """Control-link frames the tap delivered in `direction` ("e2c"/"c2e")."""
        return [
            e
            for e in self.select("tap", "frame")
            if e["direction"] == direction and e["kind"] == kind and e["delivered"]
        ]

    def payloads(self, direction: str, msg: str | None = None) -> list[LwdSimEvent]:
        return [
            e
            for e in self.frames(direction, "payload")
            if msg is None or e["msg"] == msg
        ]

    def dump(self) -> str:
        return "\n".join(
            f"{e.t} {e.src:12} {e.event:16} {json.dumps(e.fields)}" for e in self.events
        )
