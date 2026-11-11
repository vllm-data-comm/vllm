# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager

import pytest
import torch


class _Event:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def record(self, stream=None) -> None:
        return None


class _Stream:
    def wait_event(self, event) -> None:
        return None

    def wait_stream(self, stream) -> None:
        return None


@contextmanager
def _use_stream(stream):
    yield stream


@pytest.fixture(autouse=True)
def cpu_device_boundaries(monkeypatch):
    if torch.cuda.is_available():
        yield
        return

    from vllm.platforms import current_platform

    stream = _Stream()
    monkeypatch.setattr(type(current_platform), "device_type", "cpu")
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    monkeypatch.setattr(torch.cuda, "Stream", lambda *args, **kwargs: _Stream())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args, **kwargs: stream)
    monkeypatch.setattr(torch.cuda, "stream", _use_stream)
    monkeypatch.setattr(torch.cuda, "set_stream", lambda stream: None)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.buffer_utils.is_uva_available", lambda: False
    )

    for name in ("empty", "zeros", "tensor"):
        original = getattr(torch, name)

        def allocate(*args, _original=original, **kwargs):
            kwargs.pop("pin_memory", None)
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, allocate)
    yield
