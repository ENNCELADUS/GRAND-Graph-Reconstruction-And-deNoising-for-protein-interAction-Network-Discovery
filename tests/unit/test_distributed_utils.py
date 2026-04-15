"""Unit tests for distributed utility helpers."""

from __future__ import annotations

import pytest
import src.utils.distributed as distributed_module


def test_distributed_barrier_nccl_uses_local_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = distributed_module.DistributedContext(
        ddp_enabled=True,
        is_distributed=True,
        rank=1,
        local_rank=2,
        world_size=4,
    )
    observed: dict[str, object] = {}

    monkeypatch.setattr(distributed_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed_module.dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(distributed_module.torch.cuda, "is_available", lambda: True)

    def fake_barrier(*, device_ids: list[int]) -> None:
        observed["device_ids"] = device_ids

    monkeypatch.setattr(distributed_module.dist, "barrier", fake_barrier)

    distributed_module.distributed_barrier(context)

    assert observed["device_ids"] == [2]


def test_distributed_barrier_non_nccl_uses_plain_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = distributed_module.DistributedContext(
        ddp_enabled=True,
        is_distributed=True,
        rank=0,
        local_rank=0,
        world_size=2,
    )
    calls: list[str] = []

    monkeypatch.setattr(distributed_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed_module.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(distributed_module.torch.cuda, "is_available", lambda: False)

    def fake_barrier() -> None:
        calls.append("barrier")

    monkeypatch.setattr(distributed_module.dist, "barrier", fake_barrier)

    distributed_module.distributed_barrier(context)

    assert calls == ["barrier"]


def test_distributed_module_does_not_expose_deprecated_process_group_lifecycle() -> None:
    assert not hasattr(distributed_module, "initialize_distributed")
    assert not hasattr(distributed_module, "cleanup_distributed")
    assert "initialize_distributed" not in distributed_module.__all__
    assert "cleanup_distributed" not in distributed_module.__all__
