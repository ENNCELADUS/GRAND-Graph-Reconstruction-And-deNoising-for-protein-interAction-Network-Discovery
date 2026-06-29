"""Distributed math + 2-rank gradient-equivalence tests for topology subset backward.

This is a UNIT-TEST harness, not the production distributed backend. Production runs
under Accelerate (see Task 9/10); here we open a minimal gloo group only to verify
that `_all_reduce_topology_gradients` over disjoint, globally-scaled shards reproduces
the single-process full-objective gradient.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from tccig.s2gae import (
    _all_reduce_topology_gradients,
    _shard_chunks_for_rank,
    _size_balanced_chunk_scales,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x


# A fixed toy "objective": each chunk i contributes scale_i * (weight * x_i).sum().
# Chunk losses are linear in the parameter so the gradient is exactly
# Σ_i scale_i * x_i regardless of how chunks are partitioned across ranks.
_NODE_SIZES = [20, 20, 20, 200, 200]
_CHUNK_INPUTS = [
    torch.tensor([1.0, 2.0]),
    torch.tensor([3.0]),
    torch.tensor([0.5, 0.5, 0.5]),
    torch.tensor([4.0]),
    torch.tensor([1.5, 2.5]),
]


def _reference_full_grad() -> torch.Tensor:
    """Single-process gradient over the FULL chunk list (the ground truth)."""
    model = TinyModel()
    scales = _size_balanced_chunk_scales(_NODE_SIZES)
    model.zero_grad(set_to_none=True)
    for chunk_input, scale in zip(_CHUNK_INPUTS, scales, strict=True):
        loss = scale * model(chunk_input).sum()
        loss.backward()
    assert model.weight.grad is not None
    return model.weight.grad.detach().clone()


def _worker(rank: int, world_size: int, file_path: str, grad_out_path: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{file_path}",
        world_size=world_size,
        rank=rank,
    )
    try:
        model = TinyModel()
        model.zero_grad(set_to_none=True)
        shard = _shard_chunks_for_rank(
            node_sizes=_NODE_SIZES, rank=rank, world_size=world_size
        )
        for global_index, scale in shard:
            loss = scale * model(_CHUNK_INPUTS[global_index]).sum()
            loss.backward()
        runtime = type("Runtime", (), {"is_distributed": True})()
        _all_reduce_topology_gradients(model, runtime)  # type: ignore[arg-type]
        if rank == 0:
            # Pass the result back through a file, not an mp.Manager proxy: the Manager
            # proxy deadlocks with the spawn start-method (macOS default) under pytest.
            torch.save(model.weight.grad.detach().clone(), grad_out_path)
    finally:
        dist.destroy_process_group()


def test_size_balanced_scales_match_full_objective() -> None:
    sizes = [20, 20, 200, 200]
    losses = torch.tensor([1.0, 3.0, 10.0, 14.0])
    scales = torch.tensor(_size_balanced_chunk_scales(sizes))
    chunked = (losses * scales).sum()
    full = torch.tensor([(1.0 + 3.0) / 2.0, (10.0 + 14.0) / 2.0]).mean()
    assert chunked == full


def test_all_reduce_topology_gradients_noops_when_not_distributed() -> None:
    model = TinyModel()
    loss = model(torch.tensor([2.0])).sum()
    loss.backward()
    before = model.weight.grad.detach().clone()
    runtime = type("Runtime", (), {"is_distributed": False})()
    _all_reduce_topology_gradients(model, runtime)  # type: ignore[arg-type]
    assert torch.equal(model.weight.grad, before)


@pytest.mark.parametrize("world_size", [2])
def test_two_rank_sharded_backward_matches_single_process(
    world_size: int, tmp_path
) -> None:
    """Spec §12: fork (b) sharded backward + SUM all-reduce == single-process full grad."""
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend unavailable")
    reference = _reference_full_grad()
    rendezvous = str(tmp_path / "rendezvous")
    grad_out_path = str(tmp_path / "rank0_grad.pt")
    mp.spawn(
        _worker,
        args=(world_size, rendezvous, grad_out_path),
        nprocs=world_size,
        join=True,
    )
    # Rank 0 wrote its gradient to a file (not an mp.Manager proxy, which deadlocks
    # with the spawn start-method on macOS under pytest).
    assert os.path.exists(grad_out_path), "rank 0 did not report a gradient"
    result = torch.load(grad_out_path)
    # SUM all-reduce over disjoint, globally-scaled shards must equal the full-objective
    # reference with NO world_size factor. A world_size double-count would make this 2x.
    torch.testing.assert_close(result, reference)
