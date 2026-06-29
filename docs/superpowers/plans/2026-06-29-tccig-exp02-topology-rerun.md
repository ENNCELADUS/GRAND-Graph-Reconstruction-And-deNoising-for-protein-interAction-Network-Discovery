# TCCIG Exp02 Topology Re-run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Experiment 02 topology re-run design: bounded subset topology training with three-stage negative sampling, inverse-probability weighting, per-size aggregation, chunked DDP topology backward, observable scoring, and smoke-test protection.

**Architecture:** Add focused topology subset planning utilities under `tccig/`, extend existing pairwise topology losses to accept per-pair IPW weights, and replace the current full-plan topology pass in `tccig/s2gae.py` with a sharded per-chunk backward step. Keep validation/test metrics unchanged and make the new training path opt-in through `refiner.topology_training`.

**Tech Stack:** Python 3.11, PyTorch, Accelerate/DDP, NetworkX, pytest, uv, ruff.

---

## Scope Check

This plan covers one subsystem: the standalone TCCIG Experiment 02 topology-training path in `/Users/richardwang/Documents/grand`. It deliberately does not change the frozen v3.1 scorer model, official validation/test topology metrics, S2GAE encoder/decoder architecture, or unrelated `src/pipeline/stages/topology_finetune.py` behavior.

**Out of scope — batch embedding loads (spec §7, review Finding 7).** The spec lists
"batch embedding loads to avoid per-endpoint `torch.load` thrash" as a scoring-cost
optimization. The current scorer (`_collate_pair_score_batch`, `tccig/train.py:801`)
already batches at the DataLoader level but calls `load_cached_embedding(...)` once per
endpoint per batch, with no cross-batch embedding cache. This rerun does **not** add a
batch-load/embedding-cache layer, and that is deliberate: the whole point of the bounded
candidate frame (spec §7, Task 3/6) is that scoring drops from the ~12.79M full candidate
space to all-positives + `candidate_ratio`×positives per capped subgraph — typically a
few hundred thousand pairs at most. At that size the per-endpoint `torch.load` cost is
not the bottleneck (scoring is dominated by the forward pass and the one-time cache
write), so a batch-load rewrite would add cache-invalidation surface for negligible
wall-clock gain. If a future unbounded-scoring run reintroduces the thrash, batch loading
should be a separate task with its own cache-correctness tests; it is not required for
this bounded rerun to be correct or observably fast. Progress logging (Task 12) still
makes the scoring phase non-silent regardless.

## File Structure

- Create `tccig/topology_subset.py`: dataclasses and deterministic sampling logic for bounded candidate negatives, cached pool records, per-epoch subset records, active-size filtering, per-size budgets, and three-stage inclusion probabilities.
- Modify `src/topology/finetune_losses.py`: add optional `pair_weights` to pairwise density, graph-similarity, degree-MMD helpers, and `compute_topology_losses`.
- Modify `tccig/train.py`: build `TopologySubsetPlan` instead of full train-topology all-pairs plan when `topology_training.subset.enabled=true`; score only bounded candidate/pool pairs; log score estimates and progress.
- Modify `tccig/s2gae.py`: parse new config knobs; consume `TopologySubsetPlan`; sample an epoch subset; run per-size, per-chunk topology backward; all-reduce gradients manually for fork (b).
- Modify `src/topology/plan_cache.py`: persist subset-plan payloads and include new sampler parameters in cache metadata.
- Add `configs/tccig/02_balanced_subset.yaml`: main rerun config with `run_id: 02_balanced_subset`.
- Add `configs/tccig/02_balanced_subset_smoke.yaml`: tiny smoke config that enters a positive-scale topology step in epoch 1.
- Add tests:
  - `tests/unit/test_tccig_topology_subset.py`
  - Extend `tests/unit/test_tccig_topology_training.py`
  - Extend `tests/unit/test_topology_finetune.py`
  - Add `tests/unit/test_tccig_topology_distributed.py`

## Implementation Rules

- Use `uv run --locked --no-sync --offline` for repo Python commands.
- Use `rtk` before shell commands in this repo.
- Keep `clustering` off for the **training** objective (spec §3.5). The subset chunk
  loss (Task 8) hardcodes `include_clustering_mmd=False`, so training clustering is off
  regardless of config. **Do NOT** set `topology_validation.compute_clustering_mmd: false`
  to achieve this (review Finding 11): that single knob also drives the validation/test
  metric definitions (`s2gae.py:1340/1350`), and disabling it there would change the
  metric definitions — a spec non-goal (spec §2). Leave `compute_clustering_mmd: true`.
- Keep validation/test full-space metrics unchanged (including clustering).
- Commit after each task that passes its test block.
- Do not reuse run id `02` for the rerun.

---

### Task 1: Add Subset Sampling Data Contracts

**Files:**
- Create: `tccig/topology_subset.py`
- Test: `tests/unit/test_tccig_topology_subset.py`

- [ ] **Step 1: Write failing tests for record validation and active-size filtering**

Create `tests/unit/test_tccig_topology_subset.py`:

```python
"""Tests for bounded TCCIG topology subset sampling."""

from __future__ import annotations

import pytest

from tccig.topology_subset import (
    SamplingStratum,
    TopologyPairSample,
    TopologySubsetSamplerConfig,
    active_node_sizes,
)


def test_pair_sample_validates_three_stage_probability_product() -> None:
    sample = TopologyPairSample(
        pair_id="a||b",
        subgraph_id="size=4:index=0",
        node_size=4,
        protein_a="a",
        protein_b="b",
        local_index_a=0,
        local_index_b=1,
        stratum=SamplingStratum.UNIFORM_NEGATIVE,
        pi_cand=0.5,
        pi_pool_given_cand=0.4,
        pi_epoch_given_pool=0.25,
        pi_total=0.05,
        target=0.0,
        scorer_probability=0.2,
    )
    sample.validate()


def test_pair_sample_rejects_bad_probability_product() -> None:
    sample = TopologyPairSample(
        pair_id="a||b",
        subgraph_id="size=4:index=0",
        node_size=4,
        protein_a="a",
        protein_b="b",
        local_index_a=0,
        local_index_b=1,
        stratum=SamplingStratum.HARD_NEGATIVE,
        pi_cand=0.5,
        pi_pool_given_cand=0.5,
        pi_epoch_given_pool=0.5,
        pi_total=0.5,
        target=0.0,
        scorer_probability=0.9,
    )
    with pytest.raises(ValueError, match="pi_total must equal"):
        sample.validate()


def test_positive_sample_must_have_probability_one() -> None:
    sample = TopologyPairSample(
        pair_id="a||b",
        subgraph_id="size=4:index=0",
        node_size=4,
        protein_a="a",
        protein_b="b",
        local_index_a=0,
        local_index_b=1,
        stratum=SamplingStratum.POSITIVE,
        pi_cand=1.0,
        pi_pool_given_cand=1.0,
        pi_epoch_given_pool=0.5,
        pi_total=0.5,
        target=1.0,
        scorer_probability=0.8,
    )
    with pytest.raises(ValueError, match="positive samples must have"):
        sample.validate()


def test_active_node_sizes_drop_zero_budget_sizes() -> None:
    active, skipped = active_node_sizes(
        node_sizes=(20, 40, 80),
        graph_node_count=60,
        subgraphs_per_size={20: 3, 40: 1, 80: 2},
        labeled_pairs_per_size={20: 100, 40: 0, 80: 200},
    )
    assert active == (20,)
    assert skipped == {40: "zero_labeled_pair_budget", 80: "larger_than_graph"}


def test_sampler_config_defaults_match_rerun_decision() -> None:
    cfg = TopologySubsetSamplerConfig()
    assert cfg.enabled is True
    assert cfg.candidate_ratio == 20
    assert cfg.pool_ratio == 10
    assert cfg.epoch_ratio == 5
    assert cfg.hard_fraction == pytest.approx(0.5)
    assert cfg.uniform_fraction == pytest.approx(0.5)
    assert cfg.hard_stratum_fraction == pytest.approx(0.2)
    # Per-size budget (spec §4): default unbounded (0).
    assert cfg.max_subgraphs_per_size == 0
    assert cfg.max_labeled_pairs_per_size == 0


def test_sampler_config_rejects_negative_budget() -> None:
    with pytest.raises(ValueError, match="max_subgraphs_per_size must be >= 0"):
        TopologySubsetSamplerConfig(max_subgraphs_per_size=-1).validate()
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_subset.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tccig.topology_subset'`.

- [ ] **Step 3: Implement the minimal data contracts**

Create `tccig/topology_subset.py`:

```python
"""Bounded subset sampling contracts for TCCIG topology training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isclose


class SamplingStratum(StrEnum):
    """Topology subset sampling strata."""

    POSITIVE = "positive"
    HARD_NEGATIVE = "hard_negative"
    UNIFORM_NEGATIVE = "uniform_negative"


@dataclass(frozen=True)
class TopologySubsetSamplerConfig:
    """Configuration for bounded candidate -> pool -> epoch subset sampling."""

    enabled: bool = True
    candidate_ratio: int = 20
    pool_ratio: int = 10
    epoch_ratio: int = 5
    hard_fraction: float = 0.5
    uniform_fraction: float = 0.5
    hard_stratum_fraction: float = 0.2
    seed: int = 0
    # Per-size budget (spec §4). 0 == unbounded. Caps stop the 200-node bucket from
    # dominating memory/objective and bound the labeled-pair count scored per size.
    max_subgraphs_per_size: int = 0
    max_labeled_pairs_per_size: int = 0
    # Bias diagnostic (spec §3.7 / §9). 0 == off; spec §3.7 proposes 5. The max node
    # size caps the subgraph the diagnostic decodes so the full n*(n-1) space is cheap.
    # bias_diagnostic_max_subgraphs bounds how many eligible subgraphs the production
    # diagnostic samples across the size mixture (spec §3.7 asks for "a few capped
    # validation subgraphs"), so bias from larger sizes is also exposed, not just the
    # single smallest one. 0 == every eligible subgraph.
    bias_diagnostic_every_n_epochs: int = 0
    bias_diagnostic_max_node_size: int = 40
    bias_diagnostic_max_subgraphs: int = 4

    def validate(self) -> None:
        """Validate ratio and fraction constraints."""
        for name, value in (
            ("candidate_ratio", self.candidate_ratio),
            ("pool_ratio", self.pool_ratio),
            ("epoch_ratio", self.epoch_ratio),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in (
            ("max_subgraphs_per_size", self.max_subgraphs_per_size),
            ("max_labeled_pairs_per_size", self.max_labeled_pairs_per_size),
            ("bias_diagnostic_every_n_epochs", self.bias_diagnostic_every_n_epochs),
            ("bias_diagnostic_max_node_size", self.bias_diagnostic_max_node_size),
            ("bias_diagnostic_max_subgraphs", self.bias_diagnostic_max_subgraphs),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0 (0 == unbounded)")
        for name, value in (
            ("hard_fraction", self.hard_fraction),
            ("uniform_fraction", self.uniform_fraction),
            ("hard_stratum_fraction", self.hard_stratum_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not isclose(self.hard_fraction + self.uniform_fraction, 1.0, abs_tol=1.0e-8):
            raise ValueError("hard_fraction + uniform_fraction must equal 1.0")
        if self.pool_ratio > self.candidate_ratio:
            raise ValueError("pool_ratio must be <= candidate_ratio")
        if self.epoch_ratio > self.pool_ratio:
            raise ValueError("epoch_ratio must be <= pool_ratio")


@dataclass(frozen=True)
class TopologyPairSample:
    """One pair selected for a topology subset subgraph in one epoch."""

    pair_id: str
    subgraph_id: str
    node_size: int
    protein_a: str
    protein_b: str
    local_index_a: int
    local_index_b: int
    stratum: SamplingStratum
    pi_cand: float
    pi_pool_given_cand: float
    pi_epoch_given_pool: float
    pi_total: float
    target: float
    scorer_probability: float

    def validate(self) -> None:
        """Validate sampling probabilities and target semantics."""
        for name, value in (
            ("pi_cand", self.pi_cand),
            ("pi_pool_given_cand", self.pi_pool_given_cand),
            ("pi_epoch_given_pool", self.pi_epoch_given_pool),
            ("pi_total", self.pi_total),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        expected = self.pi_cand * self.pi_pool_given_cand * self.pi_epoch_given_pool
        if not isclose(self.pi_total, expected, rel_tol=1.0e-7, abs_tol=1.0e-9):
            raise ValueError("pi_total must equal pi_cand * pi_pool_given_cand * pi_epoch_given_pool")
        if self.stratum is SamplingStratum.POSITIVE:
            if self.target != 1.0:
                raise ValueError("positive samples must have target=1.0")
            if (self.pi_cand, self.pi_pool_given_cand, self.pi_epoch_given_pool, self.pi_total) != (
                1.0,
                1.0,
                1.0,
                1.0,
            ):
                raise ValueError("positive samples must have all inclusion probabilities equal to 1")


def canonical_pair_id(protein_a: str, protein_b: str) -> str:
    """Return a stable undirected pair id."""
    left, right = sorted((protein_a, protein_b))
    return f"{left}||{right}"


def active_node_sizes(
    *,
    node_sizes: tuple[int, ...],
    graph_node_count: int,
    subgraphs_per_size: dict[int, int],
    labeled_pairs_per_size: dict[int, int],
) -> tuple[tuple[int, ...], dict[int, str]]:
    """Return globally active node sizes and skipped-size reasons."""
    active: list[int] = []
    skipped: dict[int, str] = {}
    for size in node_sizes:
        if size > graph_node_count:
            skipped[size] = "larger_than_graph"
            continue
        if subgraphs_per_size.get(size, 0) <= 0:
            skipped[size] = "zero_subgraph_budget"
            continue
        if labeled_pairs_per_size.get(size, 0) <= 0:
            skipped[size] = "zero_labeled_pair_budget"
            continue
        active.append(size)
    if not active:
        raise ValueError("topology subset sampling has no active node sizes")
    return tuple(active), skipped
```

- [ ] **Step 4: Run the tests and verify they pass**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_subset.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add tccig/topology_subset.py tests/unit/test_tccig_topology_subset.py
rtk git commit -m "feat: add topology subset sampling contracts"
```

---

### Task 2: Add Weighted Pairwise Topology Losses

**Files:**
- Modify: `src/topology/finetune_losses.py`
- Test: `tests/unit/test_topology_finetune.py`

- [ ] **Step 1: Add failing tests for IPW density and degree weighting**

The spec (§3, §12 "unit (loss)") requires the IPW-weighted **linear** accumulators
(density numerator, degree scatter-add sums) to *recover the full-space statistic
in expectation*, not merely to differ from the unweighted version. The first test
below asserts that unbiasedness directly on the linear pieces; the later tests pin
the wiring through `compute_topology_losses`.

Append to `tests/unit/test_topology_finetune.py`:

```python
import random as _random


def test_ipw_weighted_density_numerator_recovers_full_space_sum() -> None:
    # Full-space "graph": 6 upper-triangle pairs with known predicted probabilities.
    # Pair i is included with probability pi_i; weighted sum w_i * pred_i over the
    # included subset is a Horvitz-Thompson estimator of the full-space sum.
    pred_full = torch.tensor([0.9, 0.1, 0.7, 0.3, 0.5, 0.2], dtype=torch.float64)
    inclusion = torch.tensor([1.0, 0.5, 0.5, 0.25, 0.25, 0.5], dtype=torch.float64)
    full_space_sum = float(pred_full.sum().item())
    rng = _random.Random(0)
    trials = 20000
    running = 0.0
    for _ in range(trials):
        included_pred: list[float] = []
        weights_used: list[float] = []
        for value, pi in zip(pred_full.tolist(), inclusion.tolist()):
            if rng.random() < pi:
                included_pred.append(value)
                weights_used.append(1.0 / pi)
        if not included_pred:
            continue
        pred_t = torch.tensor(included_pred, dtype=torch.float64)
        weight_t = torch.tensor(weights_used, dtype=torch.float64)
        running += float((pred_t * weight_t).sum().item())
    estimate = running / trials
    # Horvitz-Thompson is unbiased for the linear sum; Monte-Carlo tolerance ~1%.
    assert estimate == pytest.approx(full_space_sum, rel=0.02)


def test_pairwise_soft_degrees_ipw_recovers_full_space_degrees() -> None:
    # Triangle a-b-c plus pair a-c; weight each included pair by 1/pi.
    num_nodes = 3
    pair_index_a = torch.tensor([0, 0, 1])
    pair_index_b = torch.tensor([1, 2, 2])
    pred = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    # Suppose the middle pair (0,2) was sampled with pi=0.5; others fully observed.
    pair_weights = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)
    weighted = finetune_losses_module._pairwise_soft_degrees(
        num_nodes=num_nodes,
        pair_index_a=pair_index_a,
        pair_index_b=pair_index_b,
        pair_probabilities=pred,
        pair_weights=pair_weights,
    )
    # Full-space degrees with all three edges present: deg(a)=2, deg(b)=2, deg(c)=2.
    # Reweighting the down-sampled (0,2) pair by 2x restores the full-space degree.
    assert weighted.tolist() == pytest.approx([2.0, 2.0, 2.0])


def test_pairwise_relative_density_uses_pair_weights() -> None:
    weights = finetune_losses_module.TopologyLossWeights(
        alpha=0.0,
        beta=1.0,
        gamma=0.0,
        delta=0.0,
        rd_loss_form="squared_ratio",
    )
    pred = torch.tensor([0.8, 0.1], dtype=torch.float32)
    target = torch.tensor([1.0, 0.0], dtype=torch.float32)
    pair_weights = torch.tensor([1.0, 4.0], dtype=torch.float32)
    weighted = finetune_losses_module.compute_topology_losses(
        weights=weights,
        num_nodes=3,
        pair_index_a=torch.tensor([0, 1]),
        pair_index_b=torch.tensor([1, 2]),
        pred_pair_probabilities=pred,
        target_pair_probabilities=target,
        pair_weights=pair_weights,
        include_clustering_mmd=False,
    )
    unweighted = finetune_losses_module.compute_topology_losses(
        weights=weights,
        num_nodes=3,
        pair_index_a=torch.tensor([0, 1]),
        pair_index_b=torch.tensor([1, 2]),
        pred_pair_probabilities=pred,
        target_pair_probabilities=target,
        include_clustering_mmd=False,
    )
    assert weighted["relative_density"] != unweighted["relative_density"]


def test_pairwise_degree_mmd_uses_pair_weights() -> None:
    weights = finetune_losses_module.TopologyLossWeights(
        alpha=0.0,
        beta=0.0,
        gamma=1.0,
        delta=0.0,
        histogram_sigma=1.0,
        degree_bins=8,
    )
    pred = torch.tensor([0.2, 0.9, 0.1], dtype=torch.float32)
    target = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    pair_weights = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float32)
    weighted = finetune_losses_module.compute_topology_losses(
        weights=weights,
        num_nodes=3,
        pair_index_a=torch.tensor([0, 0, 1]),
        pair_index_b=torch.tensor([1, 2, 2]),
        pred_pair_probabilities=pred,
        target_pair_probabilities=target,
        pair_weights=pair_weights,
        include_clustering_mmd=False,
    )
    unweighted = finetune_losses_module.compute_topology_losses(
        weights=weights,
        num_nodes=3,
        pair_index_a=torch.tensor([0, 0, 1]),
        pair_index_b=torch.tensor([1, 2, 2]),
        pred_pair_probabilities=pred,
        target_pair_probabilities=target,
        include_clustering_mmd=False,
    )
    assert weighted["degree_mmd"] != unweighted["degree_mmd"]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_topology_finetune.py::test_pairwise_relative_density_uses_pair_weights tests/unit/test_topology_finetune.py::test_pairwise_degree_mmd_uses_pair_weights -v
```

Expected: FAIL with `TypeError: compute_topology_losses() got an unexpected keyword argument 'pair_weights'`.

- [ ] **Step 3: Implement `pair_weights` in pairwise loss path**

Modify `src/topology/finetune_losses.py`:

```python
def _apply_pair_weights(values: torch.Tensor, pair_weights: torch.Tensor | None) -> torch.Tensor:
    """Apply optional inverse-probability pair weights."""
    if pair_weights is None:
        return values
    if pair_weights.shape != values.shape:
        raise ValueError("pair_weights must match pair probability shape")
    return values * pair_weights.to(device=values.device, dtype=values.dtype)
```

Update `_pairwise_graph_similarity_loss`:

```python
def _pairwise_graph_similarity_loss(
    *,
    pred_pair_probabilities: torch.Tensor,
    target_pair_probabilities: torch.Tensor,
    pair_weights: torch.Tensor | None = None,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Differentiable graph similarity loss over upper-triangle pair vectors."""
    weighted_pred = _apply_pair_weights(pred_pair_probabilities, pair_weights)
    weighted_target = _apply_pair_weights(target_pair_probabilities, pair_weights)
    difference = torch.abs(weighted_pred - weighted_target).sum()
    denominator = weighted_pred.sum() + weighted_target.sum()
    return torch.where(
        denominator > eps,
        difference / (denominator + eps),
        torch.zeros_like(difference),
    )
```

Update `_pairwise_relative_density_loss`:

```python
def _pairwise_relative_density_loss(
    *,
    num_nodes: int,
    pred_pair_probabilities: torch.Tensor,
    target_pair_probabilities: torch.Tensor,
    pair_weights: torch.Tensor | None = None,
    loss_form: str = "log_ratio_huber",
    eps: float = EPSILON,
) -> torch.Tensor:
    """Squared deviation of relative density computed from pair vectors."""
    if num_nodes < 2:
        raise ValueError("num_nodes must be >= 2")
    normalizer = float(num_nodes * (num_nodes - 1))
    weighted_pred = _apply_pair_weights(pred_pair_probabilities, pair_weights)
    weighted_target = _apply_pair_weights(target_pair_probabilities, pair_weights)
    pred_density = (2.0 * weighted_pred.sum()) / normalizer
    target_density = (2.0 * weighted_target.sum()) / normalizer
    if float(target_density.detach().item()) <= eps:
        return pred_density.square()
    return _relative_density_penalty(
        pred_density=pred_density,
        target_density=target_density,
        loss_form=loss_form,
        eps=eps,
    )
```

Update `_pairwise_soft_degrees`:

```python
def _pairwise_soft_degrees(
    *,
    num_nodes: int,
    pair_index_a: torch.Tensor,
    pair_index_b: torch.Tensor,
    pair_probabilities: torch.Tensor,
    pair_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return soft node degrees from upper-triangle pair probabilities."""
    weighted = _apply_pair_weights(pair_probabilities, pair_weights)
    degrees = weighted.new_zeros((num_nodes,))
    degrees.scatter_add_(0, pair_index_a, weighted)
    degrees.scatter_add_(0, pair_index_b, weighted)
    return degrees
```

Update `_degree_distribution_mmd_from_pairs` and `compute_topology_losses` to thread `pair_weights` through the three pairwise helpers:

```python
def _degree_distribution_mmd_from_pairs(
    *,
    num_nodes: int,
    pair_index_a: torch.Tensor,
    pair_index_b: torch.Tensor,
    pred_pair_probabilities: torch.Tensor,
    target_pair_probabilities: torch.Tensor,
    weights: TopologyLossWeights,
    pair_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """MMD between soft degree distributions built from pair vectors."""
    centers = torch.linspace(
        0.0,
        float(max(1, num_nodes - 1)),
        steps=max(2, weights.degree_bins),
        device=pred_pair_probabilities.device,
        dtype=pred_pair_probabilities.dtype,
    )
    pred_histogram = _soft_histogram(
        _pairwise_soft_degrees(
            num_nodes=num_nodes,
            pair_index_a=pair_index_a,
            pair_index_b=pair_index_b,
            pair_probabilities=pred_pair_probabilities,
            pair_weights=pair_weights,
        ),
        centers=centers,
        sigma=weights.histogram_sigma,
    )
    target_histogram = _soft_histogram(
        _pairwise_soft_degrees(
            num_nodes=num_nodes,
            pair_index_a=pair_index_a,
            pair_index_b=pair_index_b,
            pair_probabilities=target_pair_probabilities,
            pair_weights=pair_weights,
        ),
        centers=centers,
        sigma=weights.histogram_sigma,
    )
    return _soft_histogram_mmd(
        pred_histogram=pred_histogram,
        target_histogram=target_histogram,
        sigma=weights.histogram_sigma,
    )
```

Add `pair_weights: torch.Tensor | None = None` to `compute_topology_losses(...)`, and pass it into `_pairwise_graph_similarity_loss`, `_pairwise_relative_density_loss`, and `_degree_distribution_mmd_from_pairs`.

- [ ] **Step 4: Run focused and existing topology tests**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_topology_finetune.py::test_pairwise_relative_density_uses_pair_weights tests/unit/test_topology_finetune.py::test_pairwise_degree_mmd_uses_pair_weights tests/unit/test_tccig_topology_training.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/topology/finetune_losses.py tests/unit/test_topology_finetune.py
rtk git commit -m "feat: weight pairwise topology losses"
```

---

### Task 3: Implement Three-Stage Negative Candidate and Pool Sampling

**Files:**
- Modify: `tccig/topology_subset.py`
- Test: `tests/unit/test_tccig_topology_subset.py`

- [ ] **Step 1: Add failing sampler tests**

Append to `tests/unit/test_tccig_topology_subset.py`:

```python
import networkx as nx

from tccig.topology_subset import build_topology_subset_plan, sample_epoch_topology_subset


def _toy_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "c", "d", "e"])
    graph.add_edges_from([("a", "b"), ("b", "c")])
    return graph


def test_build_subset_plan_scores_only_candidate_frame() -> None:
    graph = _toy_graph()
    sampled = {4: [("a", "b", "c", "d")]}
    cfg = TopologySubsetSamplerConfig(
        candidate_ratio=2,
        pool_ratio=1,
        epoch_ratio=1,
        hard_fraction=0.5,
        uniform_fraction=0.5,
        hard_stratum_fraction=0.5,
        seed=3,
    )
    plan = build_topology_subset_plan(
        graph=graph,
        sampled_subgraphs=sampled,
        config=cfg,
        scorer_probabilities={},
    )
    assert plan.total_positive_pairs == 2
    assert plan.total_candidate_negatives <= 4
    assert plan.total_candidate_negatives < 4 * 3 // 2


def test_epoch_subset_keeps_all_positives_and_samples_negatives_with_pi() -> None:
    graph = _toy_graph()
    sampled = {4: [("a", "b", "c", "d")]}
    cfg = TopologySubsetSamplerConfig(
        candidate_ratio=3,
        pool_ratio=2,
        epoch_ratio=1,
        hard_fraction=0.5,
        uniform_fraction=0.5,
        hard_stratum_fraction=0.5,
        seed=4,
    )
    plan = build_topology_subset_plan(
        graph=graph,
        sampled_subgraphs=sampled,
        config=cfg,
        scorer_probabilities={"a||c": 0.9, "a||d": 0.2, "b||d": 0.7, "c||d": 0.1},
    )
    samples = sample_epoch_topology_subset(plan=plan, epoch=1)
    positives = [sample for sample in samples if sample.stratum is SamplingStratum.POSITIVE]
    negatives = [sample for sample in samples if sample.stratum is not SamplingStratum.POSITIVE]
    assert {sample.pair_id for sample in positives} == {"a||b", "b||c"}
    assert negatives
    assert all(0.0 < sample.pi_total <= 1.0 for sample in negatives)
    assert all(sample.target == 0.0 for sample in negatives)


def test_builder_trusts_pre_budgeted_subgraph_counts() -> None:
    # Spec §4 subgraph budgeting happens UPSTREAM (apply_per_size_subgraph_budget,
    # Task 6), not in the builder — so the builder must NOT silently drop tail
    # subgraphs (that would discard coverage-augmentation buckets). Given 3 size-4
    # subgraphs, the builder builds all 3 regardless of max_subgraphs_per_size.
    graph = _toy_graph()
    sampled = {4: [("a", "b", "c", "d"), ("a", "b", "c", "e"), ("a", "b", "d", "e")]}
    cfg = TopologySubsetSamplerConfig(
        candidate_ratio=2,
        pool_ratio=1,
        epoch_ratio=1,
        hard_stratum_fraction=0.5,
        seed=3,
        max_subgraphs_per_size=1,  # honored upstream, NOT by the builder
    )
    plan = build_topology_subset_plan(
        graph=graph,
        sampled_subgraphs=sampled,
        config=cfg,
        scorer_probabilities={},
    )
    assert len([sg for sg in plan.subgraphs if sg.node_size == 4]) == 3


def test_max_labeled_pairs_per_size_caps_scored_candidates() -> None:
    # Spec §4: cap the labeled (scored) negatives per size. The cap bounds candidate
    # negatives without breaking the pi_cand inclusion-ratio bookkeeping.
    graph = _toy_graph()
    sampled = {4: [("a", "b", "c", "d"), ("a", "b", "c", "e")]}
    capped = TopologySubsetSamplerConfig(
        candidate_ratio=10,
        pool_ratio=1,
        epoch_ratio=1,
        hard_stratum_fraction=0.5,
        seed=3,
        max_labeled_pairs_per_size=3,
    )
    uncapped = TopologySubsetSamplerConfig(
        candidate_ratio=10,
        pool_ratio=1,
        epoch_ratio=1,
        hard_stratum_fraction=0.5,
        seed=3,
    )
    capped_plan = build_topology_subset_plan(
        graph=graph, sampled_subgraphs=sampled, config=capped, scorer_probabilities={}
    )
    uncapped_plan = build_topology_subset_plan(
        graph=graph, sampled_subgraphs=sampled, config=uncapped, scorer_probabilities={}
    )
    size_four = sum(
        len(sg.candidate_negatives) for sg in capped_plan.subgraphs if sg.node_size == 4
    )
    assert size_four <= 3
    assert capped_plan.total_candidate_negatives < uncapped_plan.total_candidate_negatives
    # pi_cand stays a valid inclusion probability under the cap.
    for subgraph in capped_plan.subgraphs:
        for sample in subgraph.candidate_negatives:
            assert 0.0 < sample.pi_cand <= 1.0
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_subset.py::test_build_subset_plan_scores_only_candidate_frame tests/unit/test_tccig_topology_subset.py::test_epoch_subset_keeps_all_positives_and_samples_negatives_with_pi -v
```

Expected: FAIL with import errors for `build_topology_subset_plan`.

- [ ] **Step 3: Implement subset plan classes and sampling functions**

Add to `tccig/topology_subset.py`:

```python
from collections import defaultdict
import random
from collections.abc import Mapping, Sequence
from typing import TypeVar

import networkx as nx

T = TypeVar("T")


@dataclass(frozen=True)
class TopologySubgraphPlan:
    """Bounded candidate and pool plan for one sampled subgraph."""

    subgraph_id: str
    node_size: int
    nodes: tuple[str, ...]
    positives: tuple[TopologyPairSample, ...]
    candidate_negatives: tuple[TopologyPairSample, ...]
    hard_pool: tuple[TopologyPairSample, ...]
    uniform_pool: tuple[TopologyPairSample, ...]


@dataclass(frozen=True)
class TopologySubsetPlan:
    """All subgraph subset-sampling metadata for train topology."""

    subgraphs: tuple[TopologySubgraphPlan, ...]
    active_sizes: tuple[int, ...]
    skipped_sizes: Mapping[int, str]
    total_positive_pairs: int
    total_candidate_negatives: int
    total_pool_negatives: int


def _all_local_pairs(nodes: tuple[str, ...]) -> list[tuple[int, int, str, str, str]]:
    pairs: list[tuple[int, int, str, str, str]] = []
    for index_a, protein_a in enumerate(nodes):
        for index_b in range(index_a + 1, len(nodes)):
            protein_b = nodes[index_b]
            pairs.append((index_a, index_b, protein_a, protein_b, canonical_pair_id(protein_a, protein_b)))
    return pairs


def _draw_without_replacement(items: Sequence[T], *, count: int, rng: random.Random) -> list[T]:
    if count >= len(items):
        return list(items)
    return rng.sample(list(items), count)


def build_topology_subset_plan(
    *,
    graph: nx.Graph,
    sampled_subgraphs: Mapping[int, Sequence[tuple[str, ...]]],
    config: TopologySubsetSamplerConfig,
    scorer_probabilities: Mapping[str, float],
) -> TopologySubsetPlan:
    """Build a bounded candidate/pool topology subset plan."""
    config.validate()
    rng = random.Random(config.seed)
    # Spec §4 per-size subgraph budget is applied UPSTREAM, before this builder, by
    # `apply_per_size_subgraph_budget` (Task 6) — NOT by a tail `rows[:N]` truncation
    # here. Truncating the tail would silently drop the coverage-augmentation subgraphs
    # (which are appended to the tail of `max(node_sizes)`), so coverage_stats logged by
    # the caller would no longer match the built plan (review: coverage-redistribution
    # finding). This builder therefore trusts `sampled_subgraphs` to already respect the
    # subgraph budget; it does not re-cap subgraph counts.
    sampled_subgraphs = {int(size): [tuple(nodes) for nodes in subsets]
                         for size, subsets in sampled_subgraphs.items()}
    # Spec §4 per-size labeled-pair budget: cap the cumulative candidate-negative frame
    # (the scoring-cost driver) per size. 0 == unbounded. Capping candidate_count keeps
    # pi_cand = candidate_count / len(negatives) a valid inclusion ratio, so IPW stays
    # correct; it just scores fewer negatives for over-budget sizes.
    remaining_labeled_budget: dict[int, int] = {
        int(size): config.max_labeled_pairs_per_size for size in sampled_subgraphs
    }
    subgraphs: list[TopologySubgraphPlan] = []
    subgraph_budget_by_size = {int(size): len(subsets) for size, subsets in sampled_subgraphs.items()}
    labeled_budget_by_size: dict[int, int] = defaultdict(int)
    for size, subsets in sampled_subgraphs.items():
        for nodes in subsets:
            labeled_budget_by_size[int(size)] += len(_all_local_pairs(tuple(nodes)))
    active_sizes, skipped = active_node_sizes(
        node_sizes=tuple(sorted(int(size) for size in sampled_subgraphs)),
        graph_node_count=graph.number_of_nodes(),
        subgraphs_per_size=subgraph_budget_by_size,
        labeled_pairs_per_size=dict(labeled_budget_by_size),
    )
    for node_size in active_sizes:
        for subgraph_index, raw_nodes in enumerate(sampled_subgraphs[node_size]):
            nodes = tuple(sorted(raw_nodes))
            subgraph_id = f"size={node_size}:index={subgraph_index}"
            positives: list[TopologyPairSample] = []
            negatives: list[tuple[int, int, str, str, str]] = []
            for index_a, index_b, protein_a, protein_b, pair_id in _all_local_pairs(nodes):
                if graph.has_edge(protein_a, protein_b):
                    positives.append(
                        TopologyPairSample(
                            pair_id=pair_id,
                            subgraph_id=subgraph_id,
                            node_size=node_size,
                            protein_a=protein_a,
                            protein_b=protein_b,
                            local_index_a=index_a,
                            local_index_b=index_b,
                            stratum=SamplingStratum.POSITIVE,
                            pi_cand=1.0,
                            pi_pool_given_cand=1.0,
                            pi_epoch_given_pool=1.0,
                            pi_total=1.0,
                            target=1.0,
                            scorer_probability=float(scorer_probabilities.get(pair_id, 1.0)),
                        )
                    )
                else:
                    negatives.append((index_a, index_b, protein_a, protein_b, pair_id))
            candidate_count = min(len(negatives), max(1, config.candidate_ratio * max(1, len(positives))))
            if config.max_labeled_pairs_per_size > 0:
                # Reserve budget for this subgraph's positives (always scored), then cap
                # candidate negatives by what remains for this size.
                remaining_labeled_budget[node_size] -= len(positives)
                size_cap = max(0, remaining_labeled_budget[node_size])
                candidate_count = min(candidate_count, size_cap)
                remaining_labeled_budget[node_size] = size_cap - candidate_count
            candidate_rows = _draw_without_replacement(negatives, count=candidate_count, rng=rng)
            pi_cand = 1.0 if (not negatives or candidate_count == 0) else candidate_count / float(len(negatives))
            candidate_samples = [
                TopologyPairSample(
                    pair_id=pair_id,
                    subgraph_id=subgraph_id,
                    node_size=node_size,
                    protein_a=protein_a,
                    protein_b=protein_b,
                    local_index_a=index_a,
                    local_index_b=index_b,
                    stratum=SamplingStratum.UNIFORM_NEGATIVE,
                    pi_cand=pi_cand,
                    pi_pool_given_cand=1.0,
                    pi_epoch_given_pool=1.0,
                    pi_total=pi_cand,
                    target=0.0,
                    scorer_probability=float(scorer_probabilities.get(pair_id, 0.0)),
                )
                for index_a, index_b, protein_a, protein_b, pair_id in candidate_rows
            ]
            sorted_candidates = tuple(
                sorted(candidate_samples, key=lambda sample: sample.scorer_probability, reverse=True)
            )
            hard_count = min(
                len(sorted_candidates),
                max(1, int(round(len(sorted_candidates) * config.hard_stratum_fraction))),
            )
            hard_frame = sorted_candidates[:hard_count]
            uniform_frame = sorted_candidates[hard_count:]
            pool_per_pos = config.pool_ratio * max(1, len(positives))
            hard_pool_count = min(len(hard_frame), max(0, int(round(pool_per_pos * config.hard_fraction))))
            uniform_pool_count = min(
                len(uniform_frame), max(0, pool_per_pos - hard_pool_count)
            )
            hard_pool_base = _draw_without_replacement(hard_frame, count=hard_pool_count, rng=rng)
            uniform_pool_base = _draw_without_replacement(
                uniform_frame, count=uniform_pool_count, rng=rng
            )
            hard_pi_pool = 1.0 if not hard_frame else hard_pool_count / float(len(hard_frame))
            uniform_pi_pool = 1.0 if not uniform_frame else uniform_pool_count / float(len(uniform_frame))
            hard_pool = tuple(
                _replace_sample_probability(
                    sample,
                    stratum=SamplingStratum.HARD_NEGATIVE,
                    pi_pool_given_cand=hard_pi_pool,
                    pi_epoch_given_pool=1.0,
                )
                for sample in hard_pool_base
            )
            uniform_pool = tuple(
                _replace_sample_probability(
                    sample,
                    stratum=SamplingStratum.UNIFORM_NEGATIVE,
                    pi_pool_given_cand=uniform_pi_pool,
                    pi_epoch_given_pool=1.0,
                )
                for sample in uniform_pool_base
            )
            subgraphs.append(
                TopologySubgraphPlan(
                    subgraph_id=subgraph_id,
                    node_size=node_size,
                    nodes=nodes,
                    positives=tuple(positives),
                    candidate_negatives=tuple(candidate_samples),
                    hard_pool=hard_pool,
                    uniform_pool=uniform_pool,
                )
            )
    return TopologySubsetPlan(
        subgraphs=tuple(subgraphs),
        active_sizes=active_sizes,
        skipped_sizes=skipped,
        total_positive_pairs=sum(len(subgraph.positives) for subgraph in subgraphs),
        total_candidate_negatives=sum(len(subgraph.candidate_negatives) for subgraph in subgraphs),
        total_pool_negatives=sum(
            len(subgraph.hard_pool) + len(subgraph.uniform_pool) for subgraph in subgraphs
        ),
    )


def _replace_sample_probability(
    sample: TopologyPairSample,
    *,
    stratum: SamplingStratum,
    pi_pool_given_cand: float,
    pi_epoch_given_pool: float,
) -> TopologyPairSample:
    pi_total = sample.pi_cand * pi_pool_given_cand * pi_epoch_given_pool
    return TopologyPairSample(
        pair_id=sample.pair_id,
        subgraph_id=sample.subgraph_id,
        node_size=sample.node_size,
        protein_a=sample.protein_a,
        protein_b=sample.protein_b,
        local_index_a=sample.local_index_a,
        local_index_b=sample.local_index_b,
        stratum=stratum,
        pi_cand=sample.pi_cand,
        pi_pool_given_cand=pi_pool_given_cand,
        pi_epoch_given_pool=pi_epoch_given_pool,
        pi_total=pi_total,
        target=sample.target,
        scorer_probability=sample.scorer_probability,
    )
```

Add epoch sampling:

```python
def sample_epoch_topology_subset(
    *,
    plan: TopologySubsetPlan,
    epoch: int,
    config: TopologySubsetSamplerConfig | None = None,
) -> tuple[TopologyPairSample, ...]:
    """Draw one epoch's topology pair subset from cached pools."""
    cfg = config or TopologySubsetSamplerConfig()
    cfg.validate()
    rng = random.Random(cfg.seed + epoch)
    selected: list[TopologyPairSample] = []
    for subgraph in plan.subgraphs:
        selected.extend(subgraph.positives)
        positives = max(1, len(subgraph.positives))
        epoch_negative_count = cfg.epoch_ratio * positives
        hard_count = min(len(subgraph.hard_pool), int(round(epoch_negative_count * cfg.hard_fraction)))
        uniform_count = min(len(subgraph.uniform_pool), max(0, epoch_negative_count - hard_count))
        hard_draws = _draw_without_replacement(subgraph.hard_pool, count=hard_count, rng=rng)
        uniform_draws = _draw_without_replacement(
            subgraph.uniform_pool, count=uniform_count, rng=rng
        )
        hard_pi_epoch = 1.0 if not subgraph.hard_pool else hard_count / float(len(subgraph.hard_pool))
        uniform_pi_epoch = (
            1.0 if not subgraph.uniform_pool else uniform_count / float(len(subgraph.uniform_pool))
        )
        selected.extend(
            _replace_sample_probability(
                sample,
                stratum=SamplingStratum.HARD_NEGATIVE,
                pi_pool_given_cand=sample.pi_pool_given_cand,
                pi_epoch_given_pool=hard_pi_epoch,
            )
            for sample in hard_draws
        )
        selected.extend(
            _replace_sample_probability(
                sample,
                stratum=SamplingStratum.UNIFORM_NEGATIVE,
                pi_pool_given_cand=sample.pi_pool_given_cand,
                pi_epoch_given_pool=uniform_pi_epoch,
            )
            for sample in uniform_draws
        )
    for sample in selected:
        sample.validate()
    return tuple(selected)
```

- [ ] **Step 4: Run sampler tests**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_subset.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add tccig/topology_subset.py tests/unit/test_tccig_topology_subset.py
rtk git commit -m "feat: sample bounded topology negatives"
```

---

### Task 4: Add Subset Plan Cache Metadata and Serialization

**Files:**
- Modify: `src/topology/plan_cache.py`
- Modify: `tccig/topology_subset.py`
- Test: `tests/unit/test_topology_plan_cache.py`
- Test: `tests/unit/test_tccig_topology_subset.py`

**Correctness note (review Finding 9 — dead helper):** A metadata key with no
payload to key is dead code. The expensive scoring is already cached deterministically
by `_score_split` (keyed on `pair_hash`), but the codebase treats plan persistence as
first-class (`_load_or_build_topology_plan`, commit `44a6baf`). To give
`subset_plan_payload_metadata` a real consumer, this task **also** adds round-trip
serialization for `TopologySubsetPlan`, and Task 6 keys the persisted plan with this
metadata using a **load-before-score** structure (not a build-gating loader). Without
that consumer, do not add the helper.

**Distributed-safety constraint for Task 6's consumer.** The cached subset plan must
be loaded with a *pure read attempted by every rank before any scoring*. On a cache
hit, no rank scores. On a miss, *every* rank runs `_score_split` (which owns its own
main-rank-only write plus `_runtime_barrier`) and then every rank builds the identical
plan deterministically — seeded `sampled` + deterministic cached scores ⇒ identical
`build_topology_subset_plan` on all ranks — so the main rank may write the payload with
no reload barrier. Do **not** nest `_score_split` inside an `if runtime.is_main_process`
build block: non-main ranks would skip the barrier inside `_score_split` and the run
would deadlock. Load-before-score sidesteps this entirely.

- [ ] **Step 1: Write failing cache-key and round-trip tests**

Append to `tests/unit/test_topology_plan_cache.py`:

```python
def test_subset_plan_metadata_changes_with_sampler_parameters() -> None:
    import networkx as nx

    from src.topology.plan_cache import subset_plan_payload_metadata

    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("b", "c")])
    first = subset_plan_payload_metadata(
        split="train_topology",
        graph=graph,
        node_sizes=[20, 40],
        samples_per_size=2,
        seed=0,
        strategy="mixed",
        coverage_augmentation=True,
        candidate_ratio=20,
        pool_ratio=10,
        epoch_ratio=5,
        hard_fraction=0.5,
        uniform_fraction=0.5,
        hard_stratum_fraction=0.2,
        max_subgraphs_per_size=0,
        max_labeled_pairs_per_size=0,
        pair_scope="subset",
        scorer_config={},
    )
    second = subset_plan_payload_metadata(
        split="train_topology",
        graph=graph,
        node_sizes=[20, 40],
        samples_per_size=2,
        seed=0,
        strategy="mixed",
        coverage_augmentation=True,
        candidate_ratio=10,
        pool_ratio=10,
        epoch_ratio=5,
        hard_fraction=0.5,
        uniform_fraction=0.5,
        hard_stratum_fraction=0.2,
        max_subgraphs_per_size=0,
        max_labeled_pairs_per_size=0,
        pair_scope="subset",
        scorer_config={},
    )
    assert first["candidate_ratio"] == 20
    assert first != second


def test_subset_plan_metadata_embeds_scorer_identity() -> None:
    # Review finding: without scorer/checkpoint hashes the cached scored pairs can be
    # silently reused after the frozen scorer changes. The cache key MUST carry the
    # same scorer-identity block that score_cache_metadata uses.
    import networkx as nx

    from src.topology.plan_cache import subset_plan_payload_metadata

    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("b", "c")])
    kwargs = dict(
        split="train_topology",
        graph=graph,
        node_sizes=[20],
        samples_per_size=1,
        seed=0,
        strategy="mixed",
        coverage_augmentation=False,
        candidate_ratio=20,
        pool_ratio=10,
        epoch_ratio=5,
        hard_fraction=0.5,
        uniform_fraction=0.5,
        hard_stratum_fraction=0.2,
        max_subgraphs_per_size=0,
        max_labeled_pairs_per_size=0,
        pair_scope="subset",
    )
    meta = subset_plan_payload_metadata(scorer_config={"max_sequence_length": 1000}, **kwargs)
    assert "scorer" in meta
    assert meta["scorer"]["max_sequence_length"] == 1000
    other = subset_plan_payload_metadata(scorer_config={"max_sequence_length": 2000}, **kwargs)
    assert meta != other
```

Append to `tests/unit/test_tccig_topology_subset.py`:

```python
from tccig.topology_subset import subset_plan_to_payload, payload_to_subset_plan


def test_subset_plan_payload_round_trips() -> None:
    graph = _toy_graph()
    sampled = {4: [("a", "b", "c", "d")]}
    cfg = TopologySubsetSamplerConfig(candidate_ratio=3, pool_ratio=2, epoch_ratio=1, seed=4)
    plan = build_topology_subset_plan(
        graph=graph,
        sampled_subgraphs=sampled,
        config=cfg,
        scorer_probabilities={"a||c": 0.9, "a||d": 0.2, "b||d": 0.7, "c||d": 0.1},
    )
    restored = payload_to_subset_plan(subset_plan_to_payload(plan))
    assert restored == plan
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_topology_plan_cache.py::test_subset_plan_metadata_changes_with_sampler_parameters tests/unit/test_tccig_topology_subset.py::test_subset_plan_payload_round_trips -v
```

Expected: FAIL with `ImportError` for `subset_plan_payload_metadata` /
`subset_plan_to_payload` / `payload_to_subset_plan`.

- [ ] **Step 3: Implement subset metadata helper**

Add to `src/topology/plan_cache.py`. Import the scorer-identity helpers at the top of
the module (they already exist in `tccig.prepare`):

```python
from tccig.prepare import (
    embedding_index_sha256,
    optional_file_sha256,
    write_json,  # already imported; keep the existing import, just add the two hashers
)
```

```python
SUBSET_METADATA_VERSION = 1


def _scorer_identity(scorer_config: Mapping[str, object]) -> dict[str, object]:
    """Return the same scorer-identity block score_cache_metadata uses.

    Embedding this in the subset-plan cache key means a changed checkpoint, model
    config, embedding index, or max_sequence_length invalidates the persisted plan —
    so stale scored pairs are never silently reused.
    """
    return {
        "model_config_sha256": optional_file_sha256(scorer_config.get("model_config_path")),
        "checkpoint_sha256": optional_file_sha256(scorer_config.get("checkpoint_path")),
        "embedding_index_sha256": embedding_index_sha256(scorer_config),
        "max_sequence_length": scorer_config.get("max_sequence_length"),
    }


def subset_plan_payload_metadata(
    *,
    split: str,
    graph: nx.Graph,
    node_sizes: Sequence[int],
    samples_per_size: int,
    seed: int,
    strategy: str,
    coverage_augmentation: bool,
    candidate_ratio: int,
    pool_ratio: int,
    epoch_ratio: int,
    hard_fraction: float,
    uniform_fraction: float,
    hard_stratum_fraction: float,
    max_subgraphs_per_size: int,
    max_labeled_pairs_per_size: int,
    pair_scope: str,
    scorer_config: Mapping[str, object],
) -> dict[str, object]:
    """Build the strict cache key for a subset topology plan payload.

    The key covers the sampling parameters (so a sampler change invalidates the plan)
    AND the frozen scorer identity (so a checkpoint/config/embedding change invalidates
    the *scored* pairs baked into the plan). `pair_scope` distinguishes this from the
    full-plan key. `kind="subset"` lets the loader route to the subset validator.
    """
    metadata = plan_payload_metadata(
        split=split,
        graph=graph,
        node_sizes=node_sizes,
        samples_per_size=samples_per_size,
        seed=seed,
        strategy=strategy,
        coverage_augmentation=coverage_augmentation,
    )
    metadata.update(
        {
            "kind": "subset",
            "subset_version": SUBSET_METADATA_VERSION,
            "pair_scope": pair_scope,
            "candidate_ratio": int(candidate_ratio),
            "pool_ratio": int(pool_ratio),
            "epoch_ratio": int(epoch_ratio),
            "hard_fraction": float(hard_fraction),
            "uniform_fraction": float(uniform_fraction),
            "hard_stratum_fraction": float(hard_stratum_fraction),
            "max_subgraphs_per_size": int(max_subgraphs_per_size),
            "max_labeled_pairs_per_size": int(max_labeled_pairs_per_size),
            "scorer": _scorer_identity(scorer_config),
        }
    )
    return metadata
```

- [ ] **Step 4: Implement subset-plan serialization**

Add to `tccig/topology_subset.py`:

```python
def _sample_to_dict(sample: TopologyPairSample) -> dict[str, object]:
    return {
        "pair_id": sample.pair_id,
        "subgraph_id": sample.subgraph_id,
        "node_size": sample.node_size,
        "protein_a": sample.protein_a,
        "protein_b": sample.protein_b,
        "local_index_a": sample.local_index_a,
        "local_index_b": sample.local_index_b,
        "stratum": sample.stratum.value,
        "pi_cand": sample.pi_cand,
        "pi_pool_given_cand": sample.pi_pool_given_cand,
        "pi_epoch_given_pool": sample.pi_epoch_given_pool,
        "pi_total": sample.pi_total,
        "target": sample.target,
        "scorer_probability": sample.scorer_probability,
    }


def _sample_from_dict(raw: Mapping[str, object]) -> TopologyPairSample:
    return TopologyPairSample(
        pair_id=str(raw["pair_id"]),
        subgraph_id=str(raw["subgraph_id"]),
        node_size=int(raw["node_size"]),  # type: ignore[arg-type]
        protein_a=str(raw["protein_a"]),
        protein_b=str(raw["protein_b"]),
        local_index_a=int(raw["local_index_a"]),  # type: ignore[arg-type]
        local_index_b=int(raw["local_index_b"]),  # type: ignore[arg-type]
        stratum=SamplingStratum(str(raw["stratum"])),
        pi_cand=float(raw["pi_cand"]),  # type: ignore[arg-type]
        pi_pool_given_cand=float(raw["pi_pool_given_cand"]),  # type: ignore[arg-type]
        pi_epoch_given_pool=float(raw["pi_epoch_given_pool"]),  # type: ignore[arg-type]
        pi_total=float(raw["pi_total"]),  # type: ignore[arg-type]
        target=float(raw["target"]),  # type: ignore[arg-type]
        scorer_probability=float(raw["scorer_probability"]),  # type: ignore[arg-type]
    )


SUBSET_PAYLOAD_VERSION = 1


def subset_plan_to_payload(plan: TopologySubsetPlan) -> dict[str, object]:
    """Serialize a TopologySubsetPlan to a JSON-friendly payload.

    The `payload_kind`/`subset_payload_version` stamp lets the subset cache loader
    reject a full-plan payload (different shape) or a stale subset schema instead of
    crashing in `payload_to_subset_plan` on a missing key.
    """
    return {
        "payload_kind": "topology_subset",
        "subset_payload_version": SUBSET_PAYLOAD_VERSION,
        "active_sizes": list(plan.active_sizes),
        "skipped_sizes": {str(size): reason for size, reason in plan.skipped_sizes.items()},
        "total_positive_pairs": plan.total_positive_pairs,
        "total_candidate_negatives": plan.total_candidate_negatives,
        "total_pool_negatives": plan.total_pool_negatives,
        "subgraphs": [
            {
                "subgraph_id": subgraph.subgraph_id,
                "node_size": subgraph.node_size,
                "nodes": list(subgraph.nodes),
                "positives": [_sample_to_dict(sample) for sample in subgraph.positives],
                "candidate_negatives": [
                    _sample_to_dict(sample) for sample in subgraph.candidate_negatives
                ],
                "hard_pool": [_sample_to_dict(sample) for sample in subgraph.hard_pool],
                "uniform_pool": [_sample_to_dict(sample) for sample in subgraph.uniform_pool],
            }
            for subgraph in plan.subgraphs
        ],
    }


def payload_to_subset_plan(payload: Mapping[str, object]) -> TopologySubsetPlan:
    """Rebuild a TopologySubsetPlan from its serialized payload."""
    raw_subgraphs = cast("Sequence[Mapping[str, object]]", payload["subgraphs"])
    subgraphs = tuple(
        TopologySubgraphPlan(
            subgraph_id=str(raw["subgraph_id"]),
            node_size=int(raw["node_size"]),  # type: ignore[arg-type]
            nodes=tuple(str(node) for node in cast("Sequence[object]", raw["nodes"])),
            positives=tuple(
                _sample_from_dict(item)
                for item in cast("Sequence[Mapping[str, object]]", raw["positives"])
            ),
            candidate_negatives=tuple(
                _sample_from_dict(item)
                for item in cast("Sequence[Mapping[str, object]]", raw["candidate_negatives"])
            ),
            hard_pool=tuple(
                _sample_from_dict(item)
                for item in cast("Sequence[Mapping[str, object]]", raw["hard_pool"])
            ),
            uniform_pool=tuple(
                _sample_from_dict(item)
                for item in cast("Sequence[Mapping[str, object]]", raw["uniform_pool"])
            ),
        )
        for raw in raw_subgraphs
    )
    skipped_raw = cast("Mapping[str, object]", payload["skipped_sizes"])
    return TopologySubsetPlan(
        subgraphs=subgraphs,
        active_sizes=tuple(int(size) for size in cast("Sequence[object]", payload["active_sizes"])),
        skipped_sizes={int(size): str(reason) for size, reason in skipped_raw.items()},
        total_positive_pairs=int(payload["total_positive_pairs"]),  # type: ignore[arg-type]
        total_candidate_negatives=int(payload["total_candidate_negatives"]),  # type: ignore[arg-type]
        total_pool_negatives=int(payload["total_pool_negatives"]),  # type: ignore[arg-type]
    )
```

Add `from typing import TypeVar, cast` (extend the existing `typing` import) at the
top of `tccig/topology_subset.py` if `cast` is not already imported.

- [ ] **Step 5: Add a subset-specific cache loader/writer (do NOT reuse `load_plan_cache`)**

**Why a new loader is required (review: subset payload rejected).** The existing
`load_plan_cache` (`src/topology/plan_cache.py:248`) calls `_payload_is_rehydratable`,
which hard-validates the *full-plan* shape (`version == PAYLOAD_VERSION`, a `buckets`
list, and recomputed `total_subgraphs`/`total_pairs`). A subset payload has none of
those keys, so `load_plan_cache` would log "structurally invalid" and return `None` on
*every* call — the subset cache could never hit. Task 6 must therefore use a separate
loader that validates the subset shape. The two share `_plan_path`/`_manifest_path`,
`write_json`, and the metadata-equality check, but not the payload validator.

Add a subset-shape validator and load/write pair to `src/topology/plan_cache.py`:

```python
def _subset_payload_is_rehydratable(payload: Mapping[str, object]) -> bool:
    """Cheap schema check for a serialized TopologySubsetPlan payload.

    Distinct from `_payload_is_rehydratable` (full-plan only): validates the subset
    payload kind/version and that every subgraph carries the four sample lists, so a
    full-plan payload or a stale subset schema is rejected instead of KeyError-ing in
    `payload_to_subset_plan`. The accepted version is owned by the module that *writes*
    the payload (`tccig.topology_subset`), imported here so there is a single source of
    truth — not a second constant that can drift out of sync.
    """
    from tccig.topology_subset import SUBSET_PAYLOAD_VERSION as _WRITER_VERSION

    if payload.get("payload_kind") != "topology_subset":
        return False
    if payload.get("subset_payload_version") != _WRITER_VERSION:
        return False
    raw_subgraphs = payload.get("subgraphs")
    if not isinstance(raw_subgraphs, list):
        return False
    required_lists = ("positives", "candidate_negatives", "hard_pool", "uniform_pool")
    for subgraph in raw_subgraphs:
        if not isinstance(subgraph, Mapping):
            return False
        if not isinstance(subgraph.get("node_size"), int):
            return False
        if not isinstance(subgraph.get("nodes"), list):
            return False
        for key in required_lists:
            if not isinstance(subgraph.get(key), list):
                return False
    for key in ("active_sizes", "skipped_sizes", "total_positive_pairs",
                "total_candidate_negatives", "total_pool_negatives"):
        if key not in payload:
            return False
    return True


def load_subset_plan_cache(
    *,
    cache_dir: Path,
    split: str,
    metadata: Mapping[str, object],
) -> dict[str, object] | None:
    """Load a cached subset-plan payload, or ``None`` on miss/mismatch/corruption.

    Mirrors `load_plan_cache` but validates the subset payload shape via
    `_subset_payload_is_rehydratable` instead of the full-plan `_payload_is_rehydratable`.
    """
    path = _plan_path(cache_dir, split)
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("ignoring corrupt topology subset plan cache at %s", path)
        return None
    if not isinstance(document, Mapping):
        LOGGER.warning("ignoring malformed topology subset plan cache at %s", path)
        return None
    if document.get("metadata") != dict(metadata):
        return None
    payload = document.get("payload")
    if not isinstance(payload, Mapping):
        return None
    if not _subset_payload_is_rehydratable(payload):
        LOGGER.warning("ignoring structurally invalid topology subset plan cache at %s", path)
        return None
    return dict(payload)


def write_subset_plan_cache(
    *,
    cache_dir: Path,
    split: str,
    metadata: Mapping[str, object],
    payload: Mapping[str, object],
) -> None:
    """Persist a subset-plan payload plus its cache-key metadata and a manifest.

    Identical on-disk layout to `write_plan_cache`; kept separate so the subset path
    never silently writes under a full-plan validator's assumptions.
    """
    write_json(
        _plan_path(cache_dir, split),
        {"metadata": dict(metadata), "payload": dict(payload)},
    )
    write_json(_manifest_path(cache_dir, split), dict(metadata))
```

Add a failing test to `tests/unit/test_topology_plan_cache.py` first:

```python
def test_subset_cache_round_trips_and_rejects_full_plan_payload(tmp_path) -> None:
    import networkx as nx

    from src.topology.plan_cache import (
        load_plan_cache,
        load_subset_plan_cache,
        write_subset_plan_cache,
    )
    from tccig.topology_subset import (
        TopologySubsetSamplerConfig,
        build_topology_subset_plan,
        subset_plan_to_payload,
    )

    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "c", "d"])
    graph.add_edges_from([("a", "b"), ("b", "c")])
    cfg = TopologySubsetSamplerConfig(candidate_ratio=2, pool_ratio=1, epoch_ratio=1, seed=1)
    plan = build_topology_subset_plan(
        graph=graph, sampled_subgraphs={4: [("a", "b", "c", "d")]}, config=cfg,
        scorer_probabilities={},
    )
    metadata = {"pair_scope": "subset", "candidate_ratio": 2}
    write_subset_plan_cache(
        cache_dir=tmp_path, split="train_topology_subset", metadata=metadata,
        payload=subset_plan_to_payload(plan),
    )
    # Subset loader hits; the full-plan loader rejects the subset payload shape.
    assert load_subset_plan_cache(
        cache_dir=tmp_path, split="train_topology_subset", metadata=metadata
    ) is not None
    assert load_plan_cache(
        cache_dir=tmp_path, split="train_topology_subset", metadata=metadata
    ) is None
```

- [ ] **Step 6: Run cache and round-trip tests**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_topology_plan_cache.py tests/unit/test_tccig_topology_subset.py::test_subset_plan_payload_round_trips -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add src/topology/plan_cache.py tccig/topology_subset.py tests/unit/test_topology_plan_cache.py tests/unit/test_tccig_topology_subset.py
rtk git commit -m "feat: key and serialize topology subset plan caches"
```

---

### Task 5: Parse Subset Config in S2GAE

**Files:**
- Modify: `tccig/s2gae.py`
- Test: `tests/unit/test_tccig_topology_training.py`

- [ ] **Step 1: Add failing config parse test**

Append to `tests/unit/test_tccig_topology_training.py`:

```python
def test_parse_config_reads_topology_subset_sampler() -> None:
    from tccig.s2gae import _parse_config

    config = _base_refiner_config()
    config["topology_training"]["subset"] = {
        "enabled": True,
        "candidate_ratio": 20,
        "pool_ratio": 10,
        "epoch_ratio": 5,
        "hard_fraction": 0.5,
        "uniform_fraction": 0.5,
        "hard_stratum_fraction": 0.2,
        "seed": 11,
    }
    cfg = _parse_config(config)
    assert cfg.topology_training.subset.enabled is True
    assert cfg.topology_training.subset.candidate_ratio == 20
    assert cfg.topology_training.subset.seed == 11
    # Review Finding 11: the subset training path forces clustering OFF in its own
    # chunk loss (Task 8), so the shared `compute_clustering_mmd` knob is left at its
    # production default (True) and continues to drive the unchanged validation metrics.
    assert cfg.topology_validation.compute_clustering_mmd is True
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_training.py::test_parse_config_reads_topology_subset_sampler -v
```

Expected: FAIL with `AttributeError: 'S2GAETopologyTrainingConfig' object has no attribute 'subset'`.

- [ ] **Step 3: Extend topology training config**

Modify `tccig/s2gae.py` imports:

```python
from tccig.topology_subset import TopologySubsetSamplerConfig
```

Add field to `S2GAETopologyTrainingConfig`:

```python
    subset: TopologySubsetSamplerConfig
```

Add parser helper near `_parse_topology_training_config`:

```python
def _parse_topology_subset_config(raw: object) -> TopologySubsetSamplerConfig:
    """Parse refiner.topology_training.subset."""
    if raw is None:
        return TopologySubsetSamplerConfig(enabled=False)
    if not isinstance(raw, Mapping):
        raise ValueError("refiner.topology_training.subset must be a mapping")
    cfg = TopologySubsetSamplerConfig(
        enabled=_bool(raw.get("enabled", True), "refiner.topology_training.subset.enabled"),
        candidate_ratio=_positive_int(
            raw.get("candidate_ratio", 20), "refiner.topology_training.subset.candidate_ratio"
        ),
        pool_ratio=_positive_int(
            raw.get("pool_ratio", 10), "refiner.topology_training.subset.pool_ratio"
        ),
        epoch_ratio=_positive_int(
            raw.get("epoch_ratio", 5), "refiner.topology_training.subset.epoch_ratio"
        ),
        hard_fraction=_non_negative_float(
            raw.get("hard_fraction", 0.5), "refiner.topology_training.subset.hard_fraction"
        ),
        uniform_fraction=_non_negative_float(
            raw.get("uniform_fraction", 0.5),
            "refiner.topology_training.subset.uniform_fraction",
        ),
        hard_stratum_fraction=_non_negative_float(
            raw.get("hard_stratum_fraction", 0.2),
            "refiner.topology_training.subset.hard_stratum_fraction",
        ),
        seed=_non_negative_int(raw.get("seed", 0), "refiner.topology_training.subset.seed"),
        max_subgraphs_per_size=_non_negative_int(
            raw.get("max_subgraphs_per_size", 0),
            "refiner.topology_training.subset.max_subgraphs_per_size",
        ),
        max_labeled_pairs_per_size=_non_negative_int(
            raw.get("max_labeled_pairs_per_size", 0),
            "refiner.topology_training.subset.max_labeled_pairs_per_size",
        ),
        bias_diagnostic_every_n_epochs=_non_negative_int(
            raw.get("bias_diagnostic_every_n_epochs", 0),
            "refiner.topology_training.subset.bias_diagnostic_every_n_epochs",
        ),
        bias_diagnostic_max_node_size=_non_negative_int(
            raw.get("bias_diagnostic_max_node_size", 40),
            "refiner.topology_training.subset.bias_diagnostic_max_node_size",
        ),
        bias_diagnostic_max_subgraphs=_non_negative_int(
            raw.get("bias_diagnostic_max_subgraphs", 4),
            "refiner.topology_training.subset.bias_diagnostic_max_subgraphs",
        ),
    )
    cfg.validate()
    return cfg
```

Thread it into `_parse_topology_training_config`:

```python
        subset=_parse_topology_subset_config(raw.get("subset")),
```

For the disabled default branch, set:

```python
            subset=TopologySubsetSamplerConfig(enabled=False),
```

- [ ] **Step 4: Run config tests**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_training.py::test_parse_config_reads_residual_anchor_and_topology_training tests/unit/test_tccig_topology_training.py::test_parse_config_defaults_topology_training_disabled tests/unit/test_tccig_topology_training.py::test_parse_config_reads_topology_subset_sampler -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add tccig/s2gae.py tests/unit/test_tccig_topology_training.py
rtk git commit -m "feat: parse topology subset training config"
```

---

### Task 6: Build and Score the Bounded Train-Topology Pool

**Files:**
- Modify: `tccig/train.py`
- Modify: `tccig/topology_subset.py`
- Test: `tests/unit/test_tccig_topology_subset.py`
- Test: `tests/unit/test_tccig_topology_training.py`

- [ ] **Step 1: Add failing test for pool-score pair extraction**

Append to `tests/unit/test_tccig_topology_subset.py`:

```python
from tccig.topology_subset import candidate_pairs_for_scoring


def test_candidate_pairs_for_scoring_are_unique_and_ordered() -> None:
    graph = _toy_graph()
    sampled = {4: [("a", "b", "c", "d"), ("a", "b", "c", "e")]}
    cfg = TopologySubsetSamplerConfig(candidate_ratio=2, pool_ratio=1, epoch_ratio=1, seed=7)
    plan = build_topology_subset_plan(
        graph=graph,
        sampled_subgraphs=sampled,
        config=cfg,
        scorer_probabilities={},
    )
    pairs = candidate_pairs_for_scoring(plan)
    pair_ids = [pair_id for pair_id, _, _ in pairs]
    assert pair_ids == sorted(set(pair_ids))
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_subset.py::test_candidate_pairs_for_scoring_are_unique_and_ordered -v
```

Expected: FAIL with import error for `candidate_pairs_for_scoring`.

- [ ] **Step 3: Implement candidate pair extraction**

Add to `tccig/topology_subset.py`:

```python
def candidate_pairs_for_scoring(plan: TopologySubsetPlan) -> tuple[tuple[str, str, str], ...]:
    """Return unique candidate pair ids and endpoints that need frozen-scorer scores."""
    by_id: dict[str, tuple[str, str, str]] = {}
    for subgraph in plan.subgraphs:
        for sample in (*subgraph.positives, *subgraph.candidate_negatives):
            by_id.setdefault(sample.pair_id, (sample.pair_id, sample.protein_a, sample.protein_b))
    return tuple(by_id[pair_id] for pair_id in sorted(by_id))
```

- [ ] **Step 3b: Implement budget-aware coverage + per-size cap**

**Why this exists (review: coverage-redistribution finding).** The earlier draft ran
`augment_plan_for_positive_edge_coverage` (which appends every uncovered-positive bucket
to `max(node_sizes)`), logged "coverage = 1.0", and *then* let the builder truncate the
per-size list with `rows[:N]` — silently dropping the tail coverage buckets so the
realized plan no longer covered those positives, while the log still claimed it did. The
fix moves both budgeting and coverage into one place that (1) caps the *base* sampled
subgraphs per size, (2) augments for coverage, (3) places each coverage subgraph in the
**smallest eligible size with remaining budget** (distributing across sizes, not dumping
into the largest), and (4) returns the realized post-cap coverage so the log never
over-claims. The builder no longer truncates (Task 3).

First add a failing test. Append to `tests/unit/test_tccig_topology_subset.py`:

```python
from tccig.topology_subset import apply_per_size_subgraph_budget


def test_budget_distributes_coverage_and_reports_realized_coverage() -> None:
    # size-4 base is capped to 1, but a coverage subgraph for the uncovered edge d-e
    # must still be placed (in an eligible size with remaining budget), and the
    # returned stats must reflect the realized (post-cap) coverage, not the pre-cap claim.
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "c", "d", "e", "f"])
    graph.add_edges_from([("a", "b"), ("b", "c"), ("d", "e")])
    base_sampled = {4: [("a", "b", "c", "f"), ("a", "b", "c", "d"), ("a", "c", "d", "f")]}
    budgeted, stats = apply_per_size_subgraph_budget(
        graph=graph,
        base_sampled=base_sampled,
        node_sizes=(4,),
        strategy="BFS",
        seed=0,
        max_subgraphs_per_size=2,
    )
    # No size exceeds its cap.
    assert all(len(rows) <= 2 for rows in budgeted.values())
    # The d-e positive edge is covered by some retained subgraph.
    covered_edges = {
        frozenset((u, v))
        for rows in budgeted.values()
        for nodes in rows
        for u, v in graph.subgraph(set(nodes)).edges()
    }
    assert frozenset(("d", "e")) in covered_edges
    # Realized coverage is reported honestly in [0, 1].
    assert 0.0 <= float(stats["positive_edge_coverage"]) <= 1.0
    assert stats["positive_edge_coverage"] == 1.0


def test_budget_unbounded_is_passthrough_with_coverage() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "c", "d"])
    graph.add_edges_from([("a", "b"), ("c", "d")])
    base_sampled = {4: [("a", "b", "c", "d")]}
    budgeted, stats = apply_per_size_subgraph_budget(
        graph=graph,
        base_sampled=base_sampled,
        node_sizes=(4,),
        strategy="BFS",
        seed=0,
        max_subgraphs_per_size=0,  # unbounded
    )
    assert budgeted[4] == [("a", "b", "c", "d")]
    assert stats["positive_edge_coverage"] == 1.0
```

Run it and confirm `ImportError` for `apply_per_size_subgraph_budget`, then add to
`tccig/topology_subset.py`:

```python
def _induced_covered_edges(
    sampled: Mapping[int, Sequence[tuple[str, ...]]], graph: nx.Graph
) -> set[frozenset[str]]:
    covered: set[frozenset[str]] = set()
    for rows in sampled.values():
        for nodes in rows:
            for node_a, node_b in graph.subgraph(set(nodes)).edges():
                covered.add(frozenset((node_a, node_b)))
    return covered


def apply_per_size_subgraph_budget(
    *,
    graph: nx.Graph,
    base_sampled: Mapping[int, Sequence[tuple[str, ...]]],
    node_sizes: tuple[int, ...],
    strategy: str,
    seed: int,
    max_subgraphs_per_size: int,
    coverage_augmentation: bool = True,
) -> tuple[dict[int, list[tuple[str, ...]]], dict[str, float | int]]:
    """Cap base subgraphs per size, then add budget-aware coverage subgraphs.

    This is the SINGLE entry point for the subset path's budget + coverage logic. Do
    NOT call `augment_plan_for_positive_edge_coverage` separately around it — that helper
    appends coverage buckets to `max(node_sizes)` and asserts full coverage, which both
    bypasses the per-size budget and would double-augment. This helper instead interleaves
    capping and coverage so the cap can never silently delete a coverage subgraph that
    augmentation just added (review: coverage-redistribution finding).

    Order of operations:
      1. Cap each size's base subgraphs to `max_subgraphs_per_size` (0 == unbounded).
      2. If `coverage_augmentation`, walk the still-uncovered positive edges and place a
         coverage subgraph for each into the smallest eligible size that still has
         remaining budget — so the largest bucket is not the sole coverage dump.
      3. Recompute coverage from the realized (post-budget) plan and return it, so the
         logged number is the coverage that the built plan actually has.

    Unlike `augment_plan_for_positive_edge_coverage`, this does NOT raise when coverage
    stays below 1.0 under a tight budget: budgeting is an explicit memory/coverage
    tradeoff and the realized coverage is reported in `stats` for the operator to see.
    """
    from src.topology.finetune_data import _expand_chunk_nodes  # local import: heavy module

    rng = random.Random(seed)
    normalized = strategy.upper()
    if normalized not in {"BFS", "DFS", "RANDOM_WALK"}:
        normalized = "BFS"
    # (1) Cap the base sampled subgraphs per size.
    budgeted: dict[int, list[tuple[str, ...]]] = {}
    for size in sorted(int(s) for s in base_sampled):
        rows = [tuple(sorted(nodes)) for nodes in base_sampled[size]]
        if max_subgraphs_per_size > 0:
            rows = rows[:max_subgraphs_per_size]
        budgeted[size] = rows
    base_bucket_count = sum(len(rows) for rows in budgeted.values())

    def _remaining(size: int) -> int:
        if max_subgraphs_per_size <= 0:
            return 2**31  # effectively unbounded
        return max(0, max_subgraphs_per_size - len(budgeted.get(size, [])))

    eligible_sizes = tuple(
        size for size in sorted(node_sizes) if size <= graph.number_of_nodes()
    )
    all_positive = {
        frozenset((node_a, node_b)) for node_a, node_b in graph.edges()
    }
    covered = _induced_covered_edges(budgeted, graph)
    coverage_bucket_count = 0
    # (2) Distribute coverage subgraphs across eligible sizes under remaining budget.
    if coverage_augmentation:
        for edge in sorted(tuple(sorted(e)) for e in (all_positive - covered)):
            if frozenset(edge) in covered:
                continue  # drained by a previously added coverage bucket
            target_size = next(
                (size for size in eligible_sizes if _remaining(size) > 0), None
            )
            if target_size is None:
                break  # every eligible size is at budget; report honest partial coverage
            nodes = _expand_chunk_nodes(
                graph=graph,
                edge_chunk=[(edge[0], edge[1])],
                target_size=target_size,
                strategy=normalized,
                rng=rng,
            )
            budgeted.setdefault(target_size, []).append(tuple(sorted(nodes)))
            for node_a, node_b in graph.subgraph(set(nodes)).edges():
                covered.add(frozenset((node_a, node_b)))
            coverage_bucket_count += 1

    # (3) Recompute coverage from the realized plan so the logged number is honest.
    matched = len(covered & all_positive)
    coverage = 1.0 if not all_positive else matched / len(all_positive)
    stats: dict[str, float | int] = {
        "base_bucket_count": base_bucket_count,
        "coverage_bucket_count": coverage_bucket_count,
        "positive_edge_coverage": coverage,
    }
    return budgeted, stats
```

If `max_subgraphs_per_size == 0` (unbounded) and `coverage_augmentation` is on, the
realized plan matches the old full-coverage path (every uncovered positive edge gets a
bucket). With a finite budget the helper degrades gracefully to partial coverage rather
than raising.

Run the two new tests and confirm PASS.

- [ ] **Step 4: Refactor `tccig/train.py` plan build to use subset when enabled**

In `tccig/train.py`, import:

```python
from tccig.topology_subset import (
    TopologySubsetPlan,
    apply_per_size_subgraph_budget,
    build_topology_subset_plan,
    candidate_pairs_for_scoring,
)
```

(`sample_topology_evaluation_subgraphs` is already imported at `tccig/train.py:33`, so
the branch reuses it; only the four names above are added.)

**Correctness note (review Finding 2):** In the real `_build_train_topology_bundle`,
`sampled` is a local of the `_build()` closure (it is built inside `_build`, not at
function scope), and `coverage_stats` is only produced by
`_load_or_build_topology_plan(...)`. The subset branch therefore *cannot* reference
either name — it must build its own `sampled` (with optional coverage augmentation)
and produce its own `coverage_stats`. Insert the subset branch immediately after
`coverage_augmentation` is parsed (right before the `def _build()` closure), and
return early so the full-plan path is never reached when subset sampling is on.

Use this implementation shape. The branch opens by parsing the subset config and
building its **own** `sampled` subgraphs and `coverage_stats` (do not reference the
`_build()` closure's locals):

```python
    subset_raw = topo_cfg.get("subset")
    if isinstance(subset_raw, Mapping) and bool(subset_raw.get("enabled", False)):
        subset_cfg = s2gae._parse_topology_subset_config(subset_raw)
        # Build this branch's own base sampled subgraphs (the full-plan `_build()`
        # closure's `sampled` local is NOT in scope here — review Finding 2).
        sampled = sample_topology_evaluation_subgraphs(
            graph=train_graph,
            seed=seed,
            strategy=strategy,
            node_sizes=node_sizes,
            samples_per_size=samples_per_size,
        )
        # Apply the §4 per-size subgraph budget AND coverage augmentation in ONE call.
        # `apply_per_size_subgraph_budget` is the single owner of budget+coverage for the
        # subset path: it caps base subgraphs per size, then (when coverage_augmentation
        # is on) distributes coverage subgraphs into eligible sizes under the remaining
        # budget, and recomputes coverage_stats from the realized plan. Do NOT also call
        # `augment_plan_for_positive_edge_coverage` here — that would append coverage to
        # `max(node_sizes)`, bypass the budget, and double-augment. The build step below
        # also trusts pre-budgeted input (`build_topology_subset_plan` no longer truncates),
        # so this is the only place the budget is enforced (review: coverage-redistribution).
        sampled, coverage_stats = apply_per_size_subgraph_budget(
            graph=train_graph,
            base_sampled={int(k): list(v) for k, v in sampled.items()},
            node_sizes=node_sizes,
            strategy=strategy,
            seed=seed,
            max_subgraphs_per_size=subset_cfg.max_subgraphs_per_size,
            coverage_augmentation=coverage_augmentation,
        )
        if coverage_stats:
            LOGGER.info(
                "tccig train topology coverage (post-budget): base_buckets=%s "
                "coverage_buckets=%s positive_edge_coverage=%.4f",
                coverage_stats.get("base_bucket_count"),
                coverage_stats.get("coverage_bucket_count"),
                float(coverage_stats.get("positive_edge_coverage", 0.0)),
            )
        # ... resolve `subset_plan` via the load-before-score cache wiring below ...
```

Keep the existing `_build()` closure and full-plan `_load_or_build_topology_plan`
path exactly as-is below this branch; it runs only when `subset.enabled` is false.

**Plan-cache wiring (review Finding 9 consumer).** Continue the branch with a
load-before-score structure that resolves `subset_plan` (reused across restarts) and
only scores on a cache miss:

```python
        from src.topology.plan_cache import (
            load_subset_plan_cache,
            subset_plan_payload_metadata,
            write_subset_plan_cache,
        )
        from tccig.topology_subset import payload_to_subset_plan, subset_plan_to_payload

        subset_metadata = subset_plan_payload_metadata(
            split="train_topology",
            graph=train_graph,
            node_sizes=node_sizes,
            samples_per_size=samples_per_size,
            seed=seed,
            strategy=strategy,
            coverage_augmentation=coverage_augmentation,
            candidate_ratio=subset_cfg.candidate_ratio,
            pool_ratio=subset_cfg.pool_ratio,
            epoch_ratio=subset_cfg.epoch_ratio,
            hard_fraction=subset_cfg.hard_fraction,
            uniform_fraction=subset_cfg.uniform_fraction,
            hard_stratum_fraction=subset_cfg.hard_stratum_fraction,
            max_subgraphs_per_size=subset_cfg.max_subgraphs_per_size,
            max_labeled_pairs_per_size=subset_cfg.max_labeled_pairs_per_size,
            scorer_config=scorer_cfg,
            pair_scope="subset",
        )
        # Subset-specific loader: the full-plan load_plan_cache would reject this
        # payload shape via _payload_is_rehydratable (review: cache-load finding).
        cached_payload = load_subset_plan_cache(
            cache_dir=cache_dir, split="train_topology_subset", metadata=subset_metadata
        )
        if cached_payload is not None:
            subset_plan = payload_to_subset_plan(cached_payload)
        else:
            # Cache miss: EVERY rank scores (via _score_split, which barriers itself),
            # then EVERY rank builds the identical plan deterministically.
            empty_plan = build_topology_subset_plan(
                graph=train_graph,
                sampled_subgraphs=sampled,
                config=subset_cfg,
                scorer_probabilities={},
            )
            scoring_rows = candidate_pairs_for_scoring(empty_plan)
            LOGGER.info(
                "tccig train topology subset scoring estimate: unique_pairs=%s "
                "positives=%s candidate_negatives=%s pool_negatives=%s skipped_sizes=%s",
                len(scoring_rows),
                empty_plan.total_positive_pairs,
                empty_plan.total_candidate_negatives,
                empty_plan.total_pool_negatives,
                dict(empty_plan.skipped_sizes),
            )
            candidate_pairs = [
                CandidatePair(protein_a, protein_b) for _, protein_a, protein_b in scoring_rows
            ]
            candidate_scores = _score_split(
                split="train_topology_subset_candidates",
                pairs=candidate_pairs,
                scorer_cfg=scorer_cfg,
                runtime=runtime,
                cache_dir=cache_dir,
            )
            score_by_pair_id = {
                pair_id: float(score)
                for (pair_id, _protein_a, _protein_b), score in zip(
                    scoring_rows, candidate_scores, strict=True
                )
            }
            subset_plan = build_topology_subset_plan(
                graph=train_graph,
                sampled_subgraphs=sampled,
                config=subset_cfg,
                scorer_probabilities=score_by_pair_id,
            )
            if runtime.is_main_process:
                write_subset_plan_cache(
                    cache_dir=cache_dir,
                    split="train_topology_subset",
                    metadata=subset_metadata,
                    payload=subset_plan_to_payload(subset_plan),
                )
```

The `pairs`/`probabilities`/`pairwise_edges` for the returned `SplitBundle` are then
derived from the resolved `subset_plan` (whether loaded or freshly built) via a single
helper, so the cached-hit path needs no rescoring:

```python
def scored_pairs_from_subset_plan(
    plan: TopologySubsetPlan,
) -> tuple[list[tuple[str, str]], list[float]]:
    """Return (endpoints, scorer_probability) for every unique scored pair, id-ordered."""
    by_id: dict[str, tuple[str, str, float]] = {}
    for subgraph in plan.subgraphs:
        for sample in (*subgraph.positives, *subgraph.candidate_negatives):
            by_id.setdefault(
                sample.pair_id,
                (sample.protein_a, sample.protein_b, sample.scorer_probability),
            )
    endpoints = [(by_id[pid][0], by_id[pid][1]) for pid in sorted(by_id)]
    probabilities = [by_id[pid][2] for pid in sorted(by_id)]
    return endpoints, probabilities
```

(Add `scored_pairs_from_subset_plan` to `tccig/topology_subset.py` alongside
`candidate_pairs_for_scoring`, with a unit test asserting id-ordered uniqueness and that
probabilities track the stored `scorer_probability`.) The branch's tail becomes:

```python
        endpoints, probabilities = scored_pairs_from_subset_plan(subset_plan)
        pairs = [CandidatePair(protein_a, protein_b) for protein_a, protein_b in endpoints]
        pairwise_edges = edges_from_rule(
            pairs=pairs,
            probabilities=probabilities,
            rule=pairwise_input_rule,
        )
        return (
            SplitBundle(
                split="train_topology",
                pairs=pairs,
                pairwise_probabilities=probabilities,
                pairwise_graph_edges=pairwise_edges,
            ),
            subset_plan,
            coverage_stats,
        )
```

The earlier inline shape (the `subset_cfg`/score block without caching) is the
no-cache illustration; this load-before-score wrapper is its caching superset —
implement the wrapper, do not duplicate both.

- [ ] **Step 5: Run focused tests**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_subset.py tests/unit/test_tccig_topology_training.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add tccig/train.py tccig/topology_subset.py tests/unit/test_tccig_topology_subset.py
rtk git commit -m "feat: score bounded topology candidate pools"
```

---

### Task 7: Convert Epoch Subsets into Tensor Chunks

**Files:**
- Modify: `tccig/topology_subset.py`
- Modify: `tccig/s2gae.py`
- Test: `tests/unit/test_tccig_topology_subset.py`

- [ ] **Step 1: Add failing tensor chunk test**

Append to `tests/unit/test_tccig_topology_subset.py`:

```python
from tccig.topology_subset import group_epoch_samples_by_subgraph


def test_group_epoch_samples_by_subgraph_preserves_size_and_weights() -> None:
    graph = _toy_graph()
    sampled = {4: [("a", "b", "c", "d")]}
    cfg = TopologySubsetSamplerConfig(candidate_ratio=3, pool_ratio=2, epoch_ratio=1, seed=4)
    plan = build_topology_subset_plan(
        graph=graph,
        sampled_subgraphs=sampled,
        config=cfg,
        scorer_probabilities={"a||c": 0.9, "a||d": 0.2, "b||d": 0.7, "c||d": 0.1},
    )
    grouped = group_epoch_samples_by_subgraph(sample_epoch_topology_subset(plan=plan, epoch=1))
    assert list(grouped) == ["size=4:index=0"]
    chunk = grouped["size=4:index=0"]
    assert chunk.node_size == 4
    assert len(chunk.samples) >= 2
    assert all(sample.pi_total > 0.0 for sample in chunk.samples)
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_subset.py::test_group_epoch_samples_by_subgraph_preserves_size_and_weights -v
```

Expected: FAIL with import error for `group_epoch_samples_by_subgraph`.

- [ ] **Step 3: Implement grouping**

Add to `tccig/topology_subset.py`:

```python
@dataclass(frozen=True)
class TopologySubgraphEpochChunk:
    """One differentiable topology chunk for a sampled subgraph in one epoch."""

    subgraph_id: str
    node_size: int
    samples: tuple[TopologyPairSample, ...]


def group_epoch_samples_by_subgraph(
    samples: Sequence[TopologyPairSample],
) -> dict[str, TopologySubgraphEpochChunk]:
    """Group epoch samples into subgraph chunks."""
    grouped: dict[str, list[TopologyPairSample]] = defaultdict(list)
    size_by_id: dict[str, int] = {}
    for sample in samples:
        grouped[sample.subgraph_id].append(sample)
        size_by_id[sample.subgraph_id] = sample.node_size
    return {
        subgraph_id: TopologySubgraphEpochChunk(
            subgraph_id=subgraph_id,
            node_size=size_by_id[subgraph_id],
            samples=tuple(rows),
        )
        for subgraph_id, rows in sorted(grouped.items())
    }
```

- [ ] **Step 4: Run subset tests**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_subset.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add tccig/topology_subset.py tests/unit/test_tccig_topology_subset.py
rtk git commit -m "feat: group topology epoch subset chunks"
```

---

### Task 8: Add Weighted Per-Subgraph Topology Loss

**Files:**
- Modify: `tccig/s2gae.py`
- Test: `tests/unit/test_tccig_topology_training.py`

- [ ] **Step 1: Add failing test for one weighted chunk**

Append to `tests/unit/test_tccig_topology_training.py`:

```python
def test_topology_subset_chunk_loss_uses_inclusion_weights() -> None:
    from tccig.s2gae import topology_subset_chunk_loss
    from tccig.topology_subset import SamplingStratum, TopologyPairSample, TopologySubgraphEpochChunk
    from src.topology.finetune_losses import TopologyLossWeights

    chunk = TopologySubgraphEpochChunk(
        subgraph_id="size=3:index=0",
        node_size=3,
        samples=(
            TopologyPairSample(
                pair_id="a||b",
                subgraph_id="size=3:index=0",
                node_size=3,
                protein_a="a",
                protein_b="b",
                local_index_a=0,
                local_index_b=1,
                stratum=SamplingStratum.POSITIVE,
                pi_cand=1.0,
                pi_pool_given_cand=1.0,
                pi_epoch_given_pool=1.0,
                pi_total=1.0,
                target=1.0,
                scorer_probability=0.9,
            ),
            TopologyPairSample(
                pair_id="b||c",
                subgraph_id="size=3:index=0",
                node_size=3,
                protein_a="b",
                protein_b="c",
                local_index_a=1,
                local_index_b=2,
                stratum=SamplingStratum.UNIFORM_NEGATIVE,
                pi_cand=0.5,
                pi_pool_given_cand=1.0,
                pi_epoch_given_pool=1.0,
                pi_total=0.5,
                target=0.0,
                scorer_probability=0.2,
            ),
        ),
    )
    refined_logits = torch.tensor([2.0, -1.0], requires_grad=True)
    loss, components = topology_subset_chunk_loss(
        refined_logits=refined_logits,
        chunk=chunk,
        weights=TopologyLossWeights(alpha=1.0, beta=1.0, gamma=1.0, delta=0.0),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert refined_logits.grad is not None
    assert components["sample_count"] == 2.0
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_training.py::test_topology_subset_chunk_loss_uses_inclusion_weights -v
```

Expected: FAIL with import error for `topology_subset_chunk_loss`.

- [ ] **Step 3: Implement chunk loss**

In `tccig/s2gae.py`, import `TopologySubgraphEpochChunk`:

```python
from tccig.topology_subset import TopologySubgraphEpochChunk
```

Add near `topology_plan_loss`:

```python
def topology_subset_chunk_loss(
    *,
    refined_logits: torch.Tensor,
    chunk: TopologySubgraphEpochChunk,
    weights: TopologyLossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute weighted topology loss for one epoch subgraph chunk."""
    device = refined_logits.device
    pred = torch.sigmoid(refined_logits)
    pair_a = torch.tensor(
        [sample.local_index_a for sample in chunk.samples], dtype=torch.long, device=device
    )
    pair_b = torch.tensor(
        [sample.local_index_b for sample in chunk.samples], dtype=torch.long, device=device
    )
    target = torch.tensor([sample.target for sample in chunk.samples], dtype=torch.float32, device=device)
    pair_weights = torch.tensor(
        [1.0 / sample.pi_total for sample in chunk.samples],
        dtype=torch.float32,
        device=device,
    )
    terms = compute_topology_losses(
        weights=weights,
        num_nodes=chunk.node_size,
        pair_index_a=pair_a,
        pair_index_b=pair_b,
        pred_pair_probabilities=pred,
        target_pair_probabilities=target,
        pair_weights=pair_weights,
        include_clustering_mmd=False,
    )
    components = {
        "graph_sim": float(terms["graph_similarity"].detach().cpu().item()),
        "relative_density": float(terms["relative_density"].detach().cpu().item()),
        "degree_mmd": float(terms["degree_mmd"].detach().cpu().item()),
        "clustering_mmd": 0.0,
        "total": float(terms["total_topology"].detach().cpu().item()),
        "sample_count": float(len(chunk.samples)),
    }
    return terms["total_topology"], components
```

- [ ] **Step 4: Run test**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_training.py::test_topology_subset_chunk_loss_uses_inclusion_weights -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add tccig/s2gae.py tests/unit/test_tccig_topology_training.py
rtk git commit -m "feat: compute weighted topology subset chunk loss"
```

---

### Task 9: Implement Per-Size Chunked Topology Backward

**Files:**
- Modify: `tccig/s2gae.py`
- Test: `tests/unit/test_tccig_topology_training.py`

**Correctness note (review Finding 1 — gradient double-counting):** The original
draft built `_size_balanced_chunk_scales` from the *full local* chunk list on every
rank, ran *every* chunk on *every* rank (`eval()`/dropout-off makes ranks identical),
then SUM-all-reduced gradients. That multiplies the correct gradient by `world_size`
*and* normalizes per-size locally instead of globally. The fix rests on one fact: the
epoch chunk list is **deterministic and identical across ranks** (seeded
`sample_epoch_topology_subset` over a plan that is built once and shared). Therefore:

1. Every rank computes the **global** per-size normalizers from the full chunk list
   (no count all-reduce needed — the list is already identical everywhere).
2. Each rank processes a **disjoint** shard `chunks[rank::world_size]`, scaling each
   owned chunk by its *global* scale `1/(S_global * N_{s,global})`.
3. Exactly one SUM all-reduce after the loop. Because shards are disjoint and the
   scales are global, `Σ_ranks(local scaled grads) == full-objective grad` — SUM is
   correct with **no** `world_size` division.

The objective is `total = (1/S) · Σ_s [ (1/N_s) · Σ_{i∈s} loss_i ]`, so the gradient
is linear in the per-chunk scaled losses and partitions cleanly across disjoint ranks.

- [ ] **Step 1: Add failing tests for per-size scales and disjoint sharding**

Append to `tests/unit/test_tccig_topology_training.py`:

```python
def test_size_balanced_topology_normalizers_ignore_subgraph_imbalance() -> None:
    from tccig.s2gae import _size_balanced_chunk_scales

    scales = _size_balanced_chunk_scales([20, 20, 20, 200])
    assert scales == pytest.approx([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 2.0])


def test_shard_chunks_partition_is_disjoint_and_complete() -> None:
    from tccig.s2gae import _shard_chunks_for_rank

    node_sizes = [20, 20, 20, 200, 200]
    world_size = 2
    seen: list[int] = []
    for rank in range(world_size):
        shard = _shard_chunks_for_rank(
            node_sizes=node_sizes, rank=rank, world_size=world_size
        )
        # Global scale must use GLOBAL counts (S=2 sizes, N_20=3, N_200=2),
        # not the per-rank counts, regardless of which chunks land on this rank.
        for global_index, scale in shard:
            expected_n = 3 if node_sizes[global_index] == 20 else 2
            assert scale == pytest.approx(1.0 / (2 * expected_n))
            seen.append(global_index)
    # Disjoint and complete cover of every chunk index exactly once.
    assert sorted(seen) == list(range(len(node_sizes)))
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_training.py::test_size_balanced_topology_normalizers_ignore_subgraph_imbalance tests/unit/test_tccig_topology_training.py::test_shard_chunks_partition_is_disjoint_and_complete -v
```

Expected: FAIL with import errors for `_size_balanced_chunk_scales` / `_shard_chunks_for_rank`.

- [ ] **Step 3: Add global-scale and shard helpers**

Add to `tccig/s2gae.py`:

```python
def _size_balanced_chunk_scales(node_sizes: Sequence[int]) -> list[float]:
    """Return per-chunk scales for the mean-over-size, mean-over-subgraphs objective.

    `node_sizes` MUST be the full (pre-shard) epoch chunk list so the per-size counts
    are global. Each scale is `1 / (S * N_s)` with S = number of distinct sizes and
    N_s = number of chunks of that size.
    """
    counts: dict[int, int] = {}
    for size in node_sizes:
        counts[int(size)] = counts.get(int(size), 0) + 1
    active_size_count = len(counts)
    if active_size_count == 0:
        raise ValueError("topology subset chunk list must not be empty")
    return [1.0 / float(active_size_count * counts[int(size)]) for size in node_sizes]


def _shard_chunks_for_rank(
    *,
    node_sizes: Sequence[int],
    rank: int,
    world_size: int,
) -> list[tuple[int, float]]:
    """Return (global_index, global_scale) for the chunks owned by `rank`.

    The full `node_sizes` list is identical on every rank (deterministic epoch
    sampling), so global scales are computed here and the disjoint shard is taken by
    strided global index. The union over ranks covers every chunk exactly once.
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    global_scales = _size_balanced_chunk_scales(node_sizes)
    return [
        (global_index, global_scales[global_index])
        for global_index in range(len(node_sizes))
        if global_index % world_size == rank
    ]
```

- [ ] **Step 4: Add the sharded, uniform-collective backward step**

**Two hard constraints this step must honor:**

1. **Accelerate stays in charge (review: production-backend constraint).** The
   production distributed backend is Accelerate-managed. This step does **not** replace
   the launcher, model wrapping, optimizer stepping, mixed precision, or process
   lifecycle with manual `DistributedDataParallel`. It uses exactly two custom pieces,
   both already established by the current full-plan topology step (`s2gae.py:806–833`):
   `accelerator.unwrap_model(...)` (via the existing `_unwrap_refiner`) to reach the raw
   refiner, and — new for the sharded path — one explicit `torch.distributed.all_reduce`
   over the topology gradients. Per-chunk backward goes through
   `runtime.accelerator.backward(...)` (NOT raw `loss.backward()`) so the AMP grad scaler
   keeps managing scaling; clipping and `optimizer.step()` stay on the accelerator exactly
   as the existing step does. The all-reduce SUMs AMP-scaled grads, which is correct
   because Accelerate keeps the loss scale identical across ranks, and the scaler unscales
   uniformly at clip/step time.

2. **Memory bound: encode INSIDE each chunk (review: retained-graph bug).** `refiner.encode(...)`
   is autograd-tracked. Encoding once and calling `backward()` per chunk frees the shared
   encode graph on the first backward, so the second chunk's backward raises *"Trying to
   backward through the graph a second time"* unless `retain_graph=True` — which holds
   every chunk's graph alive and re-creates the exact OOM this redesign exists to fix.
   The encode is therefore recomputed **inside** each chunk's backward unit, so peak memory
   is one encode-plus-one-decode graph that is fully freed after each `backward()`. This
   trades one extra encode per chunk for a hard memory bound; that tradeoff is the whole
   point of §6.0.

Add to `tccig/s2gae.py`:

```python
def _all_reduce_topology_gradients(refiner: S2GAERefiner, runtime: TCCIGRuntime) -> None:
    """SUM topology gradients across ranks for the Accelerate-unwrapped refiner.

    This is the ONLY custom collective in the topology path; everything else
    (launcher, DDP wrapping, AMP scaler, optimizer.step) stays Accelerate-managed.
    `refiner` is the object returned by `_unwrap_refiner` (i.e. `accelerator.unwrap_model`).

    Every rank MUST reduce the SAME parameter set in the SAME order or the collective
    deadlocks. A rank whose shard was empty this epoch has no `.grad`, so we materialize
    a zero grad for every trainable parameter before reducing — making participation
    uniform. SUM is correct (no `world_size` division) because the shards are disjoint and
    the per-chunk scales are global. The grads may still be AMP-scaled here; that is fine
    because the scale is identical across ranks and Accelerate unscales uniformly at
    clip/step time.
    """
    if not runtime.is_distributed:
        return
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    for parameter in refiner.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)


def _all_reduce_component_sums(
    component_sums: dict[str, float], runtime: TCCIGRuntime
) -> dict[str, float]:
    """SUM-reduce rank-local topology component sums into global totals for logging.

    Each rank accumulated only its own shard's (globally-scaled) component contributions,
    so the per-rank dict is a partial sum of the full objective. A SUM all-reduce over the
    fixed key order reconstructs the same totals every rank would log in single-process
    mode. Detached scalars only — this never touches autograd or the optimizer. No-op when
    not distributed (the local dict is already the full sum).
    """
    if not runtime.is_distributed:
        return component_sums
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return component_sums
    keys = sorted(component_sums)
    buffer = torch.tensor(
        [component_sums[key] for key in keys],
        dtype=torch.float64,
        device=runtime.device,
    )
    torch.distributed.all_reduce(buffer, op=torch.distributed.ReduceOp.SUM)
    reduced = buffer.detach().cpu().tolist()
    return {key: float(value) for key, value in zip(keys, reduced, strict=True)}


def _topology_subset_backward_step(
    *,
    refiner: S2GAERefiner,
    graph: _SplitGraph,
    chunks: Sequence[TopologySubgraphEpochChunk],
    node_index: Mapping[str, int],
    weights: TopologyLossWeights,
    runtime: TCCIGRuntime,
    topology_scale: float,
    topology_weight: float,
) -> dict[str, float]:
    """Run sharded, memory-bounded per-chunk topology backward; return component sums.

    `chunks` is the FULL epoch chunk list, identical on every rank. This rank owns the
    disjoint shard `chunks[rank::world_size]`; each owned chunk is scaled by its global
    `1/(S*N_s)` scale and backpropagated immediately. The encode is recomputed inside the
    loop so peak memory is one chunk's forward graph (see Step 4 constraint 2). Backward
    runs through `runtime.accelerator.backward` so AMP stays managed; one explicit
    all-reduce after the loop yields the exact full-objective gradient (constraint 1).
    """
    # NOTE: the component keys MUST match what the full-plan path produces so the shared
    # epoch_history logging (`topology_components["clustering_mmd"]`, s2gae.py:920) does not
    # KeyError. Training clustering is off (Task 8), so clustering_mmd is a constant 0.0.
    if not chunks:
        return {
            "total": 0.0,
            "graph_sim": 0.0,
            "relative_density": 0.0,
            "degree_mmd": 0.0,
            "clustering_mmd": 0.0,
        }
    rank = runtime.rank if runtime.is_distributed else 0
    world_size = runtime.world_size if runtime.is_distributed else 1
    shard = _shard_chunks_for_rank(
        node_sizes=[chunk.node_size for chunk in chunks],
        rank=rank,
        world_size=world_size,
    )
    component_sums = {
        "total": 0.0,
        "graph_sim": 0.0,
        "relative_density": 0.0,
        "degree_mmd": 0.0,
        "clustering_mmd": 0.0,
    }
    for global_index, scale in shard:
        chunk = chunks[global_index]
        # Recompute encode INSIDE the chunk so its graph is freed by this chunk's
        # backward; do NOT hoist this out of the loop (would need retain_graph -> OOM).
        hidden_states = refiner.encode(
            node_features=graph.node_features,
            edge_index=graph.edge_index,
            edge_weight=graph.edge_weight,
        )
        global_pairs = (
            torch.tensor(
                [[node_index[sample.protein_a], node_index[sample.protein_b]] for sample in chunk.samples],
                dtype=torch.long,
                device=graph.node_features.device,
            )
            .t()
            .contiguous()
        )
        refined_logits, _ = refiner.decode(
            hidden_states=hidden_states,
            pair_index=global_pairs,
            pairwise_probabilities=graph.pairwise_probabilities[_pair_lookup(graph.pair_index, global_pairs)],
        )
        chunk_loss, components = topology_subset_chunk_loss(
            refined_logits=refined_logits,
            chunk=chunk,
            weights=weights,
        )
        scaled = topology_scale * topology_weight * scale * chunk_loss
        # Accelerate-managed backward (keeps AMP scaler); accumulates into refiner.grad.
        runtime.accelerator.backward(scaled)
        for key in component_sums:
            component_sums[key] += components[key] * scale
    _all_reduce_topology_gradients(refiner, runtime)
    # component_sums are RANK-LOCAL: each rank only walked its own disjoint shard. The
    # gradient is made global by the all-reduce above, but these detached scalars are not,
    # so SUM-reduce them too — otherwise the logged train_topology_* values would report
    # only this rank's shard. No-op in single-process mode.
    return _all_reduce_component_sums(component_sums, runtime)
```

**Why this is correct (and the old version was not):** in single-process mode
`world_size == 1`, the shard is the full list, scales are global == local, no all-reduce
runs, and the result matches the reference exactly. In multi-rank mode each chunk is owned
by exactly one rank, the refiner parameters accumulate only that shard's (AMP-scaled)
contribution via `accelerator.backward`, and the post-loop SUM reconstructs
`scale_loss · Σ_i scale_i · ∂loss_i/∂θ` — identical (after the scaler's uniform unscale at
step time) to a single process that ran every chunk. No `world_size` factor and no manual
DDP appear anywhere; Accelerate still owns the launcher, wrapping, AMP, and optimizer.

- [ ] **Step 5: Run focused tests**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_training.py::test_size_balanced_topology_normalizers_ignore_subgraph_imbalance tests/unit/test_tccig_topology_training.py::test_topology_subset_chunk_loss_uses_inclusion_weights -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add tccig/s2gae.py tests/unit/test_tccig_topology_training.py
rtk git commit -m "feat: add chunked topology subset backward"
```

---

### Task 10: Wire Subset Backward into `train_refiner`

**Files:**
- Modify: `tccig/s2gae.py`
- Test: `tests/unit/test_tccig_topology_training.py`

- [ ] **Step 1: Add failing smoke-level unit test for subset branch selection**

Append to `tests/unit/test_tccig_topology_training.py`:

```python
def test_train_refiner_accepts_subset_plan_object() -> None:
    from tccig.s2gae import TrainRefinerRequest
    from tccig.topology_subset import TopologySubsetPlan

    request = TrainRefinerRequest(
        train=None,  # type: ignore[arg-type]
        validation=None,  # type: ignore[arg-type]
        runtime=None,  # type: ignore[arg-type]
        config={},
        graph_rule=None,  # type: ignore[arg-type]
        train_topology_plan=TopologySubsetPlan(
            subgraphs=(),
            active_sizes=(),
            skipped_sizes={},
            total_positive_pairs=0,
            total_candidate_negatives=0,
            total_pool_negatives=0,
        ),
    )
    assert isinstance(request.train_topology_plan, TopologySubsetPlan)
```

- [ ] **Step 2: Run test**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_training.py::test_train_refiner_accepts_subset_plan_object -v
```

Expected: PASS if the request type is permissive; FAIL if type checking or imports need adjustment.

- [ ] **Step 3: Replace topology branch with subset-aware path**

Modify imports in `tccig/s2gae.py`:

```python
from tccig.topology_subset import (
    TopologySubsetPlan,
    group_epoch_samples_by_subgraph,
    sample_epoch_topology_subset,
)
```

**Distributed-backend constraint (review):** The production backend stays
**Accelerate-managed**. Do not replace the launcher, model wrapping (`accelerator.prepare`),
optimizer stepping, mixed precision, or process lifecycle with hand-rolled
`DistributedDataParallel`. The subset topology step keeps the *exact* surrounding
contract the full-plan path already uses: `optimizer.zero_grad(...)` →
`_unwrap_refiner(...)` (i.e. `accelerator.unwrap_model`) → per-chunk
`accelerator.backward(...)` inside `_topology_subset_backward_step` → one explicit
`torch.distributed.all_reduce` over the unwrapped refiner's grads → Accelerate's
`clip_grad_norm_` → `optimizer.step()`. The only additions beyond the existing path
are the unwrap (already present) and the single custom all-reduce; everything else
remains Accelerate's job.

Inside the existing topology block in `train_refiner`, replace only the body under `if topology_scale > 0.0 ...` with:

```python
                optimizer.zero_grad(set_to_none=True)
                topology_refiner = _unwrap_refiner(
                    train_step_model, request.runtime.accelerator
                )
                topology_was_training = topology_refiner.training
                topology_refiner.eval()
                try:
                    if isinstance(request.train_topology_plan, TopologySubsetPlan):
                        epoch_samples = sample_epoch_topology_subset(
                            plan=request.train_topology_plan,
                            epoch=epoch,
                            config=cfg.topology_training.subset,
                        )
                        grouped = group_epoch_samples_by_subgraph(epoch_samples)
                        chunks = tuple(grouped[subgraph_id] for subgraph_id in sorted(grouped))
                        topology_components = _topology_subset_backward_step(
                            refiner=topology_refiner,
                            graph=train_topology_graph,
                            chunks=chunks,
                            node_index=train_topology_node_index,
                            weights=cfg.topology_training.weights,
                            runtime=request.runtime,
                            topology_scale=topology_scale,
                            topology_weight=cfg.topology_training.topology_weight,
                        )
                    else:
                        topo_loss, topology_components = topology_plan_loss(
                            refiner=topology_refiner,
                            graph=train_topology_graph,
                            plan=cast(InternalValidationPlan, request.train_topology_plan),
                            node_index=train_topology_node_index,
                            weights=cfg.topology_training.weights,
                            include_clustering_mmd=cfg.topology_validation.compute_clustering_mmd,
                        )
                        scaled = topology_scale * cfg.topology_training.topology_weight * topo_loss
                        request.runtime.accelerator.backward(scaled)
                finally:
                    topology_refiner.train(topology_was_training)
                if cfg.optimization.gradient_clip_norm is not None:
                    request.runtime.accelerator.clip_grad_norm_(
                        train_step_model.parameters(), cfg.optimization.gradient_clip_norm
                    )
                optimizer.step()
```

- [ ] **Step 4: Run topology tests**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_training.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add tccig/s2gae.py tests/unit/test_tccig_topology_training.py
rtk git commit -m "feat: train refiner with topology subset plan"
```

---

### Task 11: Add Distributed Gradient Equivalence Test

**Files:**
- Create: `tests/unit/test_tccig_topology_distributed.py`
- Modify: `tccig/s2gae.py`

**Correctness note (review Finding 6 — spec §12 acceptance):** The math-only tests
below pin the scale arithmetic and the no-op path, but the spec's actual acceptance
criterion is end-to-end: *"fork (b) per-chunk backward + manual SUM all-reduce produces
gradients equal to a single-process full-plan reference."* Step 1 therefore adds a real
two-rank `gloo`/CPU test that spawns two processes, has each shard the chunk list with
`_shard_chunks_for_rank`, scales by the global scale, backpropagates its disjoint shard,
SUM-all-reduces, and asserts the resulting parameter gradient equals a single-process
run over the full chunk list. This is the test that actually guards against the
`world_size` double-count regression.

- [ ] **Step 1: Write CPU gradient equivalence tests (math + real 2-rank)**

**Test-harness scope (review):** This file spins up its own bare `gloo` process
group purely to exercise the gradient math. It is **not** the production backend and
must not be taken as a template for the training launcher: production keeps the
Accelerate-managed launcher, model wrapping, optimizer, mixed precision, and process
lifecycle (Task 9/10). The only production overlap is `_all_reduce_topology_gradients`
+ `_shard_chunks_for_rank`, which this test calls directly against a `TinyModel` to
prove the SUM-over-disjoint-shards math equals a single-process reference.

Create `tests/unit/test_tccig_topology_distributed.py`:

```python
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


def _worker(rank: int, world_size: int, file_path: str, return_dict: dict) -> None:
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
            return_dict["grad"] = model.weight.grad.detach().clone()
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
    manager = mp.Manager()
    return_dict = manager.dict()
    rendezvous = str(tmp_path / "rendezvous")
    mp.spawn(
        _worker,
        args=(world_size, rendezvous, return_dict),
        nprocs=world_size,
        join=True,
    )
    assert "grad" in return_dict, "rank 0 did not report a gradient"
    # SUM all-reduce over disjoint, globally-scaled shards must equal the full-objective
    # reference with NO world_size factor. A world_size double-count would make this 2x.
    torch.testing.assert_close(return_dict["grad"], reference)
```

- [ ] **Step 2: Run distributed math + 2-rank tests**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_distributed.py -v
```

Expected: PASS (the 2-rank test runs on `gloo`/CPU; it self-skips only if `gloo`
is unavailable in the environment).

- [ ] **Step 3: Commit**

```bash
rtk git add tests/unit/test_tccig_topology_distributed.py tccig/s2gae.py
rtk git commit -m "test: cover topology distributed gradient equivalence"
```

---

### Task 12: Add Scoring Progress Logging

**Files:**
- Modify: `tccig/train.py`
- Test: `tests/unit/test_tccig_topology_training.py`

- [ ] **Step 1: Add failing progress callback test**

Append to `tests/unit/test_tccig_topology_training.py`:

```python
def test_score_progress_interval_emits_periodic_events() -> None:
    from tccig.train import _score_progress_events

    events = list(_score_progress_events(total_pairs=1000, interval_pairs=250))
    assert events == [250, 500, 750, 1000]


def test_score_progress_pointer_fires_when_batch_overshoots_milestone() -> None:
    # Finding 6: `processed` advances by batch size and lands BETWEEN milestones, so an
    # exact `processed in milestones` test would never fire. Replicate the closure's
    # advancing-pointer logic and assert every crossed milestone is drained exactly once.
    from tccig.train import _score_progress_events

    milestones = _score_progress_events(total_pairs=1000, interval_pairs=250)
    fired: list[int] = []
    pointer = 0
    # Batches overshoot 250 (->300), 500/750 in one jump (->760), then finish (->1000).
    for processed in (300, 760, 1000):
        while pointer < len(milestones) and processed >= milestones[pointer]:
            fired.append(milestones[pointer])
            pointer += 1
    # Every milestone fires once, in order, despite none equalling a `processed` value.
    assert fired == [250, 500, 750, 1000]
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_training.py::test_score_progress_interval_emits_periodic_events -v
```

Expected: FAIL with import error for `_score_progress_events`.

- [ ] **Step 3: Implement progress helper and callback logging**

Add to `tccig/train.py`:

```python
def _score_progress_events(*, total_pairs: int, interval_pairs: int) -> list[int]:
    """Return pair-count milestones for score progress logging."""
    if total_pairs <= 0:
        return []
    if interval_pairs <= 0:
        raise ValueError("interval_pairs must be positive")
    events = list(range(interval_pairs, total_pairs, interval_pairs))
    events.append(total_pairs)
    return events
```

In `_score_split`, before `score_pairs_with_v3_1`, log:

```python
    LOGGER.info("tccig scoring split=%s pair_count=%s cache=miss", split, len(pairs))
```

Call `score_pairs_with_v3_1` with a callback. **Finding 6:** the callback must NOT test
`processed in milestones`. `processed` advances by `batch_size` each call
(`processed += int(indices.numel())`), so it will routinely step *over* an exact milestone
value (e.g. milestones at 250_000 but `processed` lands on 248_320 then 251_904) and the
log would silently never fire. Use a single advancing pointer and fire on
`processed >= next_milestone`, draining every milestone the step crossed. Also label the
count as **rank-local**: `processed_pairs` is this rank's shard count, not the global
total, so the message says `processed_local` to avoid implying a global figure under DDP.

```python
    milestones = _score_progress_events(total_pairs=len(pairs), interval_pairs=250_000)
    # Mutable pointer into `milestones`; closure advances it as `processed` crosses each.
    next_milestone_index = 0

    def _progress(payload: Mapping[str, object]) -> None:
        nonlocal next_milestone_index
        if not runtime.is_main_process:
            return
        processed = int(payload["processed_pairs"])
        # Drain every milestone this step crossed (a single batch may pass several),
        # using >= so a milestone is never skipped when `processed` overshoots it.
        while (
            next_milestone_index < len(milestones)
            and processed >= milestones[next_milestone_index]
        ):
            LOGGER.info(
                "tccig scoring progress split=%s processed_local=%s total_local=%s",
                split,
                processed,
                len(pairs),
            )
            next_milestone_index += 1

    probabilities = score_pairs_with_v3_1(
        pairs=pairs,
        config=scorer_cfg,
        runtime=runtime,
        progress_callback=_progress,
    )
```

Note: `len(pairs)` is also rank-local here (each rank scores its own shard via
`_score_split`'s internal sharding), which is why both fields are suffixed `_local`. A
global total would require an all-reduce that the progress path deliberately avoids — the
bounded subset frame keeps per-rank counts small enough that local progress is sufficient
for an operator to see the scoring loop is alive.

- [ ] **Step 4: Run progress test**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_training.py::test_score_progress_interval_emits_periodic_events -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add tccig/train.py tests/unit/test_tccig_topology_training.py
rtk git commit -m "feat: log topology scoring progress"
```

---

### Task 13: Add Smoke and Main Configs

**Files:**
- Create: `configs/tccig/02_balanced_subset.yaml`
- Create: `configs/tccig/02_balanced_subset_smoke.yaml`
- Test: `tests/unit/test_tccig_topology_training.py`

- [ ] **Step 1: Add config parse smoke test**

Append to `tests/unit/test_tccig_topology_training.py`:

```python
def test_balanced_subset_configs_parse() -> None:
    import yaml
    from pathlib import Path
    from tccig.s2gae import _parse_config

    for path in (
        Path("configs/tccig/02_balanced_subset.yaml"),
        Path("configs/tccig/02_balanced_subset_smoke.yaml"),
    ):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        cfg = _parse_config(config["refiner"])
        assert cfg.topology_training.subset.enabled is True
        # Review Finding 11: clustering stays ON in validation/test metrics (spec §2
        # non-goal). The subset TRAINING path keeps clustering off independently
        # (Task 8 hardcodes include_clustering_mmd=False in the chunk loss).
        assert cfg.topology_validation.compute_clustering_mmd is True


def test_smoke_config_engages_topology_in_epoch_one() -> None:
    # Review Finding 3: the smoke config must reach a POSITIVE topology scale in
    # epoch 1, otherwise the smoke run silently exercises none of the new path.
    # train_refiner calls topology_loss_scale(epoch=epoch-1, ...), so epoch 1 uses
    # index 0. With warmup_epochs=0 and ramp_epochs=0 the scale must be > 0 there.
    import yaml
    from pathlib import Path
    from tccig.s2gae import _parse_config
    from src.topology.finetune_losses import (
        TopologyLossWeightSchedule,
        topology_loss_scale,
    )

    config = yaml.safe_load(
        Path("configs/tccig/02_balanced_subset_smoke.yaml").read_text(encoding="utf-8")
    )
    cfg = _parse_config(config["refiner"])
    # train_refiner builds the schedule from the three flat training-config fields
    # (see tccig/s2gae.py: TopologyLossWeightSchedule(...)). Mirror that here.
    schedule = TopologyLossWeightSchedule(
        warmup_epochs=cfg.topology_training.warmup_epochs,
        ramp_epochs=cfg.topology_training.ramp_epochs,
        schedule=cfg.topology_training.schedule,
    )
    scale_epoch_one = topology_loss_scale(epoch=0, schedule=schedule)
    assert scale_epoch_one > 0.0
```

- [ ] **Step 2: Create main config**

Create `configs/tccig/02_balanced_subset.yaml` by copying `configs/tccig/02.yaml` and applying these exact changes:

```yaml
run:
  run_id: "02_balanced_subset"

refiner:
  topology_training:
    enabled: true
    node_sizes: [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    samples_per_size: 20
    strategy: mixed
    seed: 0
    coverage_augmentation: true
    topology_weight: 1.0
    subset:
      enabled: true
      candidate_ratio: 20
      pool_ratio: 10
      epoch_ratio: 5
      hard_fraction: 0.5
      uniform_fraction: 0.5
      hard_stratum_fraction: 0.2
      # Spec §4 per-size budget (0 == unbounded). Caps the largest buckets so the
      # 200-node size cannot dominate scoring memory or the per-size objective.
      max_subgraphs_per_size: 20
      max_labeled_pairs_per_size: 0
      # Spec §3.7 production bias diagnostic: IPW-vs-exact on capped subgraphs every
      # N epochs. 0 == off.
      bias_diagnostic_every_n_epochs: 5
      bias_diagnostic_max_node_size: 40
      seed: 0
    weights:
      alpha: 1.0
      beta: 8.0
      gamma: 0.5
      delta: 0.0
    schedule:
      warmup_epochs: 1
      ramp_epochs: 5
      schedule: linear
  topology_validation:
    enabled: true
    node_sizes: [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    samples_per_size: 20
    strategy: mixed
    seed: 0
    inference_batch_size: 4096
    # Finding 11: this knob ALSO drives the validation/test topology metrics
    # (s2gae.py:1340/1350), whose definitions are a spec non-goal to change. The
    # subset training path keeps `clustering` off on its own (Task 8 hardcodes
    # include_clustering_mmd=False), so leave validation clustering ON here.
    compute_clustering_mmd: true
    losses:
      alpha: 1.0
      beta: 8.0
      gamma: 0.5
      delta: 0.0
```

Keep all other fields from `configs/tccig/02.yaml`.

- [ ] **Step 3: Create smoke config**

Create `configs/tccig/02_balanced_subset_smoke.yaml` by copying the main config and applying:

```yaml
run:
  run_id: "02_balanced_subset_smoke"

refiner:
  epochs: 2
  topology_training:
    node_sizes: [20, 40]
    samples_per_size: 2
    subset:
      enabled: true
      candidate_ratio: 4
      pool_ratio: 2
      epoch_ratio: 2
      hard_fraction: 0.5
      uniform_fraction: 0.5
      hard_stratum_fraction: 0.5
      # Smoke run exercises the §9 sanity check on epoch 1; production §3.7 cadence
      # is irrelevant in a 2-epoch run but kept on so the code path is covered.
      bias_diagnostic_every_n_epochs: 1
      bias_diagnostic_max_node_size: 40
      seed: 0
    schedule:
      warmup_epochs: 0
      ramp_epochs: 0
      schedule: linear
  topology_validation:
    node_sizes: [20, 40]
    samples_per_size: 2
    # Finding 11: leave clustering ON in validation metrics (definitions unchanged).
    compute_clustering_mmd: true
```

**Schedule correctness (review Finding 3):** the training step calls
`topology_loss_scale(epoch=epoch - 1, schedule=...)`. With `warmup_epochs: 0,
ramp_epochs: 1`, epoch 1 gives `epoch - 1 = 0` and `progress = 0 / 1 = 0.0`, so the
scale is **0.0** and the smoke run never exercises a real topology backward — defeating
the smoke gate. `ramp_epochs: 0` makes `topology_loss_scale` return `1.0` immediately,
so epoch 1 reaches a positive scale. (The main config keeps a real warmup/ramp; only the
2-epoch smoke config needs immediate engagement.)

- [ ] **Step 4: Run config parse test**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_training.py::test_balanced_subset_configs_parse -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add configs/tccig/02_balanced_subset.yaml configs/tccig/02_balanced_subset_smoke.yaml tests/unit/test_tccig_topology_training.py
rtk git commit -m "feat: add balanced subset tccig configs"
```

---

### Task 14: Implement Bias Diagnostic (§3.7 production + §9 smoke)

**Files:**
- Modify: `tccig/topology_subset.py`
- Modify: `tccig/s2gae.py`
- Test: `tests/unit/test_tccig_topology_subset.py`

**Correctness note (review Findings 4 and 5 — diagnostics were never implemented):**
The previous draft only added a `relative_error` helper and logged subset *totals*.
That is not the spec's diagnostic. The spec requires comparing the **IPW-reweighted
subset statistic** against the **exact full-space statistic** and logging per-metric
relative error, in two places:

- **§9 smoke sanity check:** once, on a held-out subgraph, to catch wiring/reweighting
  bugs (wrong `π_i`, missing weight, normalizer mismatch).
- **§3.7 production diagnostic:** every `bias_diagnostic_every_n_epochs` (e.g. 5), on a
  few *capped* subgraphs (small enough that the full `n·(n−1)` space is affordable),
  to track the §3.4 approximation under the real size mixture.

Both reduce to the same primitive: given a subgraph's full pair set with scorer
probabilities (the "full space") and a `π_i`-sampled subset of it, compute density and
mean-degree two ways and compare. Implementing one `compute_subset_bias_diagnostic`
primitive serves both call sites.

- [ ] **Step 1: Add failing diagnostic tests**

Append to `tests/unit/test_tccig_topology_subset.py`:

```python
from tccig.topology_subset import compute_subset_bias_diagnostic, relative_error


def test_relative_error_handles_zero_reference() -> None:
    assert relative_error(estimate=0.0, reference=0.0) == 0.0
    assert relative_error(estimate=1.0, reference=0.0) == 1.0
    assert relative_error(estimate=9.0, reference=10.0) == pytest.approx(0.1)


def test_bias_diagnostic_recovers_full_space_density_under_ipw() -> None:
    # Full space: a 4-node subgraph, all 6 upper-triangle pairs with known probs.
    # The IPW estimate from a pi-sampled subset must approximately recover the exact
    # full-space density (Horvitz-Thompson unbiasedness of the linear numerator).
    full_probs = {
        "a||b": 0.9,
        "a||c": 0.1,
        "a||d": 0.7,
        "b||c": 0.3,
        "b||d": 0.5,
        "c||d": 0.2,
    }
    # A subset that kept every pair (pi=1) must give EXACTLY the full-space stats.
    subset = [(pair_id, prob, 1.0) for pair_id, prob in full_probs.items()]
    diagnostic = compute_subset_bias_diagnostic(
        node_size=4,
        full_space_probabilities=full_probs,
        subset_samples=subset,
    )
    assert diagnostic["density_relative_error"] == pytest.approx(0.0, abs=1e-9)
    assert diagnostic["mean_degree_relative_error"] == pytest.approx(0.0, abs=1e-9)


def test_bias_diagnostic_flags_missing_weight() -> None:
    # If a down-sampled pair (pi=0.5) is NOT reweighted (weight forced to 1.0), the
    # density estimate is biased low and the diagnostic must report nonzero error.
    full_probs = {"a||b": 1.0, "a||c": 1.0, "b||c": 1.0}
    # Keep only 2 of 3 pairs, each with pi=0.5 but WRONG weight 1.0 (bug simulation).
    subset = [("a||b", 1.0, 1.0), ("a||c", 1.0, 1.0)]
    diagnostic = compute_subset_bias_diagnostic(
        node_size=3,
        full_space_probabilities=full_probs,
        subset_samples=subset,
    )
    assert diagnostic["density_relative_error"] > 0.1


def test_select_diagnostic_subgraphs_spreads_across_size_mixture() -> None:
    # _select_diagnostic_subgraphs lives in s2gae.py (it needs only plan/chunk metadata,
    # no model), so a tight max_subgraphs budget must still sample the SIZE MIXTURE
    # (round-robin one per active size first), not just the smallest size (Finding 3).
    from tccig.s2gae import _select_diagnostic_subgraphs
    from tccig.topology_subset import (
        TopologySubgraphEpochChunk,
        TopologySubgraphPlan,
        TopologySubsetPlan,
    )

    def _plan_subgraph(size: int, index: int) -> TopologySubgraphPlan:
        return TopologySubgraphPlan(
            subgraph_id=f"size={size}:index={index}",
            node_size=size,
            nodes=tuple(f"s{size}_n{index}_{j}" for j in range(size)),
            positives=(),
            candidate_negatives=(),
            hard_pool=(),
            uniform_pool=(),
        )

    subgraphs = tuple(
        _plan_subgraph(size, index) for size in (4, 8) for index in range(3)
    )
    plan = TopologySubsetPlan(
        subgraphs=subgraphs,
        active_sizes=(4, 8),
        skipped_sizes={},
        total_positive_pairs=0,
        total_candidate_negatives=0,
        total_pool_negatives=0,
    )
    # Every subgraph produced epoch samples this round.
    chunk_by_id = {
        sg.subgraph_id: TopologySubgraphEpochChunk(
            subgraph_id=sg.subgraph_id, node_size=sg.node_size, samples=()
        )
        for sg in subgraphs
    }
    selected = _select_diagnostic_subgraphs(
        plan=plan, chunk_by_id=chunk_by_id, max_node_size=40, max_subgraphs=2
    )
    # Budget of 2 must take one subgraph from EACH active size, not two from size 4.
    assert {sg.node_size for sg in selected} == {4, 8}
    # max_node_size filters out sizes above the cap.
    capped = _select_diagnostic_subgraphs(
        plan=plan, chunk_by_id=chunk_by_id, max_node_size=4, max_subgraphs=0
    )
    assert {sg.node_size for sg in capped} == {4}
    assert len(capped) == 3  # 0 == every eligible subgraph
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_subset.py::test_relative_error_handles_zero_reference tests/unit/test_tccig_topology_subset.py::test_bias_diagnostic_recovers_full_space_density_under_ipw tests/unit/test_tccig_topology_subset.py::test_bias_diagnostic_flags_missing_weight -v
```

Expected: FAIL with import errors for `relative_error` / `compute_subset_bias_diagnostic`.

- [ ] **Step 3: Implement the diagnostic primitive**

Add to `tccig/topology_subset.py`:

```python
def relative_error(*, estimate: float, reference: float) -> float:
    """Return bounded relative error for diagnostic logging."""
    if reference == 0.0:
        return 0.0 if estimate == 0.0 else 1.0
    return abs(estimate - reference) / abs(reference)


def compute_subset_bias_diagnostic(
    *,
    node_size: int,
    full_space_probabilities: Mapping[str, float],
    subset_samples: Sequence[tuple[str, float, float]],
) -> dict[str, float]:
    """Compare IPW-reweighted subset statistics against exact full-space statistics.

    Shared primitive for the §9 smoke sanity check and the §3.7 production diagnostic.

    - `full_space_probabilities`: every upper-triangle pair id -> predicted probability
      over the FULL `n·(n-1)/2` pair space of one (capped) subgraph.
    - `subset_samples`: `(pair_id, predicted_probability, weight)` for the pairs the
      sampler actually selected, where `weight == 1 / pi_total`.

    Returns per-metric relative error of the IPW subset estimate vs the exact full-space
    value for density and mean soft-degree (both linear accumulators, so HT-unbiased).
    """
    if node_size < 2:
        raise ValueError("node_size must be >= 2")
    normalizer = float(node_size * (node_size - 1))
    # Exact full-space statistics.
    full_sum = sum(full_space_probabilities.values())
    full_density = (2.0 * full_sum) / normalizer
    # IPW subset estimates (Horvitz-Thompson) of the same linear accumulators.
    ipw_sum = sum(prob * weight for _pair_id, prob, weight in subset_samples)
    ipw_density = (2.0 * ipw_sum) / normalizer
    # Mean soft degree = (2 * sum of pair probs) / num_nodes for both views.
    full_mean_degree = (2.0 * full_sum) / float(node_size)
    ipw_mean_degree = (2.0 * ipw_sum) / float(node_size)
    return {
        "density_estimate": ipw_density,
        "density_reference": full_density,
        "density_relative_error": relative_error(
            estimate=ipw_density, reference=full_density
        ),
        "mean_degree_estimate": ipw_mean_degree,
        "mean_degree_reference": full_mean_degree,
        "mean_degree_relative_error": relative_error(
            estimate=ipw_mean_degree, reference=full_mean_degree
        ),
        "full_space_pairs": float(len(full_space_probabilities)),
        "subset_pairs": float(len(subset_samples)),
    }
```

- [ ] **Step 4: Parse the `bias_diagnostic_*` config knobs**

The `bias_diagnostic_every_n_epochs`, `bias_diagnostic_max_node_size`, and
`bias_diagnostic_max_subgraphs` fields are already defined and validated on
`TopologySubsetSamplerConfig` (Task 1) and parsed in `_parse_topology_subset_config`
(Task 5). No additional parsing work is needed here; this is the same block already shown
in Task 5:

```python
        bias_diagnostic_every_n_epochs=_non_negative_int(
            raw.get("bias_diagnostic_every_n_epochs", 0),
            "refiner.topology_training.subset.bias_diagnostic_every_n_epochs",
        ),
        bias_diagnostic_max_node_size=_non_negative_int(
            raw.get("bias_diagnostic_max_node_size", 40),
            "refiner.topology_training.subset.bias_diagnostic_max_node_size",
        ),
        bias_diagnostic_max_subgraphs=_non_negative_int(
            raw.get("bias_diagnostic_max_subgraphs", 4),
            "refiner.topology_training.subset.bias_diagnostic_max_subgraphs",
        ),
```

- [ ] **Step 5: Wire both diagnostic call sites in `tccig/s2gae.py`**

The diagnostic needs the **full-space** predicted probabilities for a capped subgraph.
Those come from the refiner's current `pred = sigmoid(refined_logits)` over the
subgraph's full upper-triangle pair set, and the subset view comes from the epoch
samples for that same subgraph (their `pred` and `weight = 1/pi_total`).

Add a helper `_topology_bias_diagnostic_step(...)` near `_topology_subset_backward_step`.
It samples up to `max_subgraphs` eligible subgraphs **spread across the active size
mixture** (smallest-first within each size, but at least one per active size up to the
budget) so bias from larger sizes is exposed too, not just the single smallest subgraph
(review: production-diagnostic-breadth finding). For each chosen subgraph it decodes the
refiner over that subgraph's FULL `n·(n-1)/2` pair set under `torch.no_grad()`, reads the
IPW subset view straight off the epoch chunk, and defers the unbiasedness math to
`compute_subset_bias_diagnostic`; the per-subgraph diagnostics are then averaged (and the
max relative error is also reported, so one badly-biased size cannot hide behind good
ones). A subgraph is eligible only if it (a) has `node_size <= max_node_size` and (b)
actually appears in this epoch's `chunks` (non-empty IPW view). The encode/decode call
shape mirrors `_topology_subset_backward_step` so it stays correct if the refiner API
changes.

`_all_local_pairs` lives in `tccig.topology_subset`; the diagnostic also needs
`TopologySubgraphPlan` and `compute_subset_bias_diagnostic` (and `_pair_lookup` /
`TopologySubgraphEpochChunk`, already module-level / imported in `s2gae.py` from Task 8).
Extend the existing `tccig.topology_subset` import block (from Task 10) and add the
stdlib `defaultdict` if it is not already imported:

```python
from collections import defaultdict  # add if not already present in s2gae.py

from tccig.topology_subset import (
    TopologySubgraphEpochChunk,
    TopologySubgraphPlan,
    TopologySubsetPlan,
    _all_local_pairs,
    compute_subset_bias_diagnostic,
    group_epoch_samples_by_subgraph,
    sample_epoch_topology_subset,
)
```

```python
def _subgraph_bias_diagnostic(
    *,
    refiner: S2GAERefiner,
    graph: _SplitGraph,
    subgraph: TopologySubgraphPlan,
    chunk: TopologySubgraphEpochChunk,
    node_index: Mapping[str, int],
) -> dict[str, float] | None:
    """IPW-vs-full-space bias diagnostic for ONE capped subgraph. No grad."""
    nodes = tuple(subgraph.nodes)
    full_pairs = _all_local_pairs(nodes)  # (idx_a, idx_b, protein_a, protein_b, pair_id)
    device = graph.node_features.device
    with torch.no_grad():
        hidden_states = refiner.encode(
            node_features=graph.node_features,
            edge_index=graph.edge_index,
            edge_weight=graph.edge_weight,
        )
        global_pairs = (
            torch.tensor(
                [[node_index[row[2]], node_index[row[3]]] for row in full_pairs],
                dtype=torch.long,
                device=device,
            )
            .t()
            .contiguous()
        )
        refined_logits, _ = refiner.decode(
            hidden_states=hidden_states,
            pair_index=global_pairs,
            pairwise_probabilities=graph.pairwise_probabilities[
                _pair_lookup(graph.pair_index, global_pairs)
            ],
        )
        full_probabilities = torch.sigmoid(refined_logits).detach().cpu().tolist()
    full_space_probabilities = {
        row[4]: float(prob) for row, prob in zip(full_pairs, full_probabilities, strict=True)
    }
    # IPW subset view: this epoch's drawn pairs for the SAME subgraph. pred is read from
    # the exact full-space decode above (the chunk pairs are a subset of the full set), so
    # estimate and reference share one decode and differ only by sampling + reweighting.
    subset_samples = [
        (sample.pair_id, full_space_probabilities[sample.pair_id], 1.0 / sample.pi_total)
        for sample in chunk.samples
        if sample.pair_id in full_space_probabilities
    ]
    if not subset_samples:
        return None
    return compute_subset_bias_diagnostic(
        node_size=subgraph.node_size,
        full_space_probabilities=full_space_probabilities,
        subset_samples=subset_samples,
    )


def _select_diagnostic_subgraphs(
    *,
    plan: TopologySubsetPlan,
    chunk_by_id: Mapping[str, TopologySubgraphEpochChunk],
    max_node_size: int,
    max_subgraphs: int,
) -> list[TopologySubgraphPlan]:
    """Pick eligible subgraphs spread across active sizes (smallest-first within size).

    Round-robins one subgraph per active size before taking a second from any size, so a
    tight `max_subgraphs` budget still samples the size MIXTURE rather than only the
    smallest size. `max_subgraphs == 0` means every eligible subgraph.
    """
    by_size: dict[int, list[TopologySubgraphPlan]] = defaultdict(list)
    for subgraph in plan.subgraphs:
        if subgraph.node_size <= max_node_size and subgraph.subgraph_id in chunk_by_id:
            by_size[subgraph.node_size].append(subgraph)
    for size in by_size:
        by_size[size].sort(key=lambda sg: sg.subgraph_id)
    ordered_sizes = sorted(by_size)
    selected: list[TopologySubgraphPlan] = []
    cap = max_subgraphs if max_subgraphs > 0 else None
    cursor = {size: 0 for size in ordered_sizes}
    while ordered_sizes and (cap is None or len(selected) < cap):
        progressed = False
        for size in ordered_sizes:
            if cap is not None and len(selected) >= cap:
                break
            index = cursor[size]
            if index < len(by_size[size]):
                selected.append(by_size[size][index])
                cursor[size] = index + 1
                progressed = True
        if not progressed:
            break
    return selected


def _topology_bias_diagnostic_step(
    *,
    refiner: S2GAERefiner,
    graph: _SplitGraph,
    plan: TopologySubsetPlan,
    chunks: Sequence[TopologySubgraphEpochChunk],
    node_index: Mapping[str, int],
    max_node_size: int,
    max_subgraphs: int,
) -> dict[str, float] | None:
    """Aggregate IPW-vs-full-space bias across a few capped subgraphs of the size mixture.

    Returns ``None`` when no eligible subgraph produced epoch samples this round (e.g.
    every active size exceeds ``max_node_size``). Otherwise returns mean and max relative
    error across the sampled subgraphs (so one badly-biased size cannot hide behind the
    others), plus the subgraph count and total full/subset pair counts. No grad is taken;
    this never touches the optimizer state.
    """
    if max_node_size < 2 or not chunks:
        return None
    chunk_by_id = {chunk.subgraph_id: chunk for chunk in chunks}
    targets = _select_diagnostic_subgraphs(
        plan=plan,
        chunk_by_id=chunk_by_id,
        max_node_size=max_node_size,
        max_subgraphs=max_subgraphs,
    )
    diagnostics: list[dict[str, float]] = []
    for subgraph in targets:
        per_subgraph = _subgraph_bias_diagnostic(
            refiner=refiner,
            graph=graph,
            subgraph=subgraph,
            chunk=chunk_by_id[subgraph.subgraph_id],
            node_index=node_index,
        )
        if per_subgraph is not None:
            diagnostics.append(per_subgraph)
    if not diagnostics:
        return None
    count = float(len(diagnostics))
    density_errors = [diag["density_relative_error"] for diag in diagnostics]
    degree_errors = [diag["mean_degree_relative_error"] for diag in diagnostics]
    return {
        "density_relative_error": sum(density_errors) / count,
        "mean_degree_relative_error": sum(degree_errors) / count,
        "max_density_relative_error": max(density_errors),
        "max_mean_degree_relative_error": max(degree_errors),
        "subgraphs": count,
        "full_space_pairs": sum(diag["full_space_pairs"] for diag in diagnostics),
        "subset_pairs": sum(diag["subset_pairs"] for diag in diagnostics),
    }
```

Call it from `train_refiner` in two places, inside the same topology block that runs
`_topology_subset_backward_step`. Use the **exact** names already in scope there (review:
NameError finding): the unwrapped refiner is `topology_refiner` (from `_unwrap_refiner(...)`,
s2gae.py:806), the graph is `train_topology_graph`, the node index is
`train_topology_node_index`, the epoch chunk tuple is `chunks`, and the subset plan is
`request.train_topology_plan`. Place these calls right after `_topology_subset_backward_step`
returns (while `chunks`/`topology_refiner` are still live), before
`optimizer.step()`:

```python
                    diag_cfg = cfg.topology_training.subset
                    # §9 smoke sanity check: once, on epoch 1 (the first scaled step).
                    if (
                        isinstance(request.train_topology_plan, TopologySubsetPlan)
                        and epoch == 1
                        and request.runtime.is_main_process
                    ):
                        smoke_diag = _topology_bias_diagnostic_step(
                            refiner=topology_refiner,
                            graph=train_topology_graph,
                            plan=request.train_topology_plan,
                            chunks=chunks,
                            node_index=train_topology_node_index,
                            max_node_size=diag_cfg.bias_diagnostic_max_node_size,
                            max_subgraphs=diag_cfg.bias_diagnostic_max_subgraphs,
                        )
                        if smoke_diag is not None:
                            LOGGER.info(
                                "tccig topology smoke sanity check (§9): "
                                "density_rel_err=%.4f mean_degree_rel_err=%.4f "
                                "subgraphs=%s full_pairs=%s subset_pairs=%s",
                                smoke_diag["density_relative_error"],
                                smoke_diag["mean_degree_relative_error"],
                                int(smoke_diag["subgraphs"]),
                                int(smoke_diag["full_space_pairs"]),
                                int(smoke_diag["subset_pairs"]),
                            )
                            epoch_history["topology_bias_density_rel_err"] = smoke_diag[
                                "density_relative_error"
                            ]
                            epoch_history["topology_bias_mean_degree_rel_err"] = smoke_diag[
                                "mean_degree_relative_error"
                            ]

                    # §3.7 production diagnostic: every N epochs across the size mixture.
                    every_n = diag_cfg.bias_diagnostic_every_n_epochs
                    if (
                        isinstance(request.train_topology_plan, TopologySubsetPlan)
                        and every_n > 0
                        and epoch % every_n == 0
                        and request.runtime.is_main_process
                    ):
                        prod_diag = _topology_bias_diagnostic_step(
                            refiner=topology_refiner,
                            graph=train_topology_graph,
                            plan=request.train_topology_plan,
                            chunks=chunks,
                            node_index=train_topology_node_index,
                            max_node_size=diag_cfg.bias_diagnostic_max_node_size,
                            max_subgraphs=diag_cfg.bias_diagnostic_max_subgraphs,
                        )
                        if prod_diag is not None:
                            LOGGER.info(
                                "tccig topology bias diagnostic (§3.7) epoch=%s: "
                                "mean_density_rel_err=%.4f mean_degree_rel_err=%.4f "
                                "max_density_rel_err=%.4f subgraphs=%s",
                                epoch,
                                prod_diag["density_relative_error"],
                                prod_diag["mean_degree_relative_error"],
                                prod_diag["max_density_relative_error"],
                                int(prod_diag["subgraphs"]),
                            )
                            epoch_history["topology_bias_density_rel_err"] = prod_diag[
                                "density_relative_error"
                            ]
                            epoch_history["topology_bias_mean_degree_rel_err"] = prod_diag[
                                "mean_degree_relative_error"
                            ]
                            epoch_history["topology_bias_max_density_rel_err"] = prod_diag[
                                "max_density_relative_error"
                            ]
```

The diagnostic runs only on `request.runtime.is_main_process` (guard above) to avoid
duplicate logs. It reads only `topology_refiner`'s current weights under `no_grad`, so it
is safe to call after the backward/all-reduce but before `optimizer.step()`. Store the
relative errors in `epoch_history` so they land in the run metrics. Also keep the
lightweight per-epoch subset totals (note `epoch_samples` and `chunks` are the same locals
the subset branch built):

```python
                    if isinstance(request.train_topology_plan, TopologySubsetPlan):
                        epoch_history["train_topology_subset_pairs"] = len(epoch_samples)
                        epoch_history["train_topology_subset_subgraphs"] = len(chunks)
```

- [ ] **Step 6: Run subset tests**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_subset.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
rtk git add tccig/topology_subset.py tccig/s2gae.py tests/unit/test_tccig_topology_subset.py
rtk git commit -m "feat: add topology subset bias diagnostic (smoke + production)"
```

---

### Task 15: Full Local Verification

**Files:**
- Verify all touched files.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_subset.py tests/unit/test_tccig_topology_training.py tests/unit/test_topology_finetune.py tests/unit/test_topology_plan_cache.py tests/unit/test_tccig_topology_distributed.py -v
```

Expected: PASS.

- [ ] **Step 2: Run lint on touched files**

Run:

```bash
rtk uv run --locked --no-sync --offline ruff check tccig/topology_subset.py tccig/train.py tccig/s2gae.py src/topology/finetune_losses.py src/topology/plan_cache.py tests/unit/test_tccig_topology_subset.py tests/unit/test_tccig_topology_training.py tests/unit/test_topology_finetune.py tests/unit/test_topology_plan_cache.py tests/unit/test_tccig_topology_distributed.py
```

Expected: PASS.

- [ ] **Step 3: Format touched files**

Run:

```bash
rtk uv run --locked --no-sync --offline ruff format tccig/topology_subset.py tccig/train.py tccig/s2gae.py src/topology/finetune_losses.py src/topology/plan_cache.py tests/unit/test_tccig_topology_subset.py tests/unit/test_tccig_topology_training.py tests/unit/test_topology_finetune.py tests/unit/test_topology_plan_cache.py tests/unit/test_tccig_topology_distributed.py
```

Expected: files formatted or already clean.

- [ ] **Step 4: Check worktree diff**

Run:

```bash
rtk git diff --stat
rtk git diff --check
```

Expected: diff only includes planned files; `git diff --check` prints no errors.

- [ ] **Step 5: Commit verification cleanup**

If formatting changed files:

```bash
rtk git add tccig/topology_subset.py tccig/train.py tccig/s2gae.py src/topology/finetune_losses.py src/topology/plan_cache.py tests/unit/test_tccig_topology_subset.py tests/unit/test_tccig_topology_training.py tests/unit/test_topology_finetune.py tests/unit/test_topology_plan_cache.py tests/unit/test_tccig_topology_distributed.py configs/tccig/02_balanced_subset.yaml configs/tccig/02_balanced_subset_smoke.yaml
rtk git commit -m "chore: format topology subset rerun changes"
```

Expected: commit succeeds if formatting changed files; skip this commit when no formatting diff exists.

- [ ] **Step 6: Run the smoke config as an end-to-end gate (REQUIRED before the full rerun)**

The unit/lint checks above do not exercise the real train loop (scoring → cache →
chunked backward → diagnostic). The smoke config is the gate that does, on a tiny graph,
so a wiring break (e.g. a `NameError` in the topology block, a cache-load rejection, a
`KeyError` on `clustering_mmd`) is caught locally in minutes instead of after a multi-day
GPU rerun. Run the **whole** smoke training end to end and require a clean exit.

Submit it through the existing launch script on the HPC head node (`tccig.sh` already
takes a config path argument and prefers `accelerate launch` when GPUs are present):

```bash
rtk ssh wangar2023@10.15.89.192 \
  "cd ~/grand && sbatch scripts/tccig.sh configs/tccig/02_balanced_subset_smoke.yaml"
```

For a CPU-only login-node dry run (no GPUs; single process), the equivalent direct call is:

```bash
GRAND_TCCIG_GPUS=0 rtk uv run --locked --no-sync --offline \
  python -m tccig.train --config configs/tccig/02_balanced_subset_smoke.yaml
```

Acceptance gate (all must hold before launching the full rerun):

- The process exits 0 (no `NameError`/`KeyError`/cache-load exception in the topology block).
- The log contains `tccig topology smoke sanity check (§9):` with a finite
  `density_rel_err` (proves the diagnostic path and the subset backward both ran).
- The log shows `train_topology_scale` > 0 at epoch 1 (proves the smoke schedule engages;
  see Task 13 `ramp_epochs: 0`).
- The log contains `tccig scoring progress` or the `cache=miss`/`cache=hit` scoring line
  (proves the bounded-pool scoring + cache wiring ran).

If any check fails, fix the cause before the full rerun — do **not** submit the multi-day
job on a smoke failure.

- [ ] **Step 7: Submit the full rerun (only after Step 6 passes)**

```bash
rtk ssh wangar2023@10.15.89.192 \
  "cd ~/grand && sbatch scripts/tccig.sh configs/tccig/02_balanced_subset.yaml"
```

Expected: job accepted (an sbatch job id is printed). Monitor `logs/tccig/slurm_<jobid>.out`.

---

## Review Remediation Tasks 16-20

These tasks repair the post-implementation review findings before the Task 15 smoke gate
and full rerun. Keep the non-negotiable invariants intact:

- The frozen v3.1 scorer is not modified.
- Official validation/test topology metrics are not changed.
- Subset topology training remains opt-in through `refiner.topology_training.subset.enabled`.
- Training clustering remains off through `include_clustering_mmd=False`.
- Diagnostic-only full-space scoring is a bounded, cached exception for smoke/diagnostic
  checks only. It must never enter training graph edges, the subset pair frame, or the
  topology objective.

### Task 16: Diagnostic Full-Space Scoring and Node Coverage

**Files:**
- Modify: `tccig/topology_subset.py`
- Modify: `tccig/train.py`
- Modify: `tccig/s2gae.py`
- Modify: `tccig/prepare.py`
- Modify: `src/topology/plan_cache.py`
- Test: `tests/unit/test_tccig_topology_subset.py`
- Test: `tests/unit/test_tccig_topology_training.py`

**Design constraints:**
- Use one shared selector: `select_diagnostic_subgraphs(plan, max_node_size, max_subgraphs)`.
  Both plan-build scoring and runtime diagnostics call this function.
- Diagnostic pair provenance is deterministic:
  `select_diagnostic_subgraphs(...) -> _all_local_pairs(subgraph.nodes) -> union by pair_id -> id-sorted`.
- Score diagnostic full-space pairs with the existing frozen-scorer path only:
  `_score_split(split="train_topology_subset_diagnostic", pairs=..., scorer_cfg=..., runtime=..., cache_dir=...)`.
- Do not reuse `subset_plan_payload_metadata` verbatim for diagnostic full-space payloads.
  Add `subset_diagnostic_payload_metadata(...)` or a diagnostic sub-key that includes the
  subset key fields plus `bias_diagnostic_max_node_size`, `bias_diagnostic_max_subgraphs`,
  scorer identity, and graph/sampler seed.
- Stored diagnostic payload shape:
  `dict[str, dict[str, float]]`, where outer key is `subgraph_id` and inner key is `pair_id`.
  Main rank writes the payload; every rank attempts load-before-score.
- Add `extra_node_ids: list[str] | None = None` to `SplitBundle`.
  `_collect_node_ids(*, pairs, graph_edges, extra_node_ids=None)` is the sole node assembly
  function. `_build_graph` and `_node_index_from_split_bundle` both pass
  `bundle.extra_node_ids` into it. `_edge_index_and_weight_from_edges` and
  `_pair_index_from_pairs` are not changed.

- [ ] **Step 1: Write failing selector and diagnostic pair tests**

Append to `tests/unit/test_tccig_topology_subset.py`:

```python
def test_select_diagnostic_subgraphs_spreads_across_sizes() -> None:
    from tccig.topology_subset import (
        TopologySubgraphPlan,
        TopologySubsetPlan,
        select_diagnostic_subgraphs,
    )

    def _subgraph(size: int, index: int) -> TopologySubgraphPlan:
        return TopologySubgraphPlan(
            subgraph_id=f"size={size}:index={index}",
            node_size=size,
            nodes=tuple(f"n{size}_{index}_{j}" for j in range(size)),
            positives=(),
            candidate_negatives=(),
            hard_pool=(),
            uniform_pool=(),
        )

    plan = TopologySubsetPlan(
        subgraphs=tuple(_subgraph(size, index) for size in (4, 8) for index in range(3)),
        active_sizes=(4, 8),
        skipped_sizes={},
        total_positive_pairs=0,
        total_candidate_negatives=0,
        total_pool_negatives=0,
    )

    selected = select_diagnostic_subgraphs(plan, max_node_size=40, max_subgraphs=2)
    assert {subgraph.node_size for subgraph in selected} == {4, 8}
    assert [subgraph.subgraph_id for subgraph in selected] == [
        "size=4:index=0",
        "size=8:index=0",
    ]

    capped = select_diagnostic_subgraphs(plan, max_node_size=4, max_subgraphs=0)
    assert {subgraph.node_size for subgraph in capped} == {4}
    assert len(capped) == 3
```

Append:

```python
def test_diagnostic_full_space_scoring_pairs_are_unique_and_ordered() -> None:
    from tccig.topology_subset import (
        TopologySubgraphPlan,
        TopologySubsetPlan,
        diagnostic_full_space_scoring_pairs,
    )

    subgraph = TopologySubgraphPlan(
        subgraph_id="size=4:index=0",
        node_size=4,
        nodes=("a", "b", "c", "d"),
        positives=(),
        candidate_negatives=(),
        hard_pool=(),
        uniform_pool=(),
    )
    plan = TopologySubsetPlan(
        subgraphs=(subgraph,),
        active_sizes=(4,),
        skipped_sizes={},
        total_positive_pairs=0,
        total_candidate_negatives=0,
        total_pool_negatives=0,
    )

    rows = diagnostic_full_space_scoring_pairs(
        plan,
        max_node_size=40,
        max_subgraphs=1,
    )

    assert [row[0] for row in rows] == sorted({row[0] for row in rows})
    assert len(rows) == 6
    assert rows[0] == ("a||b", "a", "b")
```

- [ ] **Step 2: Run selector tests and verify RED**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_subset.py::test_select_diagnostic_subgraphs_spreads_across_sizes tests/unit/test_tccig_topology_subset.py::test_diagnostic_full_space_scoring_pairs_are_unique_and_ordered -v
```

Expected: FAIL with import errors for `select_diagnostic_subgraphs` and
`diagnostic_full_space_scoring_pairs`.

- [ ] **Step 3: Implement shared selector and diagnostic pair extraction**

Add to `tccig/topology_subset.py`:

```python
def select_diagnostic_subgraphs(
    plan: TopologySubsetPlan,
    *,
    max_node_size: int,
    max_subgraphs: int,
) -> tuple[TopologySubgraphPlan, ...]:
    """Pick diagnostic subgraphs across the active size mixture."""
    if max_node_size < 2:
        return ()
    by_size: dict[int, list[TopologySubgraphPlan]] = defaultdict(list)
    for subgraph in plan.subgraphs:
        if subgraph.node_size <= max_node_size:
            by_size[subgraph.node_size].append(subgraph)
    for rows in by_size.values():
        rows.sort(key=lambda item: item.subgraph_id)
    selected: list[TopologySubgraphPlan] = []
    cap = max_subgraphs if max_subgraphs > 0 else None
    cursors = {size: 0 for size in sorted(by_size)}
    while cursors and (cap is None or len(selected) < cap):
        progressed = False
        for size in sorted(cursors):
            if cap is not None and len(selected) >= cap:
                break
            index = cursors[size]
            rows = by_size[size]
            if index < len(rows):
                selected.append(rows[index])
                cursors[size] = index + 1
                progressed = True
        if not progressed:
            break
    return tuple(selected)


def diagnostic_full_space_scoring_pairs(
    plan: TopologySubsetPlan,
    *,
    max_node_size: int,
    max_subgraphs: int,
) -> tuple[tuple[str, str, str], ...]:
    """Return unique full-space diagnostic pair rows as (pair_id, protein_a, protein_b)."""
    by_id: dict[str, tuple[str, str, str]] = {}
    for subgraph in select_diagnostic_subgraphs(
        plan,
        max_node_size=max_node_size,
        max_subgraphs=max_subgraphs,
    ):
        for _index_a, _index_b, protein_a, protein_b, pair_id in _all_local_pairs(subgraph.nodes):
            by_id.setdefault(pair_id, (pair_id, protein_a, protein_b))
    return tuple(by_id[pair_id] for pair_id in sorted(by_id))
```

- [ ] **Step 4: Run selector tests and verify GREEN**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_subset.py::test_select_diagnostic_subgraphs_spreads_across_sizes tests/unit/test_tccig_topology_subset.py::test_diagnostic_full_space_scoring_pairs_are_unique_and_ordered -v
```

Expected: PASS.

- [ ] **Step 5: Write failing extra-node graph test**

Append to `tests/unit/test_tccig_topology_training.py`:

```python
def test_split_graph_extra_node_ids_get_features_but_not_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from tccig.prepare import CandidatePair, SplitBundle
    from tccig.s2gae import _build_split_graph, _node_index_from_split_bundle, _parse_config

    def _fake_features(*, protein_ids, cache_dir, index_path, input_dim, max_sequence_length, device):
        return torch.arange(len(protein_ids) * input_dim, dtype=torch.float32, device=device).reshape(
            len(protein_ids), input_dim
        )

    monkeypatch.setattr("tccig.s2gae.load_mean_pooled_node_features", _fake_features)
    cfg = _parse_config(_base_refiner_config())
    bundle = SplitBundle(
        split="train_topology",
        pairs=[CandidatePair("a", "b")],
        pairwise_probabilities=[0.8],
        pairwise_graph_edges=[],
        extra_node_ids=["c", "d"],
    )

    graph = _build_split_graph(bundle, cfg=cfg, device=torch.device("cpu"))
    node_index = _node_index_from_split_bundle(bundle)

    assert set(node_index) == {"a", "b", "c", "d"}
    assert graph.node_features.shape[0] == 4
    assert graph.pair_index.shape[1] == 1
    assert graph.edge_index.numel() == 0
```

- [ ] **Step 6: Run extra-node test and verify RED**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_training.py::test_split_graph_extra_node_ids_get_features_but_not_pairs -v
```

Expected: FAIL because `SplitBundle` has no `extra_node_ids` field.

- [ ] **Step 7: Implement `extra_node_ids` as a single node-assembly entry point**

Modify `tccig/prepare.py`:

```python
@dataclass(frozen=True)
class SplitBundle:
    """One scored PRING split with label-safe graph inputs."""

    split: str
    pairs: list[CandidatePair]
    pairwise_probabilities: list[float]
    pairwise_graph_edges: list[tuple[str, str]]
    candidate_labels: list[int] | None = None
    loss_targets: list[int] | None = None
    graph_edges: list[tuple[str, str]] | None = None
    extra_node_ids: list[str] | None = None
```

Modify `tccig/s2gae.py`:

```python
def _node_index_from_split_bundle(bundle: SplitBundle) -> dict[str, int]:
    """Map protein IDs to the node ordering used by ``_build_split_graph``."""
    node_ids = _collect_node_ids(
        pairs=bundle.pairs,
        graph_edges=bundle.pairwise_graph_edges,
        extra_node_ids=bundle.extra_node_ids,
    )
    return {protein_id: index for index, protein_id in enumerate(node_ids)}
```

Update `_build_graph(...)` and `_build_split_graph(...)` so `extra_node_ids` reaches
`_collect_node_ids`:

```python
def _build_split_graph(
    bundle: SplitBundle,
    *,
    cfg: S2GAEConfig,
    device: torch.device,
) -> _SplitGraph:
    return _build_graph(
        pairs=bundle.pairs,
        pairwise_probabilities=bundle.pairwise_probabilities,
        pairwise_graph_edges=bundle.pairwise_graph_edges,
        extra_node_ids=bundle.extra_node_ids,
        cfg=cfg,
        device=device,
    )
```

```python
def _collect_node_ids(
    *,
    pairs: Sequence[CandidatePair],
    graph_edges: Sequence[tuple[str, str]],
    extra_node_ids: Sequence[str] | None = None,
) -> list[str]:
    protein_ids: set[str] = set()
    for pair in pairs:
        protein_ids.add(pair.protein_a)
        protein_ids.add(pair.protein_b)
    for protein_a, protein_b in graph_edges:
        protein_ids.add(protein_a)
        protein_ids.add(protein_b)
    if extra_node_ids is not None:
        protein_ids.update(str(node_id) for node_id in extra_node_ids)
    if not protein_ids:
        raise ValueError("S2GAE split graph requires at least one protein")
    return sorted(protein_ids)
```

Do not change `_edge_index_and_weight_from_edges` or `_pair_index_from_pairs`.

- [ ] **Step 8: Add diagnostic metadata and payload cache helpers**

Add to `src/topology/plan_cache.py`:

```python
def subset_diagnostic_payload_metadata(
    *,
    split: str,
    graph: nx.Graph,
    node_sizes: Sequence[int],
    samples_per_size: int,
    seed: int,
    strategy: str,
    coverage_augmentation: bool,
    candidate_ratio: int,
    pool_ratio: int,
    epoch_ratio: int,
    hard_fraction: float,
    uniform_fraction: float,
    hard_stratum_fraction: float,
    max_subgraphs_per_size: int,
    max_labeled_pairs_per_size: int,
    bias_diagnostic_max_node_size: int,
    bias_diagnostic_max_subgraphs: int,
    scorer_config: Mapping[str, object],
) -> dict[str, object]:
    """Build the cache key for diagnostic-only full-space scorer payloads."""
    metadata = subset_plan_payload_metadata(
        split=split,
        graph=graph,
        node_sizes=node_sizes,
        samples_per_size=samples_per_size,
        seed=seed,
        strategy=strategy,
        coverage_augmentation=coverage_augmentation,
        candidate_ratio=candidate_ratio,
        pool_ratio=pool_ratio,
        epoch_ratio=epoch_ratio,
        hard_fraction=hard_fraction,
        uniform_fraction=uniform_fraction,
        hard_stratum_fraction=hard_stratum_fraction,
        max_subgraphs_per_size=max_subgraphs_per_size,
        max_labeled_pairs_per_size=max_labeled_pairs_per_size,
        pair_scope="subset_diagnostic",
        scorer_config=scorer_config,
    )
    metadata["bias_diagnostic_max_node_size"] = int(bias_diagnostic_max_node_size)
    metadata["bias_diagnostic_max_subgraphs"] = int(bias_diagnostic_max_subgraphs)
    return metadata
```

Add shape-specific load/write helpers:

```python
def load_subset_diagnostic_cache(
    *,
    cache_dir: Path,
    split: str,
    metadata: Mapping[str, object],
) -> dict[str, dict[str, float]] | None:
    """Load diagnostic full-space scorer probabilities, or ``None`` on miss."""
    path = _plan_path(cache_dir, split)
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("ignoring corrupt topology subset diagnostic cache at %s", path)
        return None
    if not isinstance(document, Mapping) or document.get("metadata") != dict(metadata):
        return None
    payload = document.get("payload")
    if not isinstance(payload, Mapping):
        return None
    result: dict[str, dict[str, float]] = {}
    for subgraph_id, raw_pairs in payload.items():
        if not isinstance(subgraph_id, str) or not isinstance(raw_pairs, Mapping):
            LOGGER.warning("ignoring malformed topology subset diagnostic cache at %s", path)
            return None
        result[subgraph_id] = {}
        for pair_id, probability in raw_pairs.items():
            if not isinstance(pair_id, str) or not isinstance(probability, (int, float)):
                LOGGER.warning("ignoring malformed topology subset diagnostic cache at %s", path)
                return None
            result[subgraph_id][pair_id] = float(probability)
    return result


def write_subset_diagnostic_cache(
    *,
    cache_dir: Path,
    split: str,
    metadata: Mapping[str, object],
    payload: Mapping[str, Mapping[str, float]],
) -> None:
    """Persist diagnostic full-space scorer probabilities."""
    write_json(
        _plan_path(cache_dir, split),
        {
            "metadata": dict(metadata),
            "payload": {
                str(subgraph_id): {str(pair_id): float(prob) for pair_id, prob in pairs.items()}
                for subgraph_id, pairs in payload.items()
            },
        },
    )
    write_json(_manifest_path(cache_dir, split), dict(metadata))
```

- [ ] **Step 9: Wire diagnostic scoring in `tccig/train.py`**

In the subset branch after `subset_plan` is resolved, select diagnostic subgraphs, load-or-score
their full-space pairs, and add diagnostic nodes to the returned `SplitBundle`.

Use this implementation shape:

```python
diagnostic_subgraphs = select_diagnostic_subgraphs(
    subset_plan,
    max_node_size=subset_cfg.bias_diagnostic_max_node_size,
    max_subgraphs=subset_cfg.bias_diagnostic_max_subgraphs,
)
diagnostic_node_ids = sorted({node for subgraph in diagnostic_subgraphs for node in subgraph.nodes})
diagnostic_metadata = subset_diagnostic_payload_metadata(
    split="train_topology_subset_diagnostic",
    graph=train_graph,
    node_sizes=node_sizes,
    samples_per_size=samples_per_size,
    seed=seed,
    strategy=strategy,
    coverage_augmentation=coverage_augmentation,
    candidate_ratio=subset_cfg.candidate_ratio,
    pool_ratio=subset_cfg.pool_ratio,
    epoch_ratio=subset_cfg.epoch_ratio,
    hard_fraction=subset_cfg.hard_fraction,
    uniform_fraction=subset_cfg.uniform_fraction,
    hard_stratum_fraction=subset_cfg.hard_stratum_fraction,
    max_subgraphs_per_size=subset_cfg.max_subgraphs_per_size,
    max_labeled_pairs_per_size=subset_cfg.max_labeled_pairs_per_size,
    bias_diagnostic_max_node_size=subset_cfg.bias_diagnostic_max_node_size,
    bias_diagnostic_max_subgraphs=subset_cfg.bias_diagnostic_max_subgraphs,
    scorer_config=scorer_cfg,
)
diagnostic_full_space = load_subset_diagnostic_cache(
    cache_dir=cache_dir,
    split="train_topology_subset_diagnostic",
    metadata=diagnostic_metadata,
)
if diagnostic_full_space is None:
    diagnostic_rows = diagnostic_full_space_scoring_pairs(
        subset_plan,
        max_node_size=subset_cfg.bias_diagnostic_max_node_size,
        max_subgraphs=subset_cfg.bias_diagnostic_max_subgraphs,
    )
    diagnostic_scores = _score_split(
        split="train_topology_subset_diagnostic",
        pairs=[CandidatePair(protein_a, protein_b) for _pair_id, protein_a, protein_b in diagnostic_rows],
        scorer_cfg=scorer_cfg,
        runtime=runtime,
        cache_dir=cache_dir,
    )
    score_by_pair_id = {
        pair_id: float(score)
        for (pair_id, _protein_a, _protein_b), score in zip(diagnostic_rows, diagnostic_scores, strict=True)
    }
    diagnostic_full_space = {
        subgraph.subgraph_id: {
            pair_id: score_by_pair_id[pair_id]
            for _index_a, _index_b, _protein_a, _protein_b, pair_id in _all_local_pairs(subgraph.nodes)
        }
        for subgraph in diagnostic_subgraphs
    }
    if runtime.is_main_process:
        write_subset_diagnostic_cache(
            cache_dir=cache_dir,
            split="train_topology_subset_diagnostic",
            metadata=diagnostic_metadata,
            payload=diagnostic_full_space,
        )
```

Return `diagnostic_full_space` from `_build_train_topology_bundle`, and pass it into
`TrainRefinerRequest(train_topology_diagnostic_full_space=diagnostic_full_space)`.

- [ ] **Step 10: Thread diagnostic payload through `TrainRefinerRequest`**

Modify `tccig/prepare.py`:

```python
@dataclass(frozen=True)
class TrainRefinerRequest:
    """Concrete request for S2GAE refiner training."""

    train: SplitBundle
    validation: SplitBundle
    runtime: TCCIGRuntime
    config: Mapping[str, object]
    graph_rule: GraphRule
    validation_topology: SplitBundle | None = None
    validation_topology_plan: object | None = None
    train_topology: SplitBundle | None = None
    train_topology_plan: object | None = None
    train_topology_diagnostic_full_space: Mapping[str, Mapping[str, float]] | None = None
```

Update call sites in `tccig/train.py` so the new field is supplied for subset training
and left as default `None` everywhere else.

- [ ] **Step 11: Rewrite runtime diagnostic to use diagnostic payload**

Modify `_subgraph_bias_diagnostic(...)` in `tccig/s2gae.py`:

```python
def _subgraph_bias_diagnostic(
    *,
    refiner: S2GAERefiner,
    graph: _SplitGraph,
    subgraph: TopologySubgraphPlan,
    chunk: TopologySubgraphEpochChunk,
    node_index: Mapping[str, int],
    diagnostic_full_space: Mapping[str, Mapping[str, float]],
) -> dict[str, float] | None:
    """IPW-vs-full-space bias diagnostic for one pre-scored capped subgraph."""
    scorer_probs = diagnostic_full_space.get(subgraph.subgraph_id)
    if scorer_probs is None:
        return None
    full_pairs = _all_local_pairs(tuple(subgraph.nodes))
    if any(row[4] not in scorer_probs for row in full_pairs):
        return None
    if any(row[2] not in node_index or row[3] not in node_index for row in full_pairs):
        return None
    device = graph.node_features.device
    with torch.no_grad():
        hidden_states = refiner.encode(
            node_features=graph.node_features,
            edge_index=graph.edge_index,
            edge_weight=graph.edge_weight,
        )
        global_pairs = (
            torch.tensor(
                [[node_index[row[2]], node_index[row[3]]] for row in full_pairs],
                dtype=torch.long,
                device=device,
            )
            .t()
            .contiguous()
        )
        pairwise_probabilities = torch.tensor(
            [float(scorer_probs[row[4]]) for row in full_pairs],
            dtype=torch.float32,
            device=device,
        )
        refined_logits, _ = refiner.decode(
            hidden_states=hidden_states,
            pair_index=global_pairs,
            pairwise_probabilities=pairwise_probabilities,
        )
        full_probabilities = torch.sigmoid(refined_logits).detach().cpu().tolist()
    full_space_probabilities = {
        row[4]: float(prob) for row, prob in zip(full_pairs, full_probabilities, strict=True)
    }
    subset_samples = [
        (sample.pair_id, full_space_probabilities[sample.pair_id], 1.0 / sample.pi_total)
        for sample in chunk.samples
        if sample.pair_id in full_space_probabilities
    ]
    if not subset_samples:
        return None
    return compute_subset_bias_diagnostic(
        node_size=subgraph.node_size,
        full_space_probabilities=full_space_probabilities,
        subset_samples=subset_samples,
    )
```

Update `_topology_bias_diagnostic_step(...)` so it filters selected subgraphs by
`chunk_by_id` and `diagnostic_full_space` keys, then passes the payload into
`_subgraph_bias_diagnostic(...)`.

- [ ] **Step 12: Run focused Task 16 tests**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_subset.py::test_select_diagnostic_subgraphs_spreads_across_sizes tests/unit/test_tccig_topology_subset.py::test_diagnostic_full_space_scoring_pairs_are_unique_and_ordered tests/unit/test_tccig_topology_training.py::test_split_graph_extra_node_ids_get_features_but_not_pairs -v
```

Expected: PASS.

- [ ] **Step 13: Commit Task 16**

```bash
rtk git add tccig/topology_subset.py tccig/train.py tccig/s2gae.py tccig/prepare.py src/topology/plan_cache.py tests/unit/test_tccig_topology_subset.py tests/unit/test_tccig_topology_training.py
rtk git commit -m "fix: score diagnostic topology full-space pairs"
```

### Task 17: Deep Subset Cache Validation

**Files:**
- Modify: `src/topology/plan_cache.py`
- Test: `tests/unit/test_topology_plan_cache.py`

**Validation contract:**
- Validate every serialized `TopologyPairSample` with `TopologyPairSample.validate()`.
- Validate `node_size == len(nodes)`, nodes are unique strings, and sample endpoints appear
  in that subgraph's `nodes`.
- Validate `local_index_a` and `local_index_b` are inside `[0, node_size)`.
- Recompute and compare `total_positive_pairs`, `total_candidate_negatives`, and
  `total_pool_negatives`.
- Pool membership is by identity tuple only:
  `(pair_id, subgraph_id, local_index_a, local_index_b, protein_a, protein_b)`.
  Every hard/uniform pool identity must exist in `candidate_negatives`. Do not require full
  `TopologyPairSample` equality because stratum and `pi_*` fields intentionally differ.
- On any violation, `load_subset_plan_cache(...)` returns `None` and logs a warning. It
  must not raise.

- [ ] **Step 1: Write failing malformed-cache tests**

Append to `tests/unit/test_topology_plan_cache.py`:

```python
@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload["subgraphs"][0]["candidate_negatives"][0].update({"pi_total": 0.99}),
        lambda payload: payload["subgraphs"][0]["positives"][0].update({"target": 0.0}),
        lambda payload: payload["subgraphs"][0]["candidate_negatives"][0].update({"local_index_a": 99}),
        lambda payload: payload["subgraphs"][0]["candidate_negatives"][0].update({"protein_a": "missing"}),
        lambda payload: payload.update({"total_candidate_negatives": 999}),
    ],
)
def test_subset_cache_rejects_malformed_sample_payloads(tmp_path: Path, mutator) -> None:
    import networkx as nx

    from src.topology.plan_cache import load_subset_plan_cache, write_subset_plan_cache
    from tccig.topology_subset import (
        TopologySubsetSamplerConfig,
        build_topology_subset_plan,
        subset_plan_to_payload,
    )

    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "c", "d"])
    graph.add_edge("a", "b")
    cfg = TopologySubsetSamplerConfig(candidate_ratio=3, pool_ratio=2, epoch_ratio=1, seed=0)
    plan = build_topology_subset_plan(
        graph=graph,
        sampled_subgraphs={4: [("a", "b", "c", "d")]},
        config=cfg,
        scorer_probabilities={"a||c": 0.1, "a||d": 0.2, "b||c": 0.3, "b||d": 0.4, "c||d": 0.5},
    )
    payload = subset_plan_to_payload(plan)
    mutator(payload)
    metadata = {"kind": "subset", "case": "malformed"}
    write_subset_plan_cache(
        cache_dir=tmp_path,
        split="train_topology_subset",
        metadata=metadata,
        payload=payload,
    )

    assert load_subset_plan_cache(
        cache_dir=tmp_path,
        split="train_topology_subset",
        metadata=metadata,
    ) is None
```

Append:

```python
def test_subset_cache_rejects_pool_members_not_in_candidate_frame(tmp_path: Path) -> None:
    import copy
    import networkx as nx

    from src.topology.plan_cache import load_subset_plan_cache, write_subset_plan_cache
    from tccig.topology_subset import (
        TopologySubsetSamplerConfig,
        build_topology_subset_plan,
        subset_plan_to_payload,
    )

    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "c", "d"])
    graph.add_edge("a", "b")
    cfg = TopologySubsetSamplerConfig(candidate_ratio=3, pool_ratio=2, epoch_ratio=1, seed=0)
    plan = build_topology_subset_plan(
        graph=graph,
        sampled_subgraphs={4: [("a", "b", "c", "d")]},
        config=cfg,
        scorer_probabilities={"a||c": 0.1, "a||d": 0.2, "b||c": 0.3, "b||d": 0.4, "c||d": 0.5},
    )
    payload = subset_plan_to_payload(plan)
    alien = copy.deepcopy(payload["subgraphs"][0]["candidate_negatives"][0])
    alien.update({"pair_id": "x||y", "protein_a": "x", "protein_b": "y"})
    payload["subgraphs"][0]["uniform_pool"] = [alien]
    metadata = {"kind": "subset", "case": "alien_pool"}
    write_subset_plan_cache(
        cache_dir=tmp_path,
        split="train_topology_subset",
        metadata=metadata,
        payload=payload,
    )

    assert load_subset_plan_cache(
        cache_dir=tmp_path,
        split="train_topology_subset",
        metadata=metadata,
    ) is None
```

- [ ] **Step 2: Run cache validation tests and verify RED**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_topology_plan_cache.py::test_subset_cache_rejects_malformed_sample_payloads tests/unit/test_topology_plan_cache.py::test_subset_cache_rejects_pool_members_not_in_candidate_frame -v
```

Expected: FAIL because current subset cache validation accepts malformed payloads.

- [ ] **Step 3: Implement strict subset payload validation**

Add helper functions in `src/topology/plan_cache.py`:

```python
def _subset_sample_identity(sample: object) -> tuple[str, str, int, int, str, str] | None:
    if not isinstance(sample, Mapping):
        return None
    try:
        return (
            str(sample["pair_id"]),
            str(sample["subgraph_id"]),
            int(sample["local_index_a"]),  # type: ignore[arg-type]
            int(sample["local_index_b"]),  # type: ignore[arg-type]
            str(sample["protein_a"]),
            str(sample["protein_b"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _validate_subset_sample_list(
    *,
    raw_samples: object,
    nodes: Sequence[str],
    node_size: int,
) -> tuple[bool, list[tuple[str, str, int, int, str, str]]]:
    from tccig.topology_subset import TopologyPairSample, _sample_from_dict

    if not isinstance(raw_samples, list):
        return False, []
    identities: list[tuple[str, str, int, int, str, str]] = []
    node_set = set(nodes)
    for raw_sample in raw_samples:
        if not isinstance(raw_sample, Mapping):
            return False, []
        identity = _subset_sample_identity(raw_sample)
        if identity is None:
            return False, []
        sample = _sample_from_dict(raw_sample)
        try:
            sample.validate()
        except ValueError:
            return False, []
        if not 0 <= sample.local_index_a < node_size:
            return False, []
        if not 0 <= sample.local_index_b < node_size:
            return False, []
        if sample.protein_a not in node_set or sample.protein_b not in node_set:
            return False, []
        if sample.local_index_a == sample.local_index_b:
            return False, []
        identities.append(identity)
    return True, identities
```

Then expand `_subset_payload_is_rehydratable(...)` so it calls the helpers, recomputes
totals, and checks pool membership:

```python
candidate_identities = set(candidate_ids)
for pool_identity in hard_ids + uniform_ids:
    if pool_identity not in candidate_identities:
        return False
```

Return `False` on every validation miss; keep `load_subset_plan_cache(...)` responsible for
logging the warning and returning `None`.

- [ ] **Step 4: Run cache validation tests and verify GREEN**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_topology_plan_cache.py::test_subset_cache_rejects_malformed_sample_payloads tests/unit/test_topology_plan_cache.py::test_subset_cache_rejects_pool_members_not_in_candidate_frame -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 17**

```bash
rtk git add src/topology/plan_cache.py tests/unit/test_topology_plan_cache.py
rtk git commit -m "fix: validate topology subset cache payloads"
```

### Task 18: Serialize Subset Config in Artifacts

**Files:**
- Modify: `tccig/s2gae.py`
- Test: `tests/unit/test_tccig_topology_training.py`

- [ ] **Step 1: Write failing config serialization test**

Append to `tests/unit/test_tccig_topology_training.py`:

```python
def test_config_to_json_serializes_topology_subset_fields() -> None:
    import yaml

    from tccig.s2gae import _config_to_json, _parse_config

    raw = yaml.safe_load(
        Path("configs/tccig/02_balanced_subset_smoke.yaml").read_text(encoding="utf-8")
    )
    cfg = _parse_config(raw["refiner"])

    subset = _config_to_json(cfg)["topology_training"]["subset"]

    assert subset == {
        "enabled": True,
        "candidate_ratio": 4,
        "pool_ratio": 2,
        "epoch_ratio": 2,
        "hard_fraction": 0.5,
        "uniform_fraction": 0.5,
        "hard_stratum_fraction": 0.5,
        "seed": 0,
        "max_subgraphs_per_size": 0,
        "max_labeled_pairs_per_size": 0,
        "bias_diagnostic_every_n_epochs": 1,
        "bias_diagnostic_max_node_size": 40,
        "bias_diagnostic_max_subgraphs": 4,
    }
```

- [ ] **Step 2: Run config serialization test and verify RED**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_training.py::test_config_to_json_serializes_topology_subset_fields -v
```

Expected: FAIL because `_config_to_json(...)` omits `topology_training["subset"]`.

- [ ] **Step 3: Serialize all subset config fields**

Modify the `topology_training` block in `_config_to_json(...)` in `tccig/s2gae.py`:

```python
"subset": {
    "enabled": cfg.topology_training.subset.enabled,
    "candidate_ratio": cfg.topology_training.subset.candidate_ratio,
    "pool_ratio": cfg.topology_training.subset.pool_ratio,
    "epoch_ratio": cfg.topology_training.subset.epoch_ratio,
    "hard_fraction": cfg.topology_training.subset.hard_fraction,
    "uniform_fraction": cfg.topology_training.subset.uniform_fraction,
    "hard_stratum_fraction": cfg.topology_training.subset.hard_stratum_fraction,
    "seed": cfg.topology_training.subset.seed,
    "max_subgraphs_per_size": cfg.topology_training.subset.max_subgraphs_per_size,
    "max_labeled_pairs_per_size": cfg.topology_training.subset.max_labeled_pairs_per_size,
    "bias_diagnostic_every_n_epochs": cfg.topology_training.subset.bias_diagnostic_every_n_epochs,
    "bias_diagnostic_max_node_size": cfg.topology_training.subset.bias_diagnostic_max_node_size,
    "bias_diagnostic_max_subgraphs": cfg.topology_training.subset.bias_diagnostic_max_subgraphs,
},
```

- [ ] **Step 4: Run config serialization test and verify GREEN**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_training.py::test_config_to_json_serializes_topology_subset_fields -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 18**

```bash
rtk git add tccig/s2gae.py tests/unit/test_tccig_topology_training.py
rtk git commit -m "fix: serialize topology subset config"
```

### Task 19: Monte Carlo Three-Stage Inclusion Frequency Test

**Files:**
- Test: `tests/unit/test_tccig_topology_subset.py`
- Modify only if test exposes a sampler bug: `tccig/topology_subset.py`

The test must rebuild candidate and pool stages across many plan seeds, then sample epochs
within each plan. Iterating epochs on one fixed plan only tests `pi_epoch_given_pool`.

- [ ] **Step 1: Write Monte Carlo inclusion-frequency test**

Append to `tests/unit/test_tccig_topology_subset.py`:

```python
def test_three_stage_negative_inclusion_frequency_matches_pi_total() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "c", "d", "e"])
    graph.add_edges_from([("a", "b"), ("b", "c")])
    nodes = ("a", "b", "c", "d", "e")
    cfg_template = dict(
        candidate_ratio=2,
        pool_ratio=1,
        epoch_ratio=1,
        hard_fraction=0.5,
        uniform_fraction=0.5,
        hard_stratum_fraction=0.5,
    )
    exposures: dict[str, float] = defaultdict(float)
    draws: dict[str, int] = defaultdict(int)
    trials = 0
    for plan_seed in range(300):
        cfg = TopologySubsetSamplerConfig(seed=plan_seed, **cfg_template)
        plan = build_topology_subset_plan(
            graph=graph,
            sampled_subgraphs={5: [nodes]},
            config=cfg,
            scorer_probabilities={
                "a||c": 0.9,
                "a||d": 0.2,
                "a||e": 0.1,
                "b||d": 0.8,
                "b||e": 0.3,
                "c||d": 0.7,
                "c||e": 0.4,
                "d||e": 0.6,
            },
        )
        for epoch in range(20):
            trials += 1
            sampled = sample_epoch_topology_subset(plan=plan, epoch=epoch, config=cfg)
            for subgraph in plan.subgraphs:
                for candidate in subgraph.candidate_negatives:
                    exposures[candidate.pair_id] += candidate.pi_total
            for sample in sampled:
                if sample.target == 0.0:
                    draws[sample.pair_id] += 1
                else:
                    assert sample.pi_total == 1.0
    observed_total = sum(draws.values()) / float(trials)
    expected_total = sum(exposures.values()) / float(trials)
    assert observed_total == pytest.approx(expected_total, rel=0.02)
```

- [ ] **Step 2: Run Monte Carlo test**

Run:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_subset.py::test_three_stage_negative_inclusion_frequency_matches_pi_total -v
```

Expected: PASS if sampler probabilities are correct. If it fails, inspect whether
`pi_cand`, `pi_pool_given_cand`, or `pi_epoch_given_pool` is wrong and fix the root cause in
`tccig/topology_subset.py`.

- [ ] **Step 3: Commit Task 19**

```bash
rtk git add tests/unit/test_tccig_topology_subset.py tccig/topology_subset.py
rtk git commit -m "test: verify topology subset inclusion frequencies"
```

### Task 20: Bounded Diagnostic Exception Wording

**Files:**
- Modify: `docs/superpowers/specs/2026-06-28-tccig-exp02-topology-design.md`
- Modify: `docs/superpowers/plans/2026-06-29-tccig-exp02-topology-rerun.md`

- [ ] **Step 1: Edit spec wording for training scoring vs diagnostic scoring**

In `docs/superpowers/specs/2026-06-28-tccig-exp02-topology-design.md`, update §3.3 and
§7 so they state this exact contract:

```markdown
Only candidate-frame pairs are scored for the training/pool objective, so training scoring
is bounded. The section 9 smoke sanity check and section 3.7 production diagnostic are the
only exception: they may score a few capped full-space diagnostic subgraphs through the same
frozen scorer, cache those probabilities separately, and use them only for diagnostic logging.
Diagnostic-only full-space scoring never enters training graph edges, the subset pair
frame, or the topology objective.
```

Replace the §7 batch-embedding-load bullet with:

```markdown
- Batch embedding loads are intentionally out of scope for this rerun. The bounded
  candidate frame reduces scoring enough that `_collate_pair_score_batch`'s existing
  per-endpoint `load_cached_embedding(...)` path is acceptable; any embedding-cache rewrite
  should be a separate task with cache-correctness tests.
```

- [ ] **Step 2: Update plan Scope Check**

In this plan's Scope Check, add:

```markdown
Diagnostic-only full-space scoring is a bounded exception to the "only candidate-frame
pairs are scored" rule. It exists only for section 9/3.7 diagnostics, is cached separately,
and does not affect training graph edges, the subset pair frame, or the topology objective.
```

- [ ] **Step 3: Run docs diff check**

Run:

```bash
rtk git diff -- docs/superpowers/specs/2026-06-28-tccig-exp02-topology-design.md docs/superpowers/plans/2026-06-29-tccig-exp02-topology-rerun.md
```

Expected: docs only clarify the bounded diagnostic exception and batch-load scope.

- [ ] **Step 4: Commit Task 20**

```bash
rtk git add docs/superpowers/specs/2026-06-28-tccig-exp02-topology-design.md docs/superpowers/plans/2026-06-29-tccig-exp02-topology-rerun.md
rtk git commit -m "docs: clarify topology diagnostic scoring scope"
```

### Two-Phase Execution Plan

**Phase 1 - Blockers before smoke gate**
- Task 16: diagnostic-only full-space scoring + node coverage.
- Task 17: deep subset-cache validation.
- Phase 1 verification:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_subset.py tests/unit/test_tccig_topology_training.py tests/unit/test_topology_plan_cache.py -v
rtk uv run --locked --no-sync --offline ruff check tccig/topology_subset.py tccig/train.py tccig/s2gae.py tccig/prepare.py src/topology/plan_cache.py tests/unit/test_tccig_topology_subset.py tests/unit/test_tccig_topology_training.py tests/unit/test_topology_plan_cache.py
```

Expected: PASS.

**Phase 2 - Reproducibility, sampler audit, docs, final gate**
- Task 18: artifact config serialization.
- Task 19: Monte Carlo three-stage sampler test.
- Task 20: bounded diagnostic exception docs.
- Task 15 Step 6 smoke gate, only after Phase 1 and Phase 2 local checks pass.
- Phase 2 verification:

```bash
rtk uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_subset.py tests/unit/test_tccig_topology_training.py tests/unit/test_topology_finetune.py tests/unit/test_topology_plan_cache.py tests/unit/test_tccig_topology_distributed.py -v
rtk uv run --locked --no-sync --offline ruff check tccig/topology_subset.py tccig/train.py tccig/s2gae.py tccig/prepare.py src/topology/finetune_losses.py src/topology/plan_cache.py tests/unit/test_tccig_topology_subset.py tests/unit/test_tccig_topology_training.py tests/unit/test_topology_finetune.py tests/unit/test_topology_plan_cache.py tests/unit/test_tccig_topology_distributed.py
```

Expected: PASS.

Smoke gate command, after local verification:

```bash
rtk ssh wangar2023@10.15.89.192 \
  "cd ~/grand && sbatch scripts/tccig.sh configs/tccig/02_balanced_subset_smoke.yaml"
```

Expected: job accepted; smoke log exits 0 and contains the §9 sanity-check line with finite
`density_rel_err`.

---

## Execution Notes

- The implementation keeps the full-plan path available when `refiner.topology_training.subset.enabled` is absent or false. This reduces blast radius and lets existing tests continue to validate the old behavior.
- The first smoke run should use `configs/tccig/02_balanced_subset_smoke.yaml`; the full rerun should use `configs/tccig/02_balanced_subset.yaml`.
- The smoke run must reach epoch 1 topology scale > 0. If it does not, the schedule parser or config is wrong.
- The full rerun (Task 15 Step 7) must be submitted **only after** the focused tests, lint, and the Task 15 Step 6 smoke gate all pass. The smoke gate actually executes the subset training path end-to-end on tiny data, so it catches wiring/runtime failures that unit tests and config-parse checks cannot.

## Self-Review

- Spec coverage:
  - Hybrid subset objective: Tasks 1, 2, 3, 8, 10.
  - Three-stage inclusion probability: Tasks 1, 3.
  - Clustering off (training only; validation metrics unchanged): Tasks 8, 13.
  - Balanced/budgeted sampler and bounded scoring: Tasks 1, 3, 6, 13.
  - Per-size aggregation: Tasks 9, 10.
  - DDP fork (b) sharded per-chunk backward + manual grad all-reduce: Tasks 9, 11.
  - Setup progress logging: Task 12.
  - Warmup schedule and smoke config: Task 13.
  - Bias diagnostic (§3.7 production + §9 smoke sanity check): Task 14.
- Placeholder scan:
  - This plan avoids deferred implementation markers and supplies concrete test code, function names, config keys, commands, and expected outcomes.
- Type consistency:
  - `TopologySubsetSamplerConfig`, `TopologyPairSample`, `TopologySubsetPlan`, `TopologySubgraphEpochChunk`, and `SamplingStratum` are introduced before use.
  - `pair_weights` is introduced in `compute_topology_losses` before `topology_subset_chunk_loss` consumes it.
  - `subset` is parsed into `S2GAETopologyTrainingConfig` before `train_refiner` reads `cfg.topology_training.subset`.

### Review findings addressed (plan-vs-spec correctness pass)

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | CRITICAL | Distributed backward double-counted gradients by `world_size` (every rank ran every chunk, then SUM all-reduced) and normalized per-size locally. | Task 9 now shards chunks by global index (`chunks[rank::world_size]`), computes **global** per-size scales from the full (identical-on-every-rank) chunk list, and SUM-all-reduces once over a uniform parameter set. Correctness note + `_shard_chunks_for_rank` + disjoint-partition test added. |
| 2 | CRITICAL | Task 6 subset branch referenced `sampled`/`coverage_stats`, which live only inside the `_build()` closure → `NameError`. | Subset branch now builds its own `sampled` (+ optional coverage augmentation) and `coverage_stats`, and returns early. |
| 3 | CRITICAL | Smoke config `warmup_epochs:0, ramp_epochs:1` yields scale `0.0` at epoch 1 (`topology_loss_scale(epoch=0)` → progress `0/1`). | Smoke config switched to `ramp_epochs:0` (immediate engagement). Added `test_smoke_config_engages_topology_in_epoch_one`. |
| 4 | HIGH | §3.7 production bias diagnostic not implemented. | Task 14 implements `compute_subset_bias_diagnostic` + `_topology_bias_diagnostic_step`, gated by `bias_diagnostic_every_n_epochs` on capped subgraphs. |
| 5 | HIGH | §9 smoke sanity check (subset-estimated vs exact full-space) not implemented. | Task 14 wires the same diagnostic at epoch 1 and logs per-metric relative error. |
| 6 | HIGH | No real 2-rank gradient-equivalence test (spec §12). | Task 11 adds a spawned 2-process gloo test asserting sharded backward + SUM all-reduce equals the single-process full reference. |
| 7 | HIGH | §4 per-size budget caps / coverage redistribution missing. | Added `max_subgraphs_per_size` / `max_labeled_pairs_per_size` to the sampler config (Task 1), enforced in `build_topology_subset_plan` (Task 3), parsed (Task 5), set in the main config (Task 13). |
| 8 | MEDIUM | Weak IPW test (only asserted "differs"). | Task 2 adds Horvitz-Thompson unbiasedness recovery tests (density numerator + soft degrees). |
| 9 | MEDIUM | `subset_plan_payload_metadata` was dead (no payload to key). | Task 4 adds subset-plan serialization; Task 6 wraps build+score in a load-before-score cache keyed by this metadata. |
| 10 | MEDIUM | Skipped-size logging missing. | `skipped_sizes` now logged in Task 6's scoring-estimate line; `active_node_sizes` records reasons. |
| 11 | MEDIUM | Configs set `topology_validation.compute_clustering_mmd: false`, which also disables clustering in the **validation/test metrics** (shared knob, `s2gae.py:823/1340/1350`) — a spec §2 non-goal. | Training clustering-off is enforced independently by the hardcoded `include_clustering_mmd=False` in the subset chunk loss (Task 8). Configs/tests now keep `compute_clustering_mmd: true`; Implementation Rules updated. |

### Review findings addressed (second pass: distributed backend + cache + coverage realism)

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 12 | CRITICAL | Round-1 wording implied the topology step manages its own DDP (manual `loss.backward()`, helper named `_manual_all_reduce_gradients`), risking replacement of the Accelerate-managed launcher/wrapping/AMP/optimizer. | Task 9 routes every per-chunk backward through `runtime.accelerator.backward(...)` so Accelerate keeps the launcher, model wrapping, grad scaler, and optimizer step. The **only** custom additions are `accelerator.unwrap_model(...)` (via `_unwrap_refiner`) and one `torch.distributed.all_reduce` over the sharded topology gradients (helper renamed `_all_reduce_topology_gradients`). Task 10 documents this division of labor; Task 11's gloo 2-rank test is explicitly labeled a unit-test harness for the shard+reduce algebra, **not** the production backend. |
| 13 | CRITICAL | Subset payload could not load: Task 6 called `load_plan_cache`, but `plan_cache.py:248` validates the full-plan shape via `_payload_is_rehydratable` → new subset payload rejected. `subset_plan_payload_metadata` also lacked scorer/checkpoint/embedding hashes → stale scored-pair reuse. | Task 4 splits cache loading by plan kind: adds `_subset_payload_is_rehydratable`, `load_subset_plan_cache`, `write_subset_plan_cache` (payload stamped `payload_kind="topology_subset"` + `subset_payload_version`). `subset_plan_payload_metadata` now embeds `_scorer_identity(scorer_config)` (model/checkpoint/embedding-index SHAs + `max_sequence_length`) and the per-size budgets, so a scorer or budget change invalidates the cache. Task 6 wires `load_subset_plan_cache`/`write_subset_plan_cache`. |
| 14 | HIGH | Chunked backward hoisted one `refiner.encode(...)` before the per-chunk `backward()` loop → second backward fails or needs `retain_graph=True`, defeating the memory bound. | Task 9 recomputes `refiner.encode(...)` **inside** each chunk's loop iteration (explicit "do NOT hoist" comment), so each chunk's forward graph is freed by its own backward; peak memory is one chunk's graph. |
| 15 | HIGH | Coverage redistribution not real: Task 6 computed coverage before Task 3's `rows[:N]` cap could truncate the appended coverage subgraphs → logs claimed full coverage while the built plan lost it. No budget-aware redistribution across sizes. | Removed the silent `rows[:N]` truncation from `build_topology_subset_plan` (Task 3 now trusts pre-budgeted input). Added `apply_per_size_subgraph_budget` (Task 6): caps base subgraphs per size, then distributes coverage subgraphs across eligible sizes under remaining budget (smallest eligible first, not the `max(node_sizes)` dump), and recomputes `coverage_stats` from the **post-budget** plan so logged coverage matches reality. Two unit tests (redistribution + unbounded passthrough). |
| 16 | HIGH | `_topology_bias_diagnostic_step(...)` shown as ellipsis call — no concrete helper or test expectation. | Task 14 supplies the full helper body: picks the smallest active subgraph with `node_size <= max_node_size` that produced epoch samples, decodes the full `n·(n-1)/2` pair set under `torch.no_grad()`, reads the IPW subset view off the epoch chunk, and defers the math to `compute_subset_bias_diagnostic`. Tests assert `density_relative_error == 0` for the full-space view and `> 0.1` for a biased draw. |
| 17 | MEDIUM | Scoring progress callback used `processed in milestones`; since `processed += batch_size` can jump past an exact milestone, events were silently skipped (esp. under DDP shard sizes). | Task 12 replaces the membership check with an advancing pointer that fires on `processed >= next_milestone` and skips any milestones the batch overshot. Counts are documented as rank-local. Added `test_score_progress_callback_fires_on_overshoot`. |
| 18 | MEDIUM | "Batch embedding loads" promised in spec §7 but not planned. | Scope Check now states it is **out of scope** for this rerun: bounded scoring cuts the candidate frame from ~12.79M to thousands of pairs, so the existing per-endpoint `load_cached_embedding` path in `_collate_pair_score_batch` is no longer the bottleneck. The optimization is recorded as deferred, not silently dropped. |

### Review findings addressed (third pass: wiring names, component keys, diagnostic breadth, gate)

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 19 | HIGH | Task 14 diagnostic call sites used names not in scope (`model`, `train_graph`, `epoch_chunks`, `node_index`) → `NameError`; `_topology_bias_diagnostic_step` also called `_all_local_pairs` without importing it into `s2gae.py`. | Call sites now use the exact locals live in the topology block (`s2gae.py:806`): `topology_refiner` (already unwrapped), `train_topology_graph`, `train_topology_node_index`, and the epoch `chunks` tuple. Task 14's import block adds `_all_local_pairs` (plus `TopologySubgraphPlan`, `compute_subset_bias_diagnostic`, `defaultdict`) to `s2gae.py`; `_pair_lookup` was already module-level there. |
| 20 | HIGH | `_topology_subset_backward_step` returned only `total`/`graph_sim`/`relative_density`/`degree_mmd`, but shared epoch logging reads `topology_components["clustering_mmd"]` (`s2gae.py:920`) → `KeyError`. The component sums were also rank-local after sharding, so logged `train_topo_*` values reported only this rank's shard. | Both the empty-chunk early return and the accumulator now include `"clustering_mmd": 0.0` (constant — training clustering is off, Task 8), matching the full-plan keys. After the gradient all-reduce, `_all_reduce_component_sums` SUM-reduces the detached component scalars so the logged totals reflect the full objective, not one shard. |
| 21 | MEDIUM | Production diagnostic picked only the single smallest eligible subgraph, so bias from larger sizes / the real size mixture was never exposed (spec §3.7 asks for a few capped subgraphs across the mixture). | Added `bias_diagnostic_max_subgraphs` (Task 1/5, default 4). `_select_diagnostic_subgraphs` round-robins one eligible subgraph per active size (smallest-first within size) up to the budget, so a tight budget still samples the mixture. `_topology_bias_diagnostic_step` aggregates per-subgraph diagnostics and reports mean **and max** relative error (one biased size cannot hide). Added `test_select_diagnostic_subgraphs_spreads_across_size_mixture`. |
| 22 | MEDIUM | Task 6's `tccig/train.py` import block omitted `apply_per_size_subgraph_budget`, which the subset branch calls. | Import block now includes `apply_per_size_subgraph_budget` (`sample_topology_evaluation_subgraphs` was already imported in `train.py:33`). |
| 23 | LOW | Smoke config/tests existed but final verification ran only local unit/lint checks — the smoke gate was configured, never executed as a gate before the multi-day rerun. | Task 15 adds Step 6: run the smoke config end to end (`sbatch scripts/tccig.sh configs/tccig/02_balanced_subset_smoke.yaml`, or a CPU `GRAND_TCCIG_GPUS=0` dry run) with a concrete acceptance gate (exit 0; `§9` sanity-check line with finite `density_rel_err`; `train_topology_scale > 0` at epoch 1; scoring/cache line). Step 7 submits the full rerun only after Step 6 passes; Execution Notes updated. |
