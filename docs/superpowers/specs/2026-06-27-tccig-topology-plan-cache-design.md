# TCCIG Topology Plan Cache — Design

**Date:** 2026-06-27
**Status:** Approved (design); pending implementation plan
**Scope:** TCCIG refiner path (`tccig/train.py`) topology bucket preparation

## Problem

The TCCIG refiner pipeline rebuilds its topology "graph buckets" from scratch on
every run. The per-run chain in `_build_train_topology_bundle`
(`tccig/train.py:457`) and `_build_validation_topology_bundle`
(`tccig/train.py:537`) is:

1. sample subgraphs (`sample_topology_evaluation_subgraphs`)
2. coverage augmentation (`augment_plan_for_positive_edge_coverage`, train only)
3. `build_internal_validation_plan` — O(n²) pair materialization + per-subgraph
   `graph.subgraph(nodes).copy()`

None of this is cached (only downstream pairwise scores are, via
`_load_score_cache`/`_write_score_cache`). The dominant cost is coverage
augmentation: `augment_plan_for_positive_edge_coverage` calls
`_covered_positive_edges(sampled=augmented, graph=graph)` inside its loop for
every uncovered edge, re-scanning all already-added buckets each time —
complexity grows as coverage buckets accumulate. Under DDP, all ranks repeat
this CPU-heavy work, wasting ~4× on a 4-rank job.

## Goal

Cache a **serializable topology plan payload** keyed by graph-edge hash +
sampling params. On a cache hit, both topology bundle builders rehydrate an
`InternalValidationPlan` without sampling and without coverage augmentation. We
do **not** pickle / `torch.save` the live `InternalValidationPlan` (it contains
`nx.Graph` objects); we store plain JSON data and reconstruct.

Three independent changes, each valuable on its own:

1. **Plan cache** — new module `src/topology/plan_cache.py`, wired into the two
   TCCIG topology bundle builders.
2. **Incremental coverage augmentation** — rewrite
   `augment_plan_for_positive_edge_coverage` to maintain `covered` incrementally
   instead of re-scanning all buckets per edge. Fixes the first-run hotspot even
   on a cache miss.
3. **DDP coordination** — main rank builds & writes the cache; other ranks
   barrier then read, mirroring `_score_split`. Removes redundant cross-rank
   work.

## Non-goals

- Refactoring `s2gae.py::_topology_plan_loss` to consume cached target labels
  directly (the "eliminate pair_records" option). Higher risk, touches the
  training loss path; documented as a follow-up, not in this spec.
- Caching the topology fine-tune stage (`src/pipeline/stages/topology_finetune.py`).
- Storing `pair_records` in the cache (O(n²) → multi-hundred-MB JSON).

## Section 1 — Architecture

New module `src/topology/plan_cache.py` owns: cache-key hashing, JSON
load/write, and payload ⇄ plan (de)serialization. Kept out of `tccig/train.py`
(already large) so the hash/serializer logic is unit-testable in isolation.

`tccig/train.py` gains a shared helper `_load_or_build_topology_plan(...)` used
by `_build_train_topology_bundle` (required) and optionally
`_build_validation_topology_bundle` (low priority — no coverage augmentation,
scores already cached).

Storage layout parallels the existing score cache:
- payload: `cache_dir/plans/{split}.json`
- manifest (cache key): `cache_dir/manifests/{split}_plan.json`

## Section 2 — Serializable payload & module API

`src/topology/plan_cache.py`:

- `plan_payload_metadata(*, split, graph, node_sizes, samples_per_size, seed,
  strategy, coverage_augmentation) -> dict` — the cache key. Edge hash = stable
  digest over `sorted(canonical_edge(u, v) for u, v in graph.edges())` plus the
  node-id set (node-only changes invalidate too), combined with all sampling
  params. **`strategy` is normalized (same normalization the sampler applies)
  before hashing** so `mixed` and `MIXED` map to one key.
- `plan_to_payload(plan) -> dict` — compact, JSON-safe. Per bucket: `node_size`,
  `sampled_subgraphs` (list of node tuples), and `target_edges` (per subgraph,
  the induced edge list). Each subgraph's `target_edges` is stored as
  `sorted(canonical_edge(a, b) ...)` → list of `[a, b]` for deterministic JSON
  and stable hashing/debugging. **Does not store `pair_records`.**
- `payload_to_plan(payload, *, graph) -> InternalValidationPlan` — rehydrates.
  Rebuilds each `target_subgraph` as a fresh `nx.Graph` from its node set +
  cached `target_edges`, and regenerates `pair_records` by the deterministic
  upper-triangle enumeration over each node tuple (pure Python, no graph scans,
  no `subgraph().copy()`, no coverage augmentation).
- `load_plan_cache(*, cache_dir, split, metadata) -> dict | None` /
  `write_plan_cache(*, cache_dir, split, metadata, payload) -> None` — JSON
  read/write, rejecting on metadata mismatch. Parallels
  `_load_score_cache`/`_write_score_cache`.

### pair_records decision (Option 1)

`pair_records` are every upper-triangle pair — O(n²) per subgraph. We
**re-materialize on rehydrate** rather than storing them: keeps the JSON compact
(O(edges)), and load is a pure CPU loop with no coverage augmentation. The
deeper "eliminate pair_records via loss refactor" (Option 2) is out of scope.

### target_subgraphs consumption (verified)

- Training loss (`s2gae.py:515-517`) uses `target_graph.has_edge(...)` only.
- **Validation** (`s2gae.py:1334`) passes the full `nx.Graph` objects to
  `evaluate_graph_samples` as ground-truth graphs for degree/clustering/density
  metrics — so the payload must preserve node set + induced edges (which it
  does via `target_edges`), rehydrated into a real node-induced `nx.Graph`. No
  pickled graph needed.

## Section 3 — Incremental coverage augmentation

Rewrite `augment_plan_for_positive_edge_coverage` (`tccig/train.py:410`):

- Compute `covered` once up front (single `_covered_positive_edges` pass over
  base buckets). Compute `all_positive` once.
- Use canonical-tuple edges (`_canonical_edge(a, b)` from
  `src/topology/finetune_data.py:156` — import it, do not duplicate) rather than
  `frozenset`, for sort-stable deterministic ordering and consistency with the
  recent self-loop work.
- `uncovered = all_positive - covered`. Loop over a deterministic (sorted)
  ordering of `uncovered`. For each edge still uncovered: expand a coverage
  bucket, then add **that new bucket's** induced edges to `covered` and remove
  them from `uncovered`. Skip edges already drained by a previously added
  bucket. No full re-scan inside the loop.
- Final coverage computed from the maintained `covered` set; keep the
  `coverage == 1.0` hard-failure assertion.

Signature unchanged. `_covered_positive_edges` stays for the initial pass and
tests.

**Equivalence claim:** for the same graph, seed, strategy, and node_sizes, the
selected coverage buckets and `coverage_stats` (`base_bucket_count`,
`coverage_bucket_count`, `positive_edge_coverage`) remain semantically identical
to the current implementation, while avoiding repeated full rescans. (No
byte-identical guarantee on logs/JSON.)

## Section 4 — DDP coordination

`_load_or_build_topology_plan(...)` mirrors `_score_split` (`tccig/train.py:364`),
with the barrier on the miss path only:

```
all ranks compute metadata
all ranks try load_plan_cache
if hit:
    rehydrate and return        # no barrier on the common path
if miss:
    if main rank:
        build sampled subgraphs
        run incremental coverage if enabled
        build plan / payload
        write cache
    barrier                     # wait for main-rank write
    all ranks load_plan_cache
    if still missing: raise RuntimeError
```

This removes the 4×-redundant coverage augmentation across ranks.
`coverage_stats` (`base_bucket_count`, `coverage_bucket_count`,
`positive_edge_coverage`) are persisted in the payload/manifest so
`_build_train_topology_bundle` returns the same stats contract on a hit (read
back, not recomputed).

**Caveat:** assumes all ranks share the filesystem at `cache_dir`. The existing
score cache already assumes this, so it is not a new risk. The post-barrier
reload fails loudly (`RuntimeError`) rather than diverging silently.

## Section 5 — Error handling & testing

### Error handling
- Cache-key mismatch → treat as miss, rebuild. Never serve a stale plan.
- Corrupt/unparseable JSON or missing manifest → treat as miss, log warning,
  rebuild. Don't crash a run over a bad cache file.
- Post-barrier reload still missing → `RuntimeError` (mirrors `_score_split`).
- Coverage assertion (`positive_edge_coverage == 1.0`) stays a hard failure in
  the build path.

### Testing (TDD, pytest)
1. **Incremental coverage equivalence** — fixed small graph + seed + strategy +
   node_sizes: rewritten function produces the same selected coverage buckets
   and identical `coverage_stats` as a reference run; coverage == 1.0 holds.
2. **Round-trip fidelity** — `payload_to_plan(plan_to_payload(plan), graph=g)`
   equals the original: same buckets, `sampled_subgraphs`, regenerated
   `pair_records`, and `target_subgraphs` with identical nodes+edges (compare
   edge sets — fresh `nx.Graph`).
3. **Cache key sensitivity** — metadata changes on edges / node set / seed /
   strategy / node_sizes / samples_per_size / coverage flag; stable across
   strategy case (`mixed` == `MIXED`) and edge ordering.
4. **Cache hit skips work** — warm-cache build returns an equal plan without
   invoking the sampler (assert via monkeypatch/spy).
5. **Miss→write→reload** — first call writes `plans/{split}.json` + manifest;
   reload returns an equal plan; corrupt JSON falls back to rebuild.
6. **DDP coordination (lightweight)** — unit-test `_load_or_build_topology_plan`
   with a fake runtime: main-rank builds+writes, non-main reloads; missing-after-
   barrier raises. Full multi-process DDP not unit-tested (matches existing
   `_score_split` test scope).

Commands: `uv run python -m pytest`, `uv run ruff check`, `uv run mypy src`.
