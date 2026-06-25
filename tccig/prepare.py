"""Preparation helpers for the concrete TCCIG Accelerate pipeline."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch.utils.data import Dataset


class AcceleratorLike(Protocol):
    """Accelerate surface used by the TCCIG path."""

    device: torch.device
    is_main_process: bool
    use_distributed: bool
    process_index: int
    local_process_index: int
    num_processes: int

    def prepare(self, *components: object) -> object:
        """Prepare models, optimizers, or dataloaders."""

    def backward(self, loss: torch.Tensor) -> None:
        """Backpropagate one loss tensor."""

    def gather(self, value: torch.Tensor) -> torch.Tensor:
        """Gather tensors across processes."""

    def gather_for_metrics(self, value: torch.Tensor) -> torch.Tensor:
        """Gather metric tensors across processes."""

    def pad_across_processes(
        self,
        value: torch.Tensor,
        dim: int = 0,
        pad_index: int = 0,
        pad_first: bool = False,
    ) -> torch.Tensor:
        """Pad tensors across processes before gathering."""

    def reduce(self, value: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        """Reduce a tensor across processes."""

    def clip_grad_norm_(
        self,
        parameters: object,
        max_norm: float,
        norm_type: float = 2.0,
    ) -> torch.Tensor:
        """Clip the gradient norm of prepared parameters and return it."""

    def wait_for_everyone(self) -> None:
        """Synchronize ranks."""

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Return an unwrapped module."""

    def save(self, obj: object, f: str | Path, safe_serialization: bool = False) -> None:
        """Save an object from the main process."""


@dataclass(frozen=True)
class CandidatePair:
    """Label-free candidate pair passed to scorer and refiner boundaries."""

    protein_a: str
    protein_b: str


@dataclass(frozen=True)
class LabeledPair:
    """Parsed pair-table row retained inside the orchestrator."""

    protein_a: str
    protein_b: str
    label: int | None

    def candidate(self) -> CandidatePair:
        """Return the label-free candidate pair view."""
        return CandidatePair(protein_a=self.protein_a, protein_b=self.protein_b)


@dataclass(frozen=True)
class PairTable:
    """Filtered PRING pair table for one split."""

    split: str
    path: Path
    records: tuple[LabeledPair, ...]
    self_pair_rows: int

    @property
    def pairs(self) -> list[CandidatePair]:
        """Return label-free pairs in file order after self-pair filtering."""
        return [record.candidate() for record in self.records]

    @property
    def labels(self) -> list[int]:
        """Return labels for splits where labels are part of the contract."""
        labels: list[int] = []
        for record in self.records:
            if record.label is None:
                raise ValueError(f"{self.split} table does not expose labels")
            labels.append(int(record.label))
        return labels

    @property
    def positive_edges(self) -> list[tuple[str, str]]:
        """Return positive edges from labeled rows."""
        return [
            canonical_edge(record.protein_a, record.protein_b)
            for record in self.records
            if record.label is not None and record.label > 0
        ]


@dataclass(frozen=True)
class GraphRule:
    """Validation-selected graph assembly rule."""

    type: str
    value: float | int

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a serializable rule payload."""
        return {"type": self.type, "value": float(self.value)}


@dataclass(frozen=True)
class TCCIGRuntime:
    """Runtime details passed through the concrete TCCIG pipeline."""

    accelerator: AcceleratorLike
    device: str
    backend: str
    mixed_precision: str
    is_distributed: bool
    rank: int
    local_rank: int
    world_size: int
    is_main_process: bool


def canonical_edge(protein_a: str, protein_b: str) -> tuple[str, str]:
    """Return an undirected edge with stable endpoint ordering."""
    return (protein_a, protein_b) if protein_a <= protein_b else (protein_b, protein_a)


def read_pair_table(
    *,
    path: Path,
    split: str,
    expose_labels: bool,
) -> PairTable:
    """Read a PRING pair file and filter self-pair rows."""
    records: list[LabeledPair] = []
    self_pair_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = [part.strip() for part in line.rstrip("\n").split("\t")]
            if len(parts) < 2 or not parts[0] or not parts[1]:
                continue
            if parts[0] == parts[1]:
                self_pair_rows += 1
                continue
            label: int | None = None
            if expose_labels:
                if len(parts) < 3 or not parts[2]:
                    raise ValueError(f"{path} contains an unlabeled row in {split}")
                label = int(float(parts[2]))
            records.append(LabeledPair(protein_a=parts[0], protein_b=parts[1], label=label))

    if not records:
        raise ValueError(f"No usable non-self PPI records found in {path}")
    return PairTable(
        split=split,
        path=path,
        records=tuple(records),
        self_pair_rows=self_pair_rows,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON artifact with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def ordered_probabilities_from_indexed_rows(
    *,
    total: int,
    rows: torch.Tensor,
) -> list[float]:
    """Map ``(index, probability)`` rows to global order.

    Keeps the first occurrence of duplicate indices (Accelerate ``even_batches``
    repeats tail samples) and raises if any index is uncovered.
    """
    ordered: dict[int, float] = {}
    for row in rows.detach().cpu():
        index = int(row[0].item())
        if index < 0 or index >= total:
            continue
        if index not in ordered:
            ordered[index] = float(row[1].item())
    missing = [index for index in range(total) if index not in ordered]
    if missing:
        raise ValueError(f"Missing probabilities for indices: {missing[:10]}")
    return [ordered[index] for index in range(total)]


def edges_from_rule(
    *,
    pairs: Sequence[CandidatePair],
    probabilities: Sequence[float],
    rule: GraphRule,
) -> list[tuple[str, str]]:
    """Select graph edges from candidate probabilities under one rule."""
    if len(pairs) != len(probabilities):
        raise ValueError("pairs and probabilities must have matching lengths")
    if rule.type == "threshold":
        threshold = float(rule.value)
        return [
            canonical_edge(pair.protein_a, pair.protein_b)
            for pair, probability in zip(pairs, probabilities, strict=True)
            if float(probability) >= threshold
        ]
    raise ValueError(f"Unsupported graph rule type: {rule.type}")


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


@dataclass(frozen=True)
class RefineRequest:
    """Concrete request for S2GAE refined prediction."""

    split: str
    pairs: Sequence[CandidatePair]
    pairwise_probabilities: Sequence[float]
    pairwise_graph_edges: Sequence[tuple[str, str]]
    refiner_state: object
    runtime: TCCIGRuntime
    config: Mapping[str, object]


@dataclass(frozen=True)
class TCCIGPipelineResult:
    """High-level result from one concrete TCCIG pipeline run."""

    manifest: dict[str, object]
    pairwise_input_threshold: dict[str, object]
    refined_output_rule: dict[str, object]
    pairwise_metrics: dict[str, float]
    topology_metrics: dict[str, float]


@dataclass(frozen=True)
class PairScoreRecord:
    """One label-free pair-scoring work item."""

    index: int
    pair: CandidatePair


class PairScoreDataset(Dataset[PairScoreRecord]):
    """Dataset over candidate pairs preserving original file order."""

    def __init__(self, pairs: Sequence[CandidatePair]) -> None:
        self._pairs = list(pairs)

    def __len__(self) -> int:
        """Return pair count."""
        return len(self._pairs)

    def __getitem__(self, index: int) -> PairScoreRecord:
        """Return one pair-scoring record."""
        return PairScoreRecord(index=index, pair=self._pairs[index])


def collate_pair_score_records(records: Sequence[PairScoreRecord]) -> dict[str, object]:
    """Collate pair-scoring records without exposing labels."""
    return {
        "index": torch.tensor([record.index for record in records], dtype=torch.long),
        "protein_a": [record.pair.protein_a for record in records],
        "protein_b": [record.pair.protein_b for record in records],
    }


@dataclass(frozen=True)
class EdgeSamplingConfig:
    """Scorer-error target sampling controls."""

    hard_fraction: float = 0.7
    easy_anchor_fraction: float = 0.3
    seed: int = 0
    reshuffle_easy_each_epoch: bool = True


@dataclass(frozen=True)
class EdgeTarget:
    """One sampled edge target for S2GAE residual training."""

    pair_index: int
    label: int
    quadrant: str
    mask_input_edge: bool


class EdgeTargetDataset(Dataset[EdgeTarget]):
    """Dataset over sampled S2GAE edge targets."""

    def __init__(self, targets: Sequence[EdgeTarget]) -> None:
        if not targets:
            raise ValueError("S2GAE edge target dataset must not be empty")
        self._targets = list(targets)

    def __len__(self) -> int:
        """Return target count."""
        return len(self._targets)

    def __getitem__(self, index: int) -> EdgeTarget:
        """Return one sampled edge target."""
        return self._targets[index]


def collate_edge_targets(targets: Sequence[EdgeTarget]) -> dict[str, torch.Tensor]:
    """Collate sampled edge targets into tensors."""
    return {
        "pair_index": torch.tensor([target.pair_index for target in targets], dtype=torch.long),
        "label": torch.tensor([float(target.label) for target in targets], dtype=torch.float32),
        "mask_input_edge": torch.tensor(
            [bool(target.mask_input_edge) for target in targets],
            dtype=torch.bool,
        ),
    }


def load_pring_tables(processed_dir: Path) -> dict[str, PairTable]:
    """Load the PRING split files used by the concrete TCCIG pipeline."""
    return {
        "train": read_pair_table(
            path=processed_dir / "human_train_ppi_ratio5_exclusive.txt",
            split="train",
            expose_labels=True,
        ),
        "validation": read_pair_table(
            path=processed_dir / "human_val_ppi_ratio5_exclusive.txt",
            split="validation",
            expose_labels=True,
        ),
        "pairwise_test": read_pair_table(
            path=processed_dir / "human_test_ppi.txt",
            split="pairwise_test",
            expose_labels=True,
        ),
        "topology_test": read_pair_table(
            path=processed_dir / "all_test_ppi.txt",
            split="topology_test",
            expose_labels=False,
        ),
    }


def parse_edge_sampling_config(raw_config: object) -> EdgeSamplingConfig:
    """Parse scorer-error target sampling controls from config."""
    if raw_config is None:
        return EdgeSamplingConfig()
    if not isinstance(raw_config, Mapping):
        raise ValueError("refiner.edge_sampling must be a mapping")
    hard_fraction = _probability(raw_config.get("hard_fraction", 0.7), "hard_fraction")
    easy_fraction = _probability(
        raw_config.get("easy_anchor_fraction", 0.3),
        "easy_anchor_fraction",
    )
    if hard_fraction <= 0.0:
        raise ValueError("refiner.edge_sampling.hard_fraction must be positive")
    if easy_fraction < 0.0:
        raise ValueError("refiner.edge_sampling.easy_anchor_fraction must be non-negative")
    return EdgeSamplingConfig(
        hard_fraction=hard_fraction,
        easy_anchor_fraction=easy_fraction,
        seed=_non_negative_int(raw_config.get("seed", 0), "refiner.edge_sampling.seed"),
        reshuffle_easy_each_epoch=_bool(
            raw_config.get("reshuffle_easy_each_epoch", True),
            "refiner.edge_sampling.reshuffle_easy_each_epoch",
        ),
    )


def classify_scorer_error_targets(
    *,
    pairs: Sequence[CandidatePair],
    labels: Sequence[int],
    pairwise_graph_edges: Sequence[tuple[str, str]],
) -> dict[str, list[EdgeTarget]]:
    """Classify train ratio5 rows into FP/FN/TP/TN under frozen ``G_pairwise``."""
    if len(pairs) != len(labels):
        raise ValueError("pairs and labels must have matching lengths")
    input_edges = {
        canonical_edge(protein_a, protein_b) for protein_a, protein_b in pairwise_graph_edges
    }
    buckets: dict[str, list[EdgeTarget]] = {"fp": [], "fn": [], "tp": [], "tn": []}
    for index, (pair, label) in enumerate(zip(pairs, labels, strict=True)):
        edge = canonical_edge(pair.protein_a, pair.protein_b)
        is_input_edge = edge in input_edges
        normalized_label = int(label)
        if is_input_edge and normalized_label == 0:
            quadrant = "fp"
        elif (not is_input_edge) and normalized_label == 1:
            quadrant = "fn"
        elif is_input_edge and normalized_label == 1:
            quadrant = "tp"
        else:
            quadrant = "tn"
        buckets[quadrant].append(
            EdgeTarget(
                pair_index=index,
                label=normalized_label,
                quadrant=quadrant,
                mask_input_edge=is_input_edge,
            )
        )
    return buckets


def sample_epoch_edge_targets(
    *,
    quadrants: Mapping[str, Sequence[EdgeTarget]],
    sampling: EdgeSamplingConfig,
    epoch: int,
) -> list[EdgeTarget]:
    """Return one epoch of scorer-error-focused edge targets."""
    hard_targets = list(quadrants.get("fp", ())) + list(quadrants.get("fn", ()))
    tp_targets = list(quadrants.get("tp", ()))
    tn_targets = list(quadrants.get("tn", ()))
    easy_targets = tp_targets + tn_targets
    if not hard_targets and not easy_targets:
        raise ValueError("No train edge targets available for S2GAE")

    if hard_targets:
        easy_count = math.ceil(
            len(hard_targets) * sampling.easy_anchor_fraction / sampling.hard_fraction
        )
    else:
        easy_count = len(easy_targets)

    rng_seed = sampling.seed + (epoch if sampling.reshuffle_easy_each_epoch else 0)
    generator = torch.Generator()
    generator.manual_seed(rng_seed)
    easy_sample = _sample_balanced_easy_targets(
        tp_targets=tp_targets,
        tn_targets=tn_targets,
        count=easy_count,
        generator=generator,
    )
    return [*hard_targets, *easy_sample]


def strict_reject_legacy_hooks(config: Mapping[str, object]) -> None:
    """Reject removed dynamic hook fields."""
    pairwise_cfg = config.get("pairwise_scorer", {})
    refiner_cfg = config.get("refiner", {})
    if isinstance(pairwise_cfg, Mapping) and "target" in pairwise_cfg:
        raise ValueError("pairwise_scorer.target is no longer supported")
    if isinstance(refiner_cfg, Mapping):
        if "train_target" in refiner_cfg:
            raise ValueError("refiner.train_target is no longer supported")
        if "predict_target" in refiner_cfg:
            raise ValueError("refiner.predict_target is no longer supported")


def score_cache_metadata(
    *,
    split: str,
    pairs: Sequence[CandidatePair],
    scorer_config: Mapping[str, object],
) -> dict[str, object]:
    """Build strict metadata for a pairwise score cache."""
    return {
        "version": 2,
        "split": split,
        "pair_count": len(pairs),
        "pair_hash": ordered_pair_hash(pairs),
        "scorer": {
            "model_config_sha256": optional_file_sha256(scorer_config.get("model_config_path")),
            "checkpoint_sha256": optional_file_sha256(scorer_config.get("checkpoint_path")),
            "embedding_index_sha256": embedding_index_sha256(scorer_config),
            "max_sequence_length": scorer_config.get("max_sequence_length"),
        },
    }


def ordered_pair_hash(pairs: Sequence[CandidatePair]) -> str:
    """Hash pairs in file order."""
    digest = hashlib.sha256()
    for pair in pairs:
        digest.update(pair.protein_a.encode("utf-8"))
        digest.update(b"\0")
        digest.update(pair.protein_b.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def embedding_index_sha256(scorer_config: Mapping[str, object]) -> str | None:
    """Hash the configured pairwise scorer embedding index when available."""
    cache_dir = scorer_config.get("embedding_cache_dir")
    if not isinstance(cache_dir, str) or not cache_dir:
        return None
    return file_sha256(Path(cache_dir) / "index.json")


def optional_file_sha256(raw_path: object) -> str | None:
    """Hash an optional config path."""
    if not isinstance(raw_path, str) or not raw_path:
        return None
    return file_sha256(Path(raw_path))


def file_sha256(path: Path) -> str:
    """Return a SHA256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, (int, float, str, bool)):
            return item
    return value


def _sample_balanced_easy_targets(
    *,
    tp_targets: Sequence[EdgeTarget],
    tn_targets: Sequence[EdgeTarget],
    count: int,
    generator: torch.Generator,
) -> list[EdgeTarget]:
    if count <= 0:
        return []
    tp_count = count // 2
    tn_count = count - tp_count
    sampled_tp = _sample_targets(tp_targets, tp_count, generator)
    sampled_tn = _sample_targets(tn_targets, tn_count, generator)
    shortfall = count - len(sampled_tp) - len(sampled_tn)
    if shortfall > 0:
        fallback_pool = tn_targets if len(sampled_tp) < tp_count else tp_targets
        sampled_tn.extend(_sample_targets(fallback_pool, shortfall, generator))
    return [*sampled_tp, *sampled_tn]


def _sample_targets(
    targets: Sequence[EdgeTarget],
    count: int,
    generator: torch.Generator,
) -> list[EdgeTarget]:
    if count <= 0 or not targets:
        return []
    if count <= len(targets):
        indices = torch.randperm(len(targets), generator=generator)[:count].tolist()
    else:
        indices = torch.randint(len(targets), (count,), generator=generator).tolist()
    return [targets[int(index)] for index in indices]


def _probability(value: object, name: str) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"refiner.edge_sampling.{name} must be in [0, 1]") from error
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"refiner.edge_sampling.{name} must be in [0, 1]")
    return parsed


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a non-negative integer") from error
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _bool(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be a boolean")
