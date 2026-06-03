"""PRING pair-table IO for the standalone TCCIG scaffold."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CandidatePair:
    """Label-free candidate pair passed to scorer and refiner hooks."""

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


def _json_safe(value: object) -> object:
    """Convert common numeric/container values into JSON-safe objects."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, (int, float, str, bool)):
            return item
    return value
