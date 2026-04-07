"""Embedding configuration models, constants, and lightweight protocols."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from src.utils.config import ConfigDict, as_str, get_section

CACHE_SCHEMA_VERSION = 1
INDEX_FILENAME = "index.json"
METADATA_FILENAME = "metadata.json"
EMBEDDINGS_SUBDIR = "embeddings"
DEFAULT_ESM3_MODEL_NAME = "esm3_sm_open_v1"
DEFAULT_EMBEDDING_DEVICE = "auto"
CSV_SUFFIXES = (".csv",)
FASTA_SUFFIXES = (".fasta", ".fa", ".faa", ".fna", ".fas")
DEFAULT_ID_COLUMNS = (
    "uniprot_id",
    "uniprotid",
    "protein_id",
    "protein",
    "id",
    "accession",
)
DEFAULT_SEQUENCE_COLUMNS = ("sequence", "seq", "protein_sequence")
FASTA_UNIPROT_PATTERN = re.compile(r"\|([^|]+)\|")
EMBED_PROGRESS_LOGGER_NAME = "grand.embed.progress"


@dataclass(frozen=True)
class EmbeddingCacheManifest:
    """Resolved embedding cache details.

    Attributes:
        cache_dir: Embedding cache root directory.
        index: Mapping from protein ID to relative embedding file path.
        required_ids: IDs required by configured split files.
    """

    cache_dir: Path
    index: dict[str, str]
    required_ids: frozenset[str]


@dataclass(frozen=True)
class _EmbeddingSettings:
    """Runtime embedding settings parsed from config."""

    source: str
    cache_dir: Path
    model_name: str
    device: str
    sequence_file: Path | None
    id_column: str | None
    sequence_column: str | None


class _Esm3ModelProtocol(Protocol):
    """Protocol for minimal ESM3 model interactions."""

    def to(self, device: str) -> _Esm3ModelProtocol:
        """Move model to target device."""

    def eval(self) -> _Esm3ModelProtocol:
        """Switch model to eval mode."""

    def encode(self, protein: object) -> object:
        """Encode an ESM protein object."""

    def logits(self, protein_tensor: object, logits_config: object) -> _LogitsOutputProtocol:
        """Run logits call returning embeddings."""


class _Esm3ClassProtocol(Protocol):
    """Protocol for ESM3 class constructor."""

    @classmethod
    def from_pretrained(cls, model_name: str) -> _Esm3ModelProtocol:
        """Instantiate a pretrained ESM3 model."""


class _ProteinCtorProtocol(Protocol):
    """Protocol for ESM protein constructor."""

    def __call__(self, *, sequence: str) -> object:
        """Build protein object from sequence."""


class _LogitsConfigCtorProtocol(Protocol):
    """Protocol for logits-config constructor."""

    def __call__(self, *, sequence: bool, return_embeddings: bool) -> object:
        """Build logits config."""


class _LogitsOutputProtocol(Protocol):
    """Protocol for ESM logits output."""

    embeddings: torch.Tensor | None


def _optional_string(value: object) -> str | None:
    """Return a stripped string when provided, else ``None``."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _parse_embedding_settings(config: ConfigDict) -> _EmbeddingSettings:
    """Parse ``data_config.embeddings`` runtime settings."""
    data_cfg = get_section(config, "data_config")
    embeddings_cfg = get_section(data_cfg, "embeddings")

    source = as_str(embeddings_cfg.get("source", ""), "data_config.embeddings.source").lower()
    cache_dir_raw = _optional_string(embeddings_cfg.get("cache_dir"))
    if cache_dir_raw is None:
        raise ValueError("data_config.embeddings.cache_dir must be a non-empty path")

    model_name = _optional_string(embeddings_cfg.get("model_name")) or DEFAULT_ESM3_MODEL_NAME
    device = _optional_string(embeddings_cfg.get("device")) or DEFAULT_EMBEDDING_DEVICE
    sequence_file_raw = _optional_string(embeddings_cfg.get("sequence_file"))
    id_column = _optional_string(embeddings_cfg.get("id_column"))
    sequence_column = _optional_string(embeddings_cfg.get("sequence_column"))

    return _EmbeddingSettings(
        source=source,
        cache_dir=Path(cache_dir_raw),
        model_name=model_name,
        device=device,
        sequence_file=Path(sequence_file_raw) if sequence_file_raw is not None else None,
        id_column=id_column,
        sequence_column=sequence_column,
    )
