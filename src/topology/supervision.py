"""Shared supervision graph loading for topology training stages."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from src.topology.finetune_data import (
    ExplicitNegativePairLookup,
    build_explicit_negative_lookup,
    build_pair_supervision_graph,
    load_split_node_ids,
)
from src.utils.config import ConfigDict, as_str, get_section


def resolve_supervision_dataset_path(
    *,
    stage_cfg: ConfigDict,
    dataloader_cfg: ConfigDict,
    stage_name: str,
    config_key: str,
    fallback_key: str,
) -> Path:
    """Resolve and validate one topology supervision dataset path."""
    raw_path = stage_cfg.get(config_key, dataloader_cfg.get(fallback_key, ""))
    path = Path(str(raw_path))
    if not str(raw_path):
        raise ValueError(f"Missing topology supervision dataset path for {stage_name}.{config_key}")
    if path.exists():
        return path
    raise FileNotFoundError(
        "Topology training supervision dataset not found: "
        f"{path}. Runtime generation is disabled; prepare them offline and update "
        f"{stage_name}.{config_key} before launching the pipeline."
    )


def load_supervision_graphs(
    *,
    config: ConfigDict,
    stage_cfg: ConfigDict,
    stage_name: str,
) -> tuple[nx.Graph, nx.Graph]:
    """Build train/validation supervision graphs without leaking validation edges."""
    data_cfg = get_section(config, "data_config")
    benchmark_cfg = get_section(data_cfg, "benchmark")
    dataloader_cfg = get_section(data_cfg, "dataloader")
    processed_dir = Path(str(benchmark_cfg.get("processed_dir", "")))
    species = as_str(benchmark_cfg.get("species", "human"), "data_config.benchmark.species")
    split_strategy = as_str(
        benchmark_cfg.get("split_strategy", "BFS"),
        "data_config.benchmark.split_strategy",
    ).upper()
    split_path = processed_dir / f"{species}_{split_strategy}_split.pkl"
    train_pair_path = resolve_supervision_dataset_path(
        stage_cfg=stage_cfg,
        dataloader_cfg=dataloader_cfg,
        stage_name=stage_name,
        config_key="supervision_train_dataset",
        fallback_key="train_dataset",
    )
    valid_pair_path = resolve_supervision_dataset_path(
        stage_cfg=stage_cfg,
        dataloader_cfg=dataloader_cfg,
        stage_name=stage_name,
        config_key="supervision_valid_dataset",
        fallback_key="valid_dataset",
    )
    train_nodes = load_split_node_ids(split_path=split_path, split_name="train")
    train_graph = build_pair_supervision_graph(
        pair_path=train_pair_path,
        node_ids=train_nodes,
    )
    internal_val_graph = build_pair_supervision_graph(
        pair_path=valid_pair_path,
        node_ids=train_nodes,
    )
    return train_graph, internal_val_graph


def load_train_negative_lookup(
    *,
    config: ConfigDict,
    stage_cfg: ConfigDict,
    stage_name: str,
) -> ExplicitNegativePairLookup:
    """Load explicit train negatives used for masked BCE supervision."""
    data_cfg = get_section(config, "data_config")
    benchmark_cfg = get_section(data_cfg, "benchmark")
    dataloader_cfg = get_section(data_cfg, "dataloader")
    processed_dir = Path(str(benchmark_cfg.get("processed_dir", "")))
    species = as_str(benchmark_cfg.get("species", "human"), "data_config.benchmark.species")
    split_strategy = as_str(
        benchmark_cfg.get("split_strategy", "BFS"),
        "data_config.benchmark.split_strategy",
    ).upper()
    split_path = processed_dir / f"{species}_{split_strategy}_split.pkl"
    train_nodes = load_split_node_ids(split_path=split_path, split_name="train")
    train_pair_path = resolve_supervision_dataset_path(
        stage_cfg=stage_cfg,
        dataloader_cfg=dataloader_cfg,
        stage_name=stage_name,
        config_key="supervision_train_dataset",
        fallback_key="train_dataset",
    )
    return build_explicit_negative_lookup(pair_path=train_pair_path, node_ids=train_nodes)
