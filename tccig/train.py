"""Standalone PRING-aligned TCCIG IO and orchestration entrypoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import logging
import math
import pickle
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import networkx as nx
import torch
import torch.distributed as _dist
import yaml  # type: ignore[import-untyped]
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.embed import load_cached_embedding
from src.pipeline.stages.train import build_model
from src.topology.finetune_data import (
    TOPOLOGY_EVAL_NODE_SIZES,
    InternalValidationPlan,
    build_internal_validation_plan,
    build_pair_supervision_graph,
    load_split_node_ids,
    sample_topology_evaluation_subgraphs,
)

from tccig.io import CandidatePair, PairTable, canonical_edge, read_pair_table, write_json
from tccig.rules import (
    DEFAULT_GRAPH_THRESHOLD,
    GraphRule,
    binary_metrics_at_threshold,
    edges_from_rule,
    parse_rules,
    threshold_for_target_precision,
)

LOGGER = logging.getLogger(__name__)

TOPOLOGY_METRIC_NAMES = [
    "graph_sim",
    "relative_density",
    "deg_dist_mmd",
    "cc_mmd",
    "laplacian_eigen_mmd",
]
TOPOLOGY_CSV_COLUMNS = [
    "scope",
    "node_size",
    "graph_count",
    *TOPOLOGY_METRIC_NAMES,
]