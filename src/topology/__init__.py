"""Topology evaluation utilities for PRING-style graph reconstruction."""

from src.topology.finetune_data import (
    TOPOLOGY_EVAL_NODE_SIZES,
    TOPOLOGY_EVAL_SAMPLES_PER_SIZE,
    EdgeCoverEpochPlan,
    ExplicitNegativePairLookup,
    SubgraphPairChunk,
    build_explicit_negative_lookup,
    build_pair_supervision_graph,
    filter_graph_to_embedding_index,
    iter_subgraph_pair_chunks,
    load_split_node_ids,
    sample_edge_cover_subgraphs,
    sample_topology_evaluation_subgraphs,
    sample_training_subgraphs,
    summarize_edge_cover_epoch,
)
from src.topology.finetune_losses import (
    TopologyLossWeights,
    build_symmetric_adjacency,
    compute_topology_losses,
    soft_graph_similarity_loss,
    soft_relative_density_loss,
)
from src.topology.metrics import (
    compute_graph_similarity,
    compute_relative_density,
    evaluate_graph_samples,
    evaluate_predicted_graph,
    reconstruct_graph,
)
from src.topology.report import (
    build_human_table2_rows,
    load_human_table2_baselines,
    write_human_table2_reports,
)

__all__ = [
    "EdgeCoverEpochPlan",
    "ExplicitNegativePairLookup",
    "SubgraphPairChunk",
    "TopologyLossWeights",
    "TOPOLOGY_EVAL_NODE_SIZES",
    "TOPOLOGY_EVAL_SAMPLES_PER_SIZE",
    "build_explicit_negative_lookup",
    "build_pair_supervision_graph",
    "build_symmetric_adjacency",
    "build_human_table2_rows",
    "compute_graph_similarity",
    "compute_relative_density",
    "compute_topology_losses",
    "evaluate_graph_samples",
    "evaluate_predicted_graph",
    "filter_graph_to_embedding_index",
    "iter_subgraph_pair_chunks",
    "load_split_node_ids",
    "load_human_table2_baselines",
    "reconstruct_graph",
    "sample_edge_cover_subgraphs",
    "sample_topology_evaluation_subgraphs",
    "sample_training_subgraphs",
    "soft_graph_similarity_loss",
    "soft_relative_density_loss",
    "summarize_edge_cover_epoch",
    "write_human_table2_reports",
]
