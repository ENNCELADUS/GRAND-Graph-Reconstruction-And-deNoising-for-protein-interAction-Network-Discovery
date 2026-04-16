"""Subgraph sampling and pair materialization for topology fine-tuning."""

from __future__ import annotations

import pickle
import random
from collections import OrderedDict, deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import torch
from torch.nn.utils.rnn import pad_sequence

from src.embed import load_cached_embedding

SUPPORTED_SAMPLING_STRATEGIES = {"BFS", "DFS", "RANDOM_WALK", "MIXED"}
BASE_SAMPLING_STRATEGIES: tuple[str, ...] = ("BFS", "DFS", "RANDOM_WALK")
TOPOLOGY_EVAL_NODE_SIZES: tuple[int, ...] = (20, 40, 60, 80, 100, 120, 140, 160, 180, 200)
TOPOLOGY_EVAL_SAMPLES_PER_SIZE = 50


@dataclass(frozen=True)
class SubgraphPairChunk:
    """Chunked pair batch materialized from one sampled training subgraph."""

    nodes: tuple[str, ...]
    emb_a: torch.Tensor
    emb_b: torch.Tensor
    len_a: torch.Tensor
    len_b: torch.Tensor
    label: torch.Tensor
    pair_index_a: torch.Tensor
    pair_index_b: torch.Tensor
    bce_label: torch.Tensor | None = None
    bce_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class ExplicitNegativePairLookup:
    """Explicitly supervised negative pairs available for BCE learning."""

    negative_pairs: frozenset[tuple[str, str]]
    partners_by_node: Mapping[str, frozenset[str]]


@dataclass(frozen=True)
class EdgeCoverEpochPlan:
    """Training-epoch plan and summary for edge-cover sampling."""

    subgraphs: tuple[tuple[str, ...], ...]
    total_positive_edges: int
    covered_positive_edges: int
    positive_edge_coverage_ratio: float
    mean_positive_edge_reuse: float


@dataclass(frozen=True)
class InternalValidationPairRecord:
    """Flattened pair metadata for batched internal topology validation."""

    subgraph_index: int
    pair_index_a: int
    pair_index_b: int
    protein_a: str
    protein_b: str


@dataclass(frozen=True)
class InternalValidationNodeBucketPlan:
    """Precomputed node bucket used for batched internal validation inference."""

    node_size: int
    sampled_subgraphs: tuple[tuple[str, ...], ...]
    target_subgraphs: tuple[nx.Graph, ...]
    pair_records: tuple[InternalValidationPairRecord, ...]


@dataclass(frozen=True)
class InternalValidationPlan:
    """Fully materialized internal-validation surface reused across epochs."""

    buckets: tuple[InternalValidationNodeBucketPlan, ...]
    protein_ids: frozenset[str]
    total_subgraphs: int
    total_pairs: int


class EmbeddingRepository:
    """Byte-bounded CPU LRU cache for embedding tensors."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        embedding_index: Mapping[str, str],
        input_dim: int,
        max_sequence_length: int,
        max_cache_bytes: int = 1_073_741_824,
    ) -> None:
        if max_cache_bytes <= 0:
            raise ValueError("max_cache_bytes must be positive")
        self._cache_dir = cache_dir
        self._embedding_index = embedding_index
        self._input_dim = input_dim
        self._max_sequence_length = max_sequence_length
        self._max_cache_bytes = max_cache_bytes
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._cache_bytes = 0

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> int:
        return int(tensor.element_size() * tensor.numel())

    def _evict_as_needed(self, incoming_bytes: int) -> None:
        while self._cache and self._cache_bytes + incoming_bytes > self._max_cache_bytes:
            _, tensor = self._cache.popitem(last=False)
            self._cache_bytes -= self._tensor_bytes(tensor)

    def get(self, protein_id: str) -> torch.Tensor:
        """Return one embedding tensor and update its LRU position."""
        cached = self._cache.get(protein_id)
        if cached is not None:
            self._cache.move_to_end(protein_id)
            return cached

        embedding = load_cached_embedding(
            cache_dir=self._cache_dir,
            index=self._embedding_index,
            protein_id=protein_id,
            expected_input_dim=self._input_dim,
            max_sequence_length=self._max_sequence_length,
        )
        tensor_bytes = self._tensor_bytes(embedding)
        if tensor_bytes <= self._max_cache_bytes:
            self._evict_as_needed(tensor_bytes)
            self._cache[protein_id] = embedding
            self._cache.move_to_end(protein_id)
            self._cache_bytes += tensor_bytes
        return embedding

    def get_many(self, protein_ids: Sequence[str]) -> dict[str, torch.Tensor]:
        """Return a dictionary of embeddings for the requested proteins."""
        return {protein_id: self.get(protein_id) for protein_id in protein_ids}

    def preload(self, protein_ids: Iterable[str]) -> int:
        """Best-effort preload of the requested protein IDs into the LRU."""
        for protein_id in protein_ids:
            self.get(protein_id)
        return len(self._cache)


def _canonical_edge(node_a: str, node_b: str) -> tuple[str, str]:
    """Return a stable undirected edge representation."""
    return (node_a, node_b) if node_a <= node_b else (node_b, node_a)


def _graph_positive_edges(graph: nx.Graph) -> set[tuple[str, str]]:
    """Return the canonical positive-edge set for a supervision graph."""
    return {_canonical_edge(node_a, node_b) for node_a, node_b in graph.edges()}


def filter_graph_to_embedding_index(
    *,
    graph: nx.Graph,
    embedding_index: Mapping[str, str],
) -> nx.Graph:
    """Return the node-induced subgraph restricted to embeddable proteins."""
    available_nodes = [node for node in graph.nodes if node in embedding_index]
    if not available_nodes:
        raise ValueError("No train-graph nodes have cached embeddings")
    return graph.subgraph(available_nodes).copy()


def load_split_node_ids(*, split_path: Path, split_name: str) -> set[str]:
    """Load protein IDs for one split from a PRING ``*_split.pkl`` file."""
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with split_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Split pickle must contain a dictionary: {split_path}")

    split_ids = payload.get(split_name)
    if not isinstance(split_ids, set) or not all(isinstance(item, str) for item in split_ids):
        raise ValueError(f"Split '{split_name}' in {split_path} must be a set of protein IDs")
    return set(split_ids)


def build_pair_supervision_graph(
    *,
    pair_path: Path,
    node_ids: set[str],
) -> nx.Graph:
    """Build a supervision graph from in-split positive pairs and a fixed node universe."""
    if not pair_path.exists():
        raise FileNotFoundError(f"Pair dataset not found: {pair_path}")

    graph = nx.Graph()
    graph.add_nodes_from(sorted(node_ids))
    with pair_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = [part.strip() for part in line.strip().split("\t")]
            if len(parts) < 3 or not parts[0] or not parts[1] or not parts[2]:
                continue
            try:
                label = int(float(parts[2]))
            except ValueError:
                continue
            if label <= 0:
                continue
            if parts[0] not in node_ids or parts[1] not in node_ids:
                continue
            graph.add_edge(parts[0], parts[1])
    return graph


def build_explicit_negative_lookup(
    *,
    pair_path: Path,
    node_ids: set[str],
) -> ExplicitNegativePairLookup:
    """Load in-split explicit negatives from a labeled PRING pair file."""
    if not pair_path.exists():
        raise FileNotFoundError(f"Pair dataset not found: {pair_path}")

    negative_pairs: set[tuple[str, str]] = set()
    partners_by_node: dict[str, set[str]] = {}
    with pair_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = [part.strip() for part in line.strip().split("\t")]
            if len(parts) < 3 or not parts[0] or not parts[1] or not parts[2]:
                continue
            try:
                label = int(float(parts[2]))
            except ValueError:
                continue
            if label >= 0 and label != 0:
                continue
            if parts[0] not in node_ids or parts[1] not in node_ids:
                continue
            pair = _canonical_edge(parts[0], parts[1])
            negative_pairs.add(pair)
            partners_by_node.setdefault(pair[0], set()).add(pair[1])
            partners_by_node.setdefault(pair[1], set()).add(pair[0])

    return ExplicitNegativePairLookup(
        negative_pairs=frozenset(negative_pairs),
        partners_by_node={
            node: frozenset(sorted(partners)) for node, partners in partners_by_node.items()
        },
    )


def _sample_target_size(
    *,
    graph: nx.Graph,
    min_nodes: int,
    max_nodes: int,
    rng: random.Random,
) -> int:
    """Sample a valid target node count for one subgraph."""
    if min_nodes <= 1:
        raise ValueError("min_nodes must be greater than 1")
    if max_nodes < min_nodes:
        raise ValueError("max_nodes must be >= min_nodes")
    graph_size = graph.number_of_nodes()
    if graph_size < min_nodes:
        raise ValueError(
            f"Train graph is too small for subgraph sampling: {graph_size} < {min_nodes}"
        )
    upper_bound = min(max_nodes, graph_size)
    return rng.randint(min_nodes, upper_bound)


def _normalize_sampling_strategy(strategy: str) -> str:
    """Return a validated uppercase sampling strategy."""
    normalized_strategy = strategy.upper()
    if normalized_strategy in SUPPORTED_SAMPLING_STRATEGIES:
        return normalized_strategy
    supported = ", ".join(sorted(SUPPORTED_SAMPLING_STRATEGIES))
    raise ValueError(f"Unsupported subgraph sampling strategy: {strategy} ({supported})")


def _choose_sampling_strategy(
    *,
    strategy: str,
    rng: random.Random,
) -> str:
    """Resolve one concrete sampling strategy, expanding MIXED lazily."""
    if strategy == "MIXED":
        return rng.choice(list(BASE_SAMPLING_STRATEGIES))
    return strategy


def _fallback_complete_nodes(
    *,
    nodes_in_order: list[str],
    graph_nodes: Sequence[str],
    target_size: int,
    rng: random.Random,
) -> tuple[str, ...]:
    """Pad a partially explored node set with random unseen nodes."""
    if len(nodes_in_order) >= target_size:
        return tuple(nodes_in_order[:target_size])

    seen = set(nodes_in_order)
    remaining = [node for node in graph_nodes if node not in seen]
    rng.shuffle(remaining)
    nodes_in_order.extend(remaining[: target_size - len(nodes_in_order)])
    return tuple(nodes_in_order)


def _sample_frontier_nodes(
    *,
    graph: nx.Graph,
    target_size: int,
    rng: random.Random,
    depth_first: bool,
) -> tuple[str, ...]:
    """Sample nodes by randomized frontier expansion with restart fallback."""
    graph_nodes = list(graph.nodes)
    frontier = deque([rng.choice(graph_nodes)])
    visited: set[str] = set()
    nodes_in_order: list[str] = []

    while frontier and len(nodes_in_order) < target_size:
        node = frontier.pop() if depth_first else frontier.popleft()
        if node in visited:
            continue
        visited.add(node)
        nodes_in_order.append(node)
        neighbors = list(graph.neighbors(node))
        rng.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in visited:
                frontier.append(neighbor)

        if not frontier and len(nodes_in_order) < target_size:
            unseen = [candidate for candidate in graph_nodes if candidate not in visited]
            if unseen:
                frontier.append(rng.choice(unseen))

    return _fallback_complete_nodes(
        nodes_in_order=nodes_in_order,
        graph_nodes=graph_nodes,
        target_size=target_size,
        rng=rng,
    )


def _sample_bfs_nodes(graph: nx.Graph, target_size: int, rng: random.Random) -> tuple[str, ...]:
    """Sample one node set by randomized breadth-first expansion."""
    return _sample_frontier_nodes(
        graph=graph,
        target_size=target_size,
        rng=rng,
        depth_first=False,
    )


def _sample_dfs_nodes(graph: nx.Graph, target_size: int, rng: random.Random) -> tuple[str, ...]:
    """Sample one node set by randomized depth-first expansion."""
    return _sample_frontier_nodes(
        graph=graph,
        target_size=target_size,
        rng=rng,
        depth_first=True,
    )


def _sample_random_walk_nodes(
    graph: nx.Graph,
    target_size: int,
    rng: random.Random,
) -> tuple[str, ...]:
    """Sample one node set by randomized walk with restart fallback."""
    graph_nodes = list(graph.nodes)
    current = rng.choice(graph_nodes)
    visited = {current}
    nodes_in_order = [current]
    max_steps = max(target_size * 20, 100)

    for _ in range(max_steps):
        if len(nodes_in_order) >= target_size:
            break
        neighbors = list(graph.neighbors(current))
        if neighbors:
            current = rng.choice(neighbors)
        else:
            unseen = [candidate for candidate in graph_nodes if candidate not in visited]
            current = rng.choice(unseen) if unseen else rng.choice(graph_nodes)
        if current not in visited:
            visited.add(current)
            nodes_in_order.append(current)

    return _fallback_complete_nodes(
        nodes_in_order=nodes_in_order,
        graph_nodes=graph_nodes,
        target_size=target_size,
        rng=rng,
    )


def _build_bfs_order(
    *,
    graph: nx.Graph,
    seed_nodes: Sequence[str],
    rng: random.Random,
) -> list[str]:
    """Return a randomized BFS traversal order from the seed nodes."""
    return _build_frontier_order(
        graph=graph,
        seed_nodes=seed_nodes,
        rng=rng,
        depth_first=False,
    )


def _build_dfs_order(
    *,
    graph: nx.Graph,
    seed_nodes: Sequence[str],
    rng: random.Random,
) -> list[str]:
    """Return a randomized DFS traversal order from the seed nodes."""
    return _build_frontier_order(
        graph=graph,
        seed_nodes=seed_nodes,
        rng=rng,
        depth_first=True,
    )


def _build_frontier_order(
    *,
    graph: nx.Graph,
    seed_nodes: Sequence[str],
    rng: random.Random,
    depth_first: bool,
) -> list[str]:
    """Return a randomized BFS/DFS traversal order with restart fallback."""
    frontier = deque(reversed(seed_nodes) if depth_first else seed_nodes)
    visited = set(seed_nodes)
    order: list[str] = list(seed_nodes)
    graph_nodes = list(graph.nodes)

    while len(order) < graph.number_of_nodes():
        while frontier and len(order) < graph.number_of_nodes():
            node = frontier.pop() if depth_first else frontier.popleft()
            neighbors = [neighbor for neighbor in graph.neighbors(node) if neighbor not in visited]
            rng.shuffle(neighbors)
            for neighbor in neighbors:
                visited.add(neighbor)
                frontier.append(neighbor)
                order.append(neighbor)

        if len(order) >= graph.number_of_nodes():
            break
        unseen = [candidate for candidate in graph_nodes if candidate not in visited]
        if not unseen:
            break
        restart = rng.choice(unseen)
        visited.add(restart)
        frontier.append(restart)
        order.append(restart)

    return order


def _build_random_walk_order(
    *,
    graph: nx.Graph,
    seed_nodes: Sequence[str],
    rng: random.Random,
) -> list[str]:
    """Return a randomized walk order from the seed nodes with restart fallback."""
    graph_nodes = list(graph.nodes)
    order: list[str] = []
    visited: set[str] = set()
    for node in seed_nodes:
        if node not in visited:
            order.append(node)
            visited.add(node)

    current = seed_nodes[-1]
    max_steps = max(graph.number_of_nodes() * 20, 100)
    for _ in range(max_steps):
        if len(order) >= graph.number_of_nodes():
            break
        neighbors = list(graph.neighbors(current))
        if neighbors:
            current = rng.choice(neighbors)
        else:
            unseen = [candidate for candidate in graph_nodes if candidate not in visited]
            if not unseen:
                break
            current = rng.choice(unseen)
        if current not in visited:
            order.append(current)
            visited.add(current)

    if len(order) < graph.number_of_nodes():
        unseen = [candidate for candidate in graph_nodes if candidate not in visited]
        rng.shuffle(unseen)
        order.extend(unseen)
    return order


def _traversal_rank(
    *,
    graph: nx.Graph,
    seed_nodes: Sequence[str],
    strategy: str,
    rng: random.Random,
) -> dict[str, int]:
    """Return a node-to-rank mapping for traversal tie-breaking."""
    if strategy == "BFS":
        order = _build_bfs_order(graph=graph, seed_nodes=seed_nodes, rng=rng)
    elif strategy == "DFS":
        order = _build_dfs_order(graph=graph, seed_nodes=seed_nodes, rng=rng)
    else:
        order = _build_random_walk_order(graph=graph, seed_nodes=seed_nodes, rng=rng)
    return {node: index for index, node in enumerate(order)}


def _selected_positive_edges(
    *,
    graph: nx.Graph,
    nodes: Sequence[str],
) -> set[tuple[str, str]]:
    """Return positive train-graph edges fully contained in a node set."""
    node_list = tuple(nodes)
    selected_edges: set[tuple[str, str]] = set()
    for index, node_a in enumerate(node_list):
        for node_b in node_list[index + 1 :]:
            if graph.has_edge(node_a, node_b):
                selected_edges.add(_canonical_edge(node_a, node_b))
    return selected_edges


def summarize_edge_cover_epoch(
    *,
    graph: nx.Graph,
    subgraphs: Sequence[Sequence[str]],
) -> EdgeCoverEpochPlan:
    """Summarize positive-edge coverage and reuse for a sampled epoch."""
    total_positive_edges = _graph_positive_edges(graph)
    edge_cover_counts: dict[tuple[str, str], int] = dict.fromkeys(total_positive_edges, 0)
    for nodes in subgraphs:
        for edge in _selected_positive_edges(graph=graph, nodes=nodes):
            edge_cover_counts[edge] += 1

    covered_positive_edges = sum(1 for count in edge_cover_counts.values() if count > 0)
    total_positive_edge_count = len(total_positive_edges)
    if total_positive_edge_count == 0:
        coverage_ratio = 1.0
        mean_reuse = 0.0
    else:
        coverage_ratio = covered_positive_edges / float(total_positive_edge_count)
        mean_reuse = sum(count for count in edge_cover_counts.values() if count > 0) / float(
            max(1, covered_positive_edges)
        )

    return EdgeCoverEpochPlan(
        subgraphs=tuple(tuple(nodes) for nodes in subgraphs),
        total_positive_edges=total_positive_edge_count,
        covered_positive_edges=covered_positive_edges,
        positive_edge_coverage_ratio=coverage_ratio,
        mean_positive_edge_reuse=mean_reuse,
    )


def _default_edge_chunk_size(max_nodes: int) -> int:
    """Return the default positive-edge chunk size for one subgraph core."""
    if max_nodes <= 1:
        raise ValueError("max_nodes must be greater than 1")
    return max(1, (max_nodes * (max_nodes - 1)) // 4)


def _partition_edges(
    *,
    positive_edges: Sequence[tuple[str, str]],
    chunk_size: int,
) -> tuple[tuple[tuple[str, str], ...], ...]:
    """Partition shuffled positive edges into fixed-size chunks."""
    if chunk_size <= 0:
        raise ValueError("edge_chunk_size must be positive")
    return tuple(
        tuple(positive_edges[start : start + chunk_size])
        for start in range(0, len(positive_edges), chunk_size)
    )


def _chunk_core_nodes(edge_chunk: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    """Return chunk core nodes in first-seen edge order."""
    ordered_nodes: list[str] = []
    seen: set[str] = set()
    for node_a, node_b in edge_chunk:
        if node_a not in seen:
            ordered_nodes.append(node_a)
            seen.add(node_a)
        if node_b not in seen:
            ordered_nodes.append(node_b)
            seen.add(node_b)
    return tuple(ordered_nodes)


def _expand_chunk_nodes(
    *,
    graph: nx.Graph,
    edge_chunk: Sequence[tuple[str, str]],
    target_size: int,
    strategy: str,
    rng: random.Random,
) -> tuple[str, ...]:
    """Expand one positive-edge chunk into a node-induced training subgraph."""
    core_nodes = _chunk_core_nodes(edge_chunk)
    if len(core_nodes) >= target_size:
        return core_nodes
    if strategy == "BFS":
        order = _build_bfs_order(graph=graph, seed_nodes=core_nodes, rng=rng)
    elif strategy == "DFS":
        order = _build_dfs_order(graph=graph, seed_nodes=core_nodes, rng=rng)
    else:
        order = _build_random_walk_order(graph=graph, seed_nodes=core_nodes, rng=rng)
    return tuple(order[:target_size])


def _select_seed_edge(
    *,
    positive_edges: Sequence[tuple[str, str]],
    uncovered_edges: set[tuple[str, str]],
    edge_cover_counts: Mapping[tuple[str, str], int],
    rng: random.Random,
) -> tuple[str, str]:
    """Select the next seed edge, preferring uncovered edges first."""
    candidate_edges = list(uncovered_edges) if uncovered_edges else list(positive_edges)
    rng.shuffle(candidate_edges)
    return min(candidate_edges, key=lambda edge: (edge_cover_counts.get(edge, 0), edge))


def _pick_next_node(
    *,
    graph: nx.Graph,
    selected_nodes: Sequence[str],
    candidate_nodes: Sequence[str],
    uncovered_edges: set[tuple[str, str]],
    edge_cover_counts: Mapping[tuple[str, str], int],
    traversal_rank: Mapping[str, int],
    overlap_penalty: float,
    rng: random.Random,
) -> str:
    """Pick the next node using uncovered-edge gain, overlap, and traversal rank."""
    shuffled_candidates = list(candidate_nodes)
    rng.shuffle(shuffled_candidates)
    max_rank = len(traversal_rank) + len(shuffled_candidates) + 1
    selected_set = set(selected_nodes)

    def _score(candidate: str) -> tuple[float, float, int]:
        connecting_edges = [
            _canonical_edge(candidate, node)
            for node in selected_set
            if graph.has_edge(candidate, node)
        ]
        uncovered_gain = float(sum(1 for edge in connecting_edges if edge in uncovered_edges))
        overlap_cost = float(
            sum(
                edge_cover_counts.get(edge, 0)
                for edge in connecting_edges
                if edge not in uncovered_edges
            )
        )
        return (
            uncovered_gain,
            -(overlap_penalty * overlap_cost),
            -traversal_rank.get(candidate, max_rank),
        )

    return max(shuffled_candidates, key=_score)


def sample_training_subgraphs(
    *,
    graph: nx.Graph,
    num_subgraphs: int,
    min_nodes: int,
    max_nodes: int,
    strategy: str,
    seed: int,
) -> list[tuple[str, ...]]:
    """Sample node-induced training subgraphs from a PRING train graph."""
    if num_subgraphs < 0:
        raise ValueError("num_subgraphs must be non-negative")

    normalized_strategy = _normalize_sampling_strategy(strategy)

    rng = random.Random(seed)
    sampled_nodes: list[tuple[str, ...]] = []
    for _ in range(num_subgraphs):
        selected_strategy = _choose_sampling_strategy(strategy=normalized_strategy, rng=rng)
        target_size = _sample_target_size(
            graph=graph,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            rng=rng,
        )
        if selected_strategy == "BFS":
            sampled_nodes.append(_sample_bfs_nodes(graph, target_size, rng))
        elif selected_strategy == "DFS":
            sampled_nodes.append(_sample_dfs_nodes(graph, target_size, rng))
        else:
            sampled_nodes.append(_sample_random_walk_nodes(graph, target_size, rng))
    return sampled_nodes


def sample_topology_evaluation_subgraphs(
    *,
    graph: nx.Graph,
    seed: int,
    strategy: str = "mixed",
    node_sizes: Sequence[int] = TOPOLOGY_EVAL_NODE_SIZES,
    samples_per_size: int = TOPOLOGY_EVAL_SAMPLES_PER_SIZE,
) -> dict[int, list[tuple[str, ...]]]:
    """Sample topology-evaluation-style node buckets for internal validation."""
    if graph.number_of_nodes() < 2:
        raise ValueError("Topology evaluation sampling requires at least two graph nodes")
    if samples_per_size <= 0:
        raise ValueError("samples_per_size must be positive")

    eligible_sizes = [size for size in node_sizes if size <= graph.number_of_nodes()]
    if not eligible_sizes:
        eligible_sizes = [graph.number_of_nodes()]

    sampled_by_size: dict[int, list[tuple[str, ...]]] = {}
    for offset, node_size in enumerate(eligible_sizes):
        sampled_by_size[int(node_size)] = [
            tuple(sorted(nodes))
            for nodes in sample_training_subgraphs(
                graph=graph,
                num_subgraphs=samples_per_size,
                min_nodes=int(node_size),
                max_nodes=int(node_size),
                strategy=strategy,
                seed=seed + offset,
            )
        ]
    return sampled_by_size


def build_internal_validation_plan(
    *,
    graph: nx.Graph,
    sampled_subgraphs: Mapping[int, Sequence[tuple[str, ...]]],
) -> InternalValidationPlan:
    """Build batched internal-validation metadata for all node-size buckets."""
    buckets: list[InternalValidationNodeBucketPlan] = []
    protein_ids: set[str] = set()
    total_subgraphs = 0
    total_pairs = 0

    for node_size in sorted(sampled_subgraphs):
        node_sets = tuple(tuple(nodes) for nodes in sampled_subgraphs[node_size])
        target_subgraphs = tuple(graph.subgraph(nodes).copy() for nodes in node_sets)
        pair_records: list[InternalValidationPairRecord] = []
        for subgraph_index, nodes in enumerate(node_sets):
            protein_ids.update(nodes)
            for index_a, protein_a in enumerate(nodes):
                for index_b in range(index_a + 1, len(nodes)):
                    pair_records.append(
                        InternalValidationPairRecord(
                            subgraph_index=subgraph_index,
                            pair_index_a=index_a,
                            pair_index_b=index_b,
                            protein_a=protein_a,
                            protein_b=nodes[index_b],
                        )
                    )
        buckets.append(
            InternalValidationNodeBucketPlan(
                node_size=int(node_size),
                sampled_subgraphs=node_sets,
                target_subgraphs=target_subgraphs,
                pair_records=tuple(pair_records),
            )
        )
        total_subgraphs += len(node_sets)
        total_pairs += len(pair_records)

    return InternalValidationPlan(
        buckets=tuple(buckets),
        protein_ids=frozenset(protein_ids),
        total_subgraphs=total_subgraphs,
        total_pairs=total_pairs,
    )


def sample_edge_cover_subgraphs(
    *,
    graph: nx.Graph,
    num_subgraphs: int,
    min_nodes: int,
    max_nodes: int,
    strategy: str,
    seed: int,
    overlap_penalty: float = 0.5,
    edge_chunk_size: int | None = None,
) -> EdgeCoverEpochPlan:
    """Plan one epoch of training subgraphs with positive-edge coverage guarantees."""
    if num_subgraphs < 0:
        raise ValueError("num_subgraphs must be non-negative")
    if overlap_penalty < 0.0:
        raise ValueError("overlap_penalty must be non-negative")

    normalized_strategy = _normalize_sampling_strategy(strategy)

    positive_edges = sorted(_graph_positive_edges(graph))
    if not positive_edges:
        fallback_subgraphs = sample_training_subgraphs(
            graph=graph,
            num_subgraphs=num_subgraphs,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            strategy=strategy,
            seed=seed,
        )
        return summarize_edge_cover_epoch(graph=graph, subgraphs=fallback_subgraphs)

    rng = random.Random(seed)
    shuffled_edges = list(positive_edges)
    rng.shuffle(shuffled_edges)
    resolved_edge_chunk_size = (
        _default_edge_chunk_size(max_nodes)
        if edge_chunk_size is None
        else int(edge_chunk_size)
    )
    edge_chunks = _partition_edges(
        positive_edges=shuffled_edges,
        chunk_size=resolved_edge_chunk_size,
    )
    sampled_subgraphs = [
        _expand_chunk_nodes(
            graph=graph,
            edge_chunk=edge_chunk,
            target_size=_sample_target_size(
                graph=graph,
                min_nodes=min_nodes,
                max_nodes=max_nodes,
                rng=rng,
            ),
            strategy=_choose_sampling_strategy(strategy=normalized_strategy, rng=rng),
            rng=rng,
        )
        for edge_chunk in edge_chunks
    ]

    return summarize_edge_cover_epoch(graph=graph, subgraphs=sampled_subgraphs)


def _subgraph_pair_tuples(
    *,
    graph: nx.Graph,
    nodes: tuple[str, ...],
    negative_lookup: ExplicitNegativePairLookup | None = None,
    negative_ratio: int = 1,
    rng: random.Random | None = None,
) -> list[tuple[int, int, str, str, float, float, float]]:
    """Return all upper-triangle node pairs with topology labels and BCE masks."""
    if negative_ratio < 0:
        raise ValueError("negative_ratio must be non-negative")

    pair_rows: list[tuple[int, int, str, str, float, float, float]] = []
    positive_pairs: set[tuple[str, str]] = set()
    negative_candidates: list[tuple[str, str]] = []
    for index_a, protein_a in enumerate(nodes):
        for index_b in range(index_a + 1, len(nodes)):
            protein_b = nodes[index_b]
            pair = _canonical_edge(protein_a, protein_b)
            topology_label = 1.0 if graph.has_edge(protein_a, protein_b) else 0.0
            if topology_label > 0.0:
                positive_pairs.add(pair)
            elif negative_lookup is not None and protein_b in negative_lookup.partners_by_node.get(
                protein_a, frozenset()
            ):
                negative_candidates.append(pair)
            pair_rows.append((index_a, index_b, protein_a, protein_b, topology_label, 0.0, 0.0))
    if not pair_rows:
        raise ValueError("A sampled subgraph must contain at least one node pair")

    if negative_lookup is None:
        return [
            (index_a, index_b, protein_a, protein_b, topology_label, topology_label, 1.0)
            for index_a, index_b, protein_a, protein_b, topology_label, _, _ in pair_rows
        ]

    sampled_negative_pairs: set[tuple[str, str]] = set()
    desired_negative_count = min(len(negative_candidates), len(positive_pairs) * negative_ratio)
    if desired_negative_count > 0:
        candidate_pool = sorted(negative_candidates)
        sampler = rng or random.Random(0)
        sampled_negative_pairs = set(sampler.sample(candidate_pool, desired_negative_count))

    return [
        (
            index_a,
            index_b,
            protein_a,
            protein_b,
            topology_label,
            topology_label,
            1.0,
        )
        if topology_label > 0.0
        else (
            index_a,
            index_b,
            protein_a,
            protein_b,
            topology_label,
            0.0,
            1.0 if _canonical_edge(protein_a, protein_b) in sampled_negative_pairs else 0.0,
        )
        for index_a, index_b, protein_a, protein_b, topology_label, _, _ in pair_rows
    ]


def iter_subgraph_pair_chunks(
    *,
    graph: nx.Graph,
    nodes: Sequence[str],
    cache_dir: Path,
    embedding_index: Mapping[str, str],
    input_dim: int,
    max_sequence_length: int,
    pair_batch_size: int,
    embedding_repository: EmbeddingRepository | None = None,
    negative_lookup: ExplicitNegativePairLookup | None = None,
    negative_ratio: int = 1,
    seed: int | None = None,
) -> Iterator[SubgraphPairChunk]:
    """Yield all within-subgraph pairs as padded mini-batches."""
    if pair_batch_size <= 0:
        raise ValueError("pair_batch_size must be positive")
    if negative_ratio < 0:
        raise ValueError("negative_ratio must be non-negative")

    node_tuple = tuple(nodes)
    if len(node_tuple) < 2:
        raise ValueError("A sampled subgraph must contain at least two nodes")

    if embedding_repository is not None:
        embeddings = embedding_repository.get_many(node_tuple)
    else:
        embeddings = {
            protein_id: load_cached_embedding(
                cache_dir=cache_dir,
                index=embedding_index,
                protein_id=protein_id,
                expected_input_dim=input_dim,
                max_sequence_length=max_sequence_length,
            )
            for protein_id in node_tuple
        }
    pair_rows = _subgraph_pair_tuples(
        graph=graph,
        nodes=node_tuple,
        negative_lookup=negative_lookup,
        negative_ratio=negative_ratio,
        rng=random.Random(seed) if seed is not None else None,
    )

    for chunk_start in range(0, len(pair_rows), pair_batch_size):
        rows = pair_rows[chunk_start : chunk_start + pair_batch_size]
        emb_a = pad_sequence(
            [embeddings[protein_a] for _, _, protein_a, _, _, _, _ in rows],
            batch_first=True,
        )
        emb_b = pad_sequence(
            [embeddings[protein_b] for _, _, _, protein_b, _, _, _ in rows],
            batch_first=True,
        )
        yield SubgraphPairChunk(
            nodes=node_tuple,
            emb_a=emb_a,
            emb_b=emb_b,
            len_a=torch.tensor(
                [embeddings[protein_a].size(0) for _, _, protein_a, _, _, _, _ in rows],
                dtype=torch.long,
            ),
            len_b=torch.tensor(
                [embeddings[protein_b].size(0) for _, _, _, protein_b, _, _, _ in rows],
                dtype=torch.long,
            ),
            label=torch.tensor(
                [topology_label for _, _, _, _, topology_label, _, _ in rows],
                dtype=torch.float32,
            ),
            pair_index_a=torch.tensor(
                [index_a for index_a, _, _, _, _, _, _ in rows],
                dtype=torch.long,
            ),
            pair_index_b=torch.tensor(
                [index_b for _, index_b, _, _, _, _, _ in rows],
                dtype=torch.long,
            ),
            bce_label=torch.tensor(
                [bce_label for _, _, _, _, _, bce_label, _ in rows],
                dtype=torch.float32,
            ),
            bce_mask=torch.tensor(
                [bce_mask for _, _, _, _, _, _, bce_mask in rows],
                dtype=torch.float32,
            ),
        )
