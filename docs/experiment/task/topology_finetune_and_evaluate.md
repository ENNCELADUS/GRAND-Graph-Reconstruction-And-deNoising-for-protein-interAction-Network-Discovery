# Topology Fine-tune and Topology Evaluate Task Setup

This note explains the `topology_finetune`, `evaluate`, and
`topology_evaluate` setup implemented under `src/pipeline/stages/`, using
`data/PRING/human/BFS` as the concrete PRING Human BFS example. The code reads
the same file names from
`data_config.benchmark.processed_dir`; older YAMLs may point to
`data/PRING/species_processed_data/human/BFS`, but the stage contract is path
relative and does not depend on that parent directory name.

## File Inputs

For `processed_dir = data/PRING/human/BFS`, the relevant files are:

| File | Used by | Role |
| --- | --- | --- |
| `human_BFS_split.pkl` | `topology_finetune` | Provides the `train` protein node set used to build the train and internal-validation supervision graphs. |
| `human_train_ppi.txt` | base dataloader, fallback supervision | Labeled train pairs for standard pairwise training. Used by `topology_finetune` only if `supervision_train_dataset` is absent. |
| `human_train_ppi_ratio5_exclusive.txt` | `topology_finetune` | Preferred train supervision file. Positive rows define train graph edges; negative rows define explicit BCE negatives. |
| `human_val_ppi.txt` | base dataloader | Labeled validation pairs used for pairwise validation loss and validation AUPRC during fine-tuning. |
| `human_val_ppi_ratio5_exclusive.txt` | `topology_finetune` | Preferred internal-validation supervision file. Positive rows define the held-out topology target graph on train-split nodes; negative rows are ignored for the topology graph. |
| `human_test_ppi.txt` | `evaluate` | Labeled pairwise test set for binary PPI metrics. |
| `all_test_ppi.txt` | `topology_evaluate` | Candidate universe for graph reconstruction. The stage predicts one binary label per row and reconstructs the predicted test graph from rows predicted positive. |
| `human_test_graph.pkl` | `topology_evaluate` | Ground-truth test graph for topology metrics. |
| `test_sampled_nodes.pkl` | `topology_evaluate` | PRING node buckets for graph-level metric sampling. |

All pair files are tab-separated records:

```text
protein_a<TAB>protein_b<TAB>label
```

`label > 0` is positive; `label <= 0` is negative. The pair dataset parser can
default missing labels to positive, but the PRING split files used here carry an
explicit third column.

## Topology Fine-tune Train Task

`topology_finetune` trains a pairwise scoring model with graph-shaped
supervision. It does not use the PRING test files.

The train graph is built as follows:

1. Load the train protein IDs from `human_BFS_split.pkl`.
2. Read `topology_finetune.supervision_train_dataset`, normally
   `human_train_ppi_ratio5_exclusive.txt`.
3. Add every train-split protein as a graph node.
4. Add only positive in-split rows as train graph edges.
5. Read negative rows from the same train supervision file into an explicit
   negative-pair lookup for BCE supervision.

Each epoch creates an edge-cover subgraph plan from this train graph. Positive
train edges are shuffled, partitioned, expanded into node-induced subgraphs by
the configured sampling strategy (`BFS`, `DFS`, `RANDOM_WALK`, or `mixed`), and
paired with up to `bce_negative_ratio` explicit negatives per assigned positive
edge. The model then scores all upper-triangle protein pairs inside each sampled
subgraph.

The train objective has two parts:

1. Masked BCE over assigned supervised pairs. Assigned positives have target
   `1`; assigned explicit negatives have target `0`; unassigned pairs do not
   contribute to BCE when an assignment mask exists.
2. Differentiable topology losses over the whole sampled subgraph. The model
   probabilities form a soft predicted adjacency, and the positive-edge labels
   form the target adjacency.

The topology loss terms are:

| Term | Meaning |
| --- | --- |
| graph similarity loss | `1 - graph_sim`, implemented as absolute adjacency difference divided by predicted plus target edge mass. |
| relative density loss | penalty for predicted density differing from target density, using the configured `rd_loss_form`. |
| degree MMD | soft histogram MMD between predicted and target degree distributions. |
| clustering MMD | soft histogram MMD between predicted and target clustering-coefficient distributions, when enabled. |

The weighted topology objective is:

```text
alpha * graph_similarity_loss
+ beta * relative_density_loss
+ gamma * degree_mmd
+ delta * clustering_mmd
```

The final per-subgraph train loss is BCE plus grouped topology objectives. The
configured topology-loss schedule can set the topology scale to `0` during
warmup, in which case the epoch trains on supervised BCE only.

## Topology Fine-tune Validation Task

Fine-tune validation has two surfaces:

1. Pairwise validation: run the model on `human_val_ppi.txt`, compute BCE loss
   and validation AUPRC through the generic `Evaluator`.
2. Internal topology validation: build a validation target graph from positive
   rows in `human_val_ppi_ratio5_exclusive.txt`, restricted to the train protein
   node set from `human_BFS_split.pkl`; sample fixed topology-evaluation-style
   node buckets from that validation graph; threshold model probabilities at
   `0.5`; reconstruct hard predicted subgraphs; compute graph metrics against
   the validation target subgraphs.

The validation topology loss used for monitoring is a hard-metric penalty:

```text
alpha * (1 - graph_sim)
+ beta * (relative_density - 1)^2
+ gamma * deg_dist_mmd
+ delta * cc_mmd
```

`topology_finetune.monitor_metric` selects the checkpoint criterion. Supported
monitor names include `val_loss`, `val_topology_loss`,
`internal_val_graph_sim`, `internal_val_relative_density`, and `val_auprc`.

The stage writes:

```text
logs/{model}/topology_finetune/{run_id}/topology_finetune_step.csv
logs/{model}/topology_finetune/{run_id}/topology_finetune_metrics.json
models/{model}/topology_finetune/{run_id}/best_model.pth
```

## Pairwise Test Task

When `run_config.stages` includes `evaluate`, the pipeline loads the best
available checkpoint, normally from `topology_finetune` when that stage ran
before evaluation, and evaluates binary pair prediction on `human_test_ppi.txt`.

The input to the model is one row at a time after batching:

```text
(embedding(protein_a), embedding(protein_b), label)
```

The model outputs logits. The evaluator converts logits to probabilities with
`sigmoid(logit)` and uses the fixed PRING threshold `0.5` for hard predictions.
It writes:

```text
logs/{model}/evaluate/{run_id}/evaluate.csv
```

with columns:

```text
split, auroc, auprc, accuracy, sensitivity, specificity, precision, recall, f1, mcc
```

AUROC and AUPRC are computed from probabilities. Accuracy, sensitivity,
specificity, precision, recall, F1, and MCC are computed from hard predictions
at threshold `0.5`. If a split does not contain both classes, AUC-like metrics
return `0.0`.

## Topology Test Task

When `run_config.stages` includes `topology_evaluate`, the pipeline evaluates
graph reconstruction, not labeled pair classification.

The stage reads:

```text
data/PRING/human/BFS/all_test_ppi.txt
data/PRING/human/BFS/human_test_graph.pkl
data/PRING/human/BFS/test_sampled_nodes.pkl
```

`all_test_ppi.txt` is used as the candidate universe. The stage runs the model
on every candidate row, thresholds `sigmoid(logit) >= 0.5`, writes the predicted
labels to `all_test_ppi_pred.txt`, and reconstructs the predicted graph from
candidate pairs with predicted label `1`. The labels in `all_test_ppi.txt` are
not the topology metric source of truth; `human_test_graph.pkl` is.

The reconstructed graph is compared with `human_test_graph.pkl` on the sampled
node buckets in `test_sampled_nodes.pkl`. For each node bucket, the stage forms
the induced predicted and ground-truth subgraphs and computes:

| Metric | Direction | Definition |
| --- | --- | --- |
| `graph_sim` | higher is better | `1 - sum(abs(A_pred - A_gt)) / (sum(A_pred) + sum(A_gt))`; returns `1.0` when both graphs are empty. |
| `relative_density` | closer to `1` is better | `density(pred_graph) / density(gt_graph)`. |
| `deg_dist_mmd` | lower is better | Paper-normalized MMD ratio between predicted and ground-truth degree histograms. |
| `cc_mmd` | lower is better | Paper-normalized MMD ratio between clustering-coefficient histograms. |
| `laplacian_eigen_mmd` | lower is better | Paper-normalized MMD ratio between normalized-Laplacian spectral histograms. |

`graph_sim` and `relative_density` are averaged over all sampled subgraphs. The
MMD metrics are computed per node-size bucket and then averaged across buckets.

The stage writes:

```text
logs/{model}/topology_evaluate/{run_id}/all_test_ppi_pred.txt
logs/{model}/topology_evaluate/{run_id}/topology_metrics.json
logs/{model}/topology_evaluate/{run_id}/topology_metrics.csv
logs/{model}/topology_evaluate/{run_id}/graph_eval_results.pkl
```

## BFS Example Sanity Check

The `data/PRING/human/BFS` files checked on 2026-06-03 have these sizes:

| File | Rows | Positives | Negatives | Proteins | Self-pair rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| `human_train_ppi.txt` | 85,824 | 42,880 | 42,944 | 7,869 | 4,680 |
| `human_train_ppi_ratio5_exclusive.txt` | 257,280 | 42,880 | 214,400 | 9,964 | 4,680 |
| `human_val_ppi.txt` | 21,456 | 10,760 | 10,696 | 6,608 | 1,198 |
| `human_val_ppi_ratio5_exclusive.txt` | 64,560 | 10,760 | 53,800 | 9,964 | 1,198 |
| `human_test_ppi.txt` | 64,038 | 32,019 | 32,019 | 2,018 | 1,891 |
| `all_test_ppi.txt` | 2,037,171 | 32,019 | 2,005,152 | 2,018 | 2,018 |

The split and graph pickles in the same directory contain:

| File | Contents |
| --- | --- |
| `human_BFS_split.pkl` | `train`: 8,072 proteins; `test`: 2,018 proteins. |
| `human_train_graph.pkl` | 8,072 nodes and 53,640 edges. |
| `human_test_graph.pkl` | 2,018 nodes and 32,019 edges. |
| `test_sampled_nodes.pkl` | 50 sampled node sets for each size `20, 40, ..., 200`. |

The implementation in `src/pipeline/stages/topology_evaluate.py` forwards every
row in `all_test_ppi.txt` through the topology inference loader. If a prepared
dataset contains self-pair rows, they are part of the candidate file unless
removed before launch.
