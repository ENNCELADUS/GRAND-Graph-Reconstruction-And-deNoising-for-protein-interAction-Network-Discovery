# GRAND Context Glossary

## TCCIG

Topology Constrained Conditional Interactome Generator. A train-graph-prior-conditioned
retrieval and reranking model that reconstructs a PRING candidate interactome from
protein-intrinsic embeddings without receiving target topology in the model forward pass.
_Avoid_: dense all-pairs graph generator, pairwise-only classifier

## Graph Assembly

The TCCIG evaluation procedure: score PRING candidate protein pairs, select a sparse
edge set with the official assembly rule, and evaluate the resulting hard graph with
the existing PRING topology metrics.

## Graph Assembly Evaluation

The TCCIG `evaluate` mode that scores the PRING topology candidate universe
(`all_test_ppi.txt`) with graph-context probabilities, computes ranking metrics from
those probabilities, and computes hard binary metrics from the same hybrid
validation-density/degree-cap assembly rule used by topology evaluation.

## Internal Validation Candidate Universe

The deduplicated set of unordered protein pairs induced by all sampled internal
validation subgraphs. TCCIG internal validation uses this universe for
reconstruction-oriented diagnostics and combines pair ranking with topology metrics
through the composite monitor.

## Edge Probability

A calibrated TCCIG candidate-pair score after sigmoid. Edge probabilities are used for
ranking candidate pairs and calibration diagnostics; they are not by themselves the
definition of the predicted graph.

## Edge Probability Saturation

A TCCIG failure mode where sigmoid edge probabilities collapse near `1.0` across the
candidate universe, making fixed-threshold diagnostics useless and reducing useful
ranking and calibration signal.

## Fixed Threshold Diagnostic

An observability-only view of how many TCCIG edge probabilities cross a fixed threshold,
usually `0.5`. It helps detect score saturation but does not define the primary TCCIG
topology-monitor graph.

## Graph Density Prior

The train-graph edge density used to initialize the TCCIG density-bias head. It is
computed from positive training edges over the full train-node pair universe and is
distinct from the supervised BCE positive rate induced by negative sampling.

## Debug Assembly

A non-official TCCIG topology-evaluation assembly used to diagnose budget choice under
the same ranked edge probabilities. Debug assemblies can compare model `m_hat`,
validation-density, and oracle-test-density budgets, but they do not redefine the
official Graph Assembly decision rule.

## Validation-Calibrated Pairwise Threshold

A TCCIG-only pairwise classification threshold selected from validation-set edge
probabilities and labels. It defines pairwise hard metrics for diagnostic
classification views, but it does not define the assembled TCCIG graph.

## Train-Only Topology Teacher

An auxiliary graph autoencoder trained only on the training PPI graph. It provides
offline structural embeddings, degree targets, edge priors, and hard-negative seeds
for TCCIG student training and is not used during validation or test graph assembly.

## MGAE Teacher

The first train-only graph-prior teacher backend for TCCIG. It masks positive
training edges, encodes the visible graph, samples non-edge candidates, and
reconstructs masked positives versus sampled negatives.

## Candidate Universe

The set of protein pairs considered for reconstruction. TCCIG trains on sampled
train-graph candidates, validates on a PRING-like reconstruction universe, and
evaluates on PRING `all_test_ppi.txt`.

## Retrieval Candidate

A candidate edge proposed by exact top-k retrieval from feature-derived query/key and
residue-factorized embeddings before reranking.
_Avoid_: ANN edge, dense pair

## Hybrid Assembly

The official TCCIG graph assembly rule that uses a validation-density global edge
budget plus predicted per-node degree caps.
_Avoid_: fixed threshold, learned `m_hat` rule
