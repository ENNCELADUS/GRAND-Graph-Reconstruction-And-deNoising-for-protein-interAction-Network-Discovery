# GRAND Context Glossary

## TCCIG

Topology Constrained Conditional Interactome Generator. A feature-only graph generator
that consumes cached protein embeddings for a protein set and predicts a candidate
interactome without receiving target topology in the model forward pass.

## Graph Assembly

The TCCIG evaluation procedure: score candidate protein pairs, predict an edge budget
`m_hat`, select the top-`m_hat` candidate edges, and evaluate the resulting hard graph
with the existing PRING topology metrics.

## Graph Assembly Evaluation

The TCCIG `evaluate` mode that scores the PRING topology candidate universe
(`all_test_ppi.txt`) with graph-context probabilities, computes ranking metrics from
those probabilities, and computes hard binary metrics from the same top-`m_hat`
assembly rule used by topology evaluation.

## Internal Validation Candidate Universe

The deduplicated set of unordered protein pairs induced by all sampled internal
validation subgraphs. TCCIG internal validation assembles one validation-wide
top-`m_hat` graph over this universe, then projects the hard decisions back onto
the sampled subgraphs for topology metrics.

## Edge Probability

A calibrated TCCIG candidate-pair score after sigmoid. Edge probabilities are used for
ranking candidate pairs and calibration diagnostics; they are not by themselves the
definition of the predicted graph.

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

An auxiliary graph autoencoder trained only on training subgraphs. It provides soft
candidate-edge targets for distillation during TCCIG student training and is not used
during validation or test graph assembly.

## MGAE Teacher

The first train-only teacher backend for TCCIG. It masks positive training edges,
encodes the visible graph, samples true-negative edges, and reconstructs masked
positive edges versus sampled negatives.

## Candidate Universe

The set of candidate protein pairs scored by TCCIG. Version 1 uses all unordered
within-subgraph pairs for training and PRING `all_test_ppi.txt` pairs for topology
evaluation.
