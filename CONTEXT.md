# GRAND Context Glossary

## TCCIG

Topology Constrained Conditional Interactome Generator. A feature-only graph generator
that consumes cached protein embeddings for a protein set and predicts a candidate
interactome without receiving target topology in the model forward pass.

## Graph Assembly

The TCCIG evaluation procedure: score candidate protein pairs, predict an edge budget
`m_hat`, select the top-`m_hat` candidate edges, and evaluate the resulting hard graph
with the existing PRING topology metrics.

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
