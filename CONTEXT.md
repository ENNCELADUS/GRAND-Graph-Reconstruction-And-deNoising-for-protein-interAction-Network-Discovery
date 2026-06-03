# GRAND Domain Context

## Glossary

### PRING candidate universe

The set of protein pairs that a PRING split allows the model pipeline to score.
For topology test, `all_test_ppi.txt` is the candidate universe; its labels are
not a source of topology truth.

### Pairwise-generated graph

The graph built from pairwise classifier scores over a split's candidate
universe. This graph is a model input for TCCIG. It must be derived from
pairwise predictions, not from PRING true labels or ground-truth graph pickles.

### TCCIG pairwise scorer

The model boundary that scores label-free PRING candidate pairs before TCCIG
graph construction. It produces the scores used to build a pairwise-generated
graph and must not receive topology ground truth or labels from topology-test
candidate files.

### S2GAE residual denoiser

The second TCCIG model boundary after the pairwise scorer. It receives protein
features, a pairwise-generated graph, and label-free candidate pairs, then
learns residual score corrections against train/validation topology targets.
At test time it must only refine pairwise-generated inputs and must not receive
test topology truth.

### True topology target

The PRING labels or graph used as supervision or evaluation truth. Train labels
may define loss targets, validation truth may select checkpoints and graph
decision rules, and test truth may compute metrics. True topology targets are
not model input graphs.

### Refined graph decision rule

The validation-selected rule that converts refined pair scores into a hard
predicted graph. Supported rule families are probability threshold, per-node
top-k, and global top-M. Test evaluation must reuse the validation-selected
rule without reselecting on test data.

### Validation topology candidate universe

The set of protein pairs materialized inside fixed validation topology buckets.
For TCCIG, these are all non-self pairs inside PRING-style validation node
buckets sampled from the validation true topology target, not only the ratio5
validation supervision rows.
