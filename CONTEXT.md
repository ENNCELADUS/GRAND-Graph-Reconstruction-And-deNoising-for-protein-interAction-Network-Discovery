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
