# GRAND Domain Context

## Glossary

### PRING candidate universe

The set of protein pairs that a PRING split allows the model pipeline to score.
For topology test, `all_test_ppi.txt` is the candidate universe; its labels are
not a source of topology truth.

### Pairwise-generated graph

The weighted graph built from pairwise classifier scores over a split's
candidate universe. This graph is a model input for TCCIG, including pairwise
confidence weights on selected edges. It must be derived from pairwise
predictions, not from PRING true labels or ground-truth graph pickles.

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

### Scorer error quadrant

The four-way partition of train candidate pairs induced by frozen pairwise
scores, the frozen pairwise input graph threshold, and true train labels. `FP`
and `FN` are scorer-error hard cases; `TP` and `TN` are scorer-correct easy
cases.

### Sampled edge target objective

The TCCIG S2GAE training objective that prioritizes all scorer-error hard cases
and samples scorer-correct `TP`/`TN` pairs as calibration anchors. It trains the
refiner to correct `G_pairwise` without letting the edge currently being
predicted serve as an encoder input edge.

### Pairwise input graph threshold

The scorer-only threshold used to construct the noisy `G_pairwise` input graph
for S2GAE. It is frozen at epoch 0 from validation scorer outputs, currently by
selecting the lowest threshold that reaches `precision >= 0.8`, and then reused
for train, validation, pairwise test, and topology test input-graph assembly.
It is not a decision rule for refined S2GAE outputs.

### Refined output threshold

The output-side hard decision rule that converts S2GAE refined probabilities
into a final predicted graph. It is independent from the pairwise input graph
threshold; the current TCCIG pipeline uses fixed `p_refined >= 0.5` for
validation, pairwise test, and topology test. Per-node top-k and global top-M
rules are forbidden because they impose non-biological degree or edge-count
budgets.

### Validation topology candidate universe

The set of protein pairs materialized inside fixed validation topology buckets.
For TCCIG, these are all non-self pairs inside PRING-style validation node
buckets sampled from the validation true topology target, not only the ratio5
validation supervision rows.
