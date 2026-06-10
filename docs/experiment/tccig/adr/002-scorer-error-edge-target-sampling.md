# ADR 002: Scorer-Error Edge Target Sampling

## Status

Accepted.

## Context

The first standalone TCCIG S2GAE training contract treated the available PRING
train rows as one exhaustive train-pair loss surface. That was simple, but it
did not fully reflect the S2GAE masked-edge principle: the edge currently being
predicted should not also be available as a message-passing edge in the encoder
input.

TCCIG does not start from a true observed graph. It starts from `G_pairwise`, a
noisy graph produced by a frozen pairwise scorer and a frozen pairwise input
threshold. The refiner should therefore focus on the scorer's errors while
keeping enough easy scorer-correct pairs to preserve calibration.

## Decision

TCCIG training defines four scorer-error quadrants from frozen scorer
probabilities, frozen `tau_pair`, and train labels:

```text
TP = s_ij >= tau_pair and y_ij = 1
FP = s_ij >= tau_pair and y_ij = 0
FN = s_ij <  tau_pair and y_ij = 1
TN = s_ij <  tau_pair and y_ij = 0
```

Each epoch uses all hard scorer errors and sampled easy anchors:

```text
epoch_targets = all FP + all FN + sampled TP + sampled TN
```

The default training distribution is `hard_fraction = 0.7` and
`easy_anchor_fraction = 0.3`. TP and TN anchors split the anchor budget evenly
by default, with either side filling any shortage from the other side when
needed.

For each optimizer step, the model samples an edge target batch. Edges in that
batch that already exist in `G_pairwise` are removed from the encoder input for
that step:

```text
G_input_step = G_pairwise - (FP_batch union TP_batch)
```

The encoder runs on the masked full graph. The decoder and loss run only on the
sampled edge targets. The objective remains BCE on the sampled labels plus the
residual anchor:

```text
BCEWithLogits(l_refined, y)
  + residual_weight * mean(delta^2)
```

This is not a direct reproduction of vanilla S2GAE. It transfers the masked-edge
principle to a scorer-generated noisy graph.

## Consequences

The refiner is trained as a scorer-error-focused residual denoiser rather than a
full train-pair BCE model. FP and FN pairs drive correction; TP and TN anchors
prevent probability calibration drift and overcorrection.

Training becomes edge-batched. The encoder still receives full-graph context,
but the supervised loss is computed only on the sampled edge target batch.
Validation and test do not use training masks; they refine pairwise-generated
inputs and keep the fixed refined output threshold.
