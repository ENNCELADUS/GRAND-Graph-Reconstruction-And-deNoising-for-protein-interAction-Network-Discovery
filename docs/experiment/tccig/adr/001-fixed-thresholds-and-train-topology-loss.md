# ADR 001: Fixed TCCIG Thresholds

## Status

Superseded by ADR 002 for the training objective. The fixed-threshold decision remains active.

## Context

The previous TCCIG S2GAE path mixed two separate decisions:

- scorer probabilities were thresholded to build `G_pairwise`, the noisy input graph;
- refined S2GAE probabilities were thresholded to produce the final hard graph.

Validation-time dynamic output-threshold changes made epoch time, edge counts, and topology
metrics difficult to interpret.

## Decision

TCCIG now uses two independent fixed rules:

- `graph_selection.pairwise_input_threshold` freezes the scorer-only input threshold at epoch 0.
  The current config uses validation scorer labels to choose the lowest threshold with
  `precision >= 0.8`.
- `graph_selection.refined_output_rule` thresholds refined S2GAE probabilities independently.
  The current config uses fixed `p_refined >= 0.5`.

Dynamic validation calibration and validation-selected refined thresholds are removed from the
standalone TCCIG pipeline.

S2GAE training no longer treats every train pair as one exhaustive train-pair
loss surface. ADR 002 defines the current sampled scorer-error edge objective.
The base loss remains BCE plus a residual anchor on sampled edge targets:

```text
BCE + residual_weight * mean(delta^2)
```

This rewrite rejects `refiner.topology_loss.enabled: true`; hard NetworkX metrics remain
validation/test reporting metrics and checkpoint-monitoring signals.

## Consequences

Input graph density is controlled by the frozen scorer threshold and no longer swings with
validation calibration. Refined output graph density is controlled by the independent fixed
threshold, so scorer input selection and S2GAE output evaluation can be audited separately.

Train-time topology loss is not part of the current rewrite. Degree and clustering MMD remain
reporting and monitoring metrics only.
