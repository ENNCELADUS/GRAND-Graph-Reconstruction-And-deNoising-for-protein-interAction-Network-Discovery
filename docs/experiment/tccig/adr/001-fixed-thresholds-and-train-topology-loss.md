# ADR 001: Fixed TCCIG Thresholds and Optional Train Topology Loss

## Status

Accepted.

## Context

The previous TCCIG S2GAE path mixed two separate decisions:

- scorer probabilities were thresholded to build `G_pairwise`, the noisy input graph;
- refined S2GAE probabilities were thresholded to produce the final hard graph.

Validation-time dynamic threshold and logit-bias sweeps changed the output graph rule during
training, which made epoch time, edge counts, and topology metrics difficult to interpret.

## Decision

TCCIG now uses two independent fixed rules:

- `graph_selection.pairwise_input_threshold` freezes the scorer-only input threshold at epoch 0.
  The current config uses validation scorer labels to choose the lowest threshold with
  `precision >= 0.8`.
- `graph_selection.refined_output_rule` thresholds refined S2GAE probabilities independently.
  The current config uses fixed `p_refined >= 0.5`.

Dynamic validation calibration, logit-bias sweeps, and validation-selected refined thresholds are
removed from the standalone TCCIG pipeline.

S2GAE training uses supervised BCE plus a residual anchor in the current
`03_fixed_threshold` config:

```text
BCE + residual_weight * mean(delta^2)
```

The code also supports an optional train-only soft topology loss behind
`refiner.topology_loss.enabled`:

```text
BCE + residual_weight * mean(delta^2)
  + topology_weight * (alpha * soft_graph_similarity + beta * soft_relative_density)
```

The current config sets `refiner.topology_loss.enabled: false`. When it is enabled, degree and
clustering topology terms are still intentionally zero in backward. Hard NetworkX metrics remain
validation/test reporting metrics and checkpoint-monitoring signals.

## Consequences

Input graph density is controlled by the frozen scorer threshold and no longer swings with
validation calibration. Refined output graph density is controlled by the independent fixed
threshold, so scorer input selection and S2GAE output evaluation can be audited separately.

The GS/RD-only topology loss remains a supported stabilization option, not the default training
objective for the fixed-threshold run. Degree and clustering MMD can be added later after the
fixed-threshold pipeline runs reliably.
