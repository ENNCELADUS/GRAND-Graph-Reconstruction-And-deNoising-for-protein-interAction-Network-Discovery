# Topology Finetune: Current Loss Issues

## Scope

This note summarizes the current training-loss behavior observed in `logs/v3/topology_finetune/20260416_214619` and the recommended next fixes.

## Observations

| Epoch | BCE | GS Loss | RD Loss | Deg MMD | Clus MMD | Total |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2.197 | 0.973 | 128.53 | 0.226 | 0.290 | 6.141 |
| 2 | 2.310 | 0.973 | 10.72 | 0.210 | 0.280 | 2.719 |
| 3 | 2.299 | 0.973 | 10.31 | 0.204 | 0.277 | 2.696 |
| 4 | 2.269 | 0.973 | 10.59 | 0.210 | 0.281 | 2.674 |
| 5 | 2.263 | 0.973 | 10.57 | 0.210 | 0.282 | 2.668 |

Additional validation signals:

- `internal_val_graph_sim` stays at `0.0793`
- `internal_val_relative_density` stays around `47.18`

## What This Means

- The predicted graph remains massively over-dense. The model is still assigning overly high probabilities to too many pairs.
- `RD loss` is numerically much larger than the other topology terms and dominates the topology gradient.
- `GS loss` is stuck near its worst value and provides little useful gradient in this regime.
- BCE is trying to separate positives from negatives, while RD is mostly pushing global probability mass down. Those directions conflict, so training stalls.

## Main Issues

1. `RD` uses a squared density ratio, so over-dense predictions explode the loss scale.
2. The topology objective is a raw weighted sum of terms with very different magnitudes.
3. `scratch` initialization likely starts too dense for a sparse graph task.
4. The current hard topology monitor is too blunt to show soft progress early.
5. GradNorm on the raw four topology terms is not the first fix; it will react to bad scales rather than solve them.

## Recommended Fixes

### Priority 1: Make warmup actually effective

- Keep topology loss off for the first few epochs, then ramp it in.
- Verify runtime logs include `Topology Loss Scale` and that early epochs are truly `0.0`.

### Priority 2: Replace RD with a stable formulation

Use log-ratio instead of squared ratio:

```python
log_ratio = torch.log((pred_density + eps) / (target_density + eps))
rd_loss = torch.nn.functional.smooth_l1_loss(
    log_ratio,
    torch.zeros_like(log_ratio),
)
```

Why:

- compresses extreme density mismatch
- stays symmetric for over-/under-dense cases
- avoids RD overwhelming the whole topology objective

### Priority 3: Normalize topology terms before weighting

Use detached EMA normalization per term:

```python
ema_i = rho * ema_i + (1 - rho) * raw_i.detach()
norm_i = raw_i / (ema_i + eps)
norm_i = norm_i.clamp(max=5.0)
```

Apply weights to normalized losses, not raw losses.

### Priority 4: Use sparse-prior bias init for scratch mode

- Initialize the final classifier bias to `logit(train_edge_density)`.
- This avoids the default `sigmoid(0)=0.5` behavior that makes early predictions far too dense.

### Priority 5: Use GradNorm only after normalization

If dynamic weighting is still needed, apply GradNorm on grouped objectives:

- `L_bce`
- `L_density`
- `L_shape = GS + Deg + Clus`

Do not apply GradNorm directly to the raw four topology terms. It should be a secondary stabilizer, not the primary fix.

## Recommended Order

1. Ensure topology warmup/ramp is active in the actual run.
2. Change RD to log-ratio + SmoothL1.
3. Add EMA normalization for topology terms.
4. Add sparse-prior bias init for scratch runs.
5. Only then consider grouped GradNorm.

## Implementation Checklist

- [x] Verify topology warmup/ramp is active and logged
- [x] Replace RD with log-ratio + SmoothL1
- [x] Add EMA normalization for topology terms
- [x] Add sparse-prior bias init for scratch mode
- [x] Add grouped GradNorm dynamic task weighting

## Success Criteria

- `internal_val_relative_density` drops sharply toward `1`
- `GS loss` starts moving off the `~0.973` plateau
- topology contribution is no longer dominated by RD
- `internal_val_graph_sim` and soft topology metrics begin to improve after warmup
