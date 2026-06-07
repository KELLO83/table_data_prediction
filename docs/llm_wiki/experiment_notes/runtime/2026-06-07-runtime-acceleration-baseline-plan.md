# Experiment Note: Runtime Acceleration Baseline Plan

## Run Metadata

- Date: 2026-06-07
- Command: not run yet
- Config: planned runtime ablation matrix
- Dataset: superconductivity first, then categorical-heavy and large numeric datasets
- Model: `ft_transformer`, `tab_transformer`
- Seed: 42 unless otherwise noted
- Device: CUDA when available
- Results row: pending

## Planned Matrix

| Setting | Values |
|---|---|
| SDPA | on, off |
| AMP | on, off |
| AMP dtype | `float16` first; `bfloat16` only if stability requires it |
| Compile | auto, forced on, off |
| Compile mode | `reduce-overhead` first |
| Matmul precision | `high`, `highest` control |
| Batch size | current default, one larger CUDA-safe value |

## Metrics

| Metric | Required |
|---|---|
| RMSE | yes |
| MAE | yes |
| WAPE | yes |
| Train time sec | yes |
| Predict time sec | yes |
| SDPA backend log | yes |
| Compile report | yes |

## Interpretation

The current code supports SDPA patching, AMP mixed precision, compile gating,
and SDPA backend logging for direct PyTorch tabular transformer wrappers. The
next evidence step is not more refactoring; it is a controlled runtime ablation
on real benchmark data.

## Claim Boundary

No runtime setting is currently allowed to be called globally best. The allowed
claim is only that the code path exists and has unit coverage.

## Next Action

Run the superconductivity baseline matrix and promote a sanitized TSV into
`experiment_notes/artifacts/runtime/`.

