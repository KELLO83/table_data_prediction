# Runtime Experiment Notes

This folder records runtime acceleration experiments for tabular PyTorch models.
Use it for SDPA, AMP, mixed precision, `torch.compile`, DataLoader transfer, and
prediction-loop changes.

## Required Metadata

Each note must name:

- Dataset and target
- Model
- Seed and split policy
- Device and PyTorch version
- Batch size and epoch count
- `enable_sdpa`
- `enable_amp`
- `amp_dtype`
- `enable_compile`
- `compile_mode`
- `matmul_precision`
- SDPA backend log line
- Result artifact path, if promoted

## Claim Rules

- Compare train/predict time only within the same dataset, model, split, epoch
  count, batch size policy, and device.
- Compare RMSE/MAE/WAPE alongside runtime; faster but materially worse metrics
  is not a pure acceleration win.
- Treat first-run compile latency separately from steady-state speed.
- Do not generalize one dataset's runtime result to all table regimes.

