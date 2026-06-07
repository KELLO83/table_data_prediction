# Runtime Acceleration Ablation Protocol

## Purpose

This note defines how to evaluate SDPA, AMP, mixed precision, `torch.compile`,
and batch-transfer settings without mixing incompatible evidence.

## Required Controls

Keep these fixed inside one runtime comparison:

- Dataset
- Target
- Feature exclusion policy
- Split seed
- Model
- Epoch count
- Batch size, unless batch size is the ablation variable
- Device
- PyTorch version

## Required Runtime Fields

Record these for every promoted run:

| Field | Why it matters |
|---|---|
| `enable_sdpa` | Determines whether MHA no-weight SDPA path is enabled |
| `sdpa_backend` | Confirms expected CUDA backend such as flash, efficient, cudnn, or math |
| `enable_amp` | Enables mixed precision regions |
| `amp_dtype` | `float16` and `bfloat16` can differ in speed and stability |
| `enable_compile` | Controls graph compilation policy |
| `compile_mode` | Changes compile overhead and steady-state behavior |
| `matmul_precision` | Controls TF32/internal matmul precision behavior |
| `batch_size` | Changes GPU utilization and compile shapes |

## Decision Rules

- A runtime setting is a win only if it improves time without materially
  degrading RMSE, MAE, or WAPE.
- `torch.compile` must report both first-run wall time and steady-state
  interpretation when possible.
- AMP instability must be recorded as an accuracy/stability finding, not hidden
  as a failed run.
- SDPA backend logs are expected backend diagnostics, not proof of global speed.

## Promotion Rule

Promote a result into the wiki only when it has:

- a command or config reference
- a result row or sanitized TSV artifact
- runtime settings
- metrics
- one interpretation paragraph
- one next action

