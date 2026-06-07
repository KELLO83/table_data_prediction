# Current Experiment Findings

## Current State

The project is set up for multi-dataset tabular regression comparison. The
maintained model list includes GBDT, neural tabular, transformer/attention, and
foundation-model families.

The current engineering focus is runtime-safe experimentation:

- direct PyTorch transformer wrappers can use SDPA, AMP, mixed precision, and
  optional `torch.compile`
- SDPA backend expectations are logged from representative model metadata
- runtime acceleration settings should be recorded with every promoted result

## Maintained Findings

| Finding | Status | Evidence |
|---|---|---|
| One dataset is insufficient for a model-family claim | Active | `concepts/benchmark_dataset_matrix.md` |
| Table shape should influence model priority | Active | `concepts/model_selection_by_table_shape.md` |
| Runtime acceleration must be evaluated with accuracy metrics | Active | `concepts/runtime_acceleration_ablation_protocol.md` |
| SDPA/AMP/compile code path exists for direct transformer wrappers | Active | Runtime code and unit tests; real-data ablation pending |

## Current Claim Boundary

- Do not claim one model family is best from one dataset.
- Do not claim one runtime setting is best without naming dataset, model,
  device, seed, batch size, and metrics.
- Treat compile speedups as workload-dependent because first-run compilation
  and recompilation can dominate short runs.
- Treat MAPE as diagnostic only when targets can be zero or near zero.

## Next Evidence Needed

1. Superconductivity runtime ablation for `ft_transformer`.
2. Superconductivity runtime ablation for `tab_transformer`.
3. One categorical-heavy dataset run to test whether TabTransformer runtime
   behavior differs from numeric-heavy data.
4. One larger-row dataset run to see whether compile break-even improves.

