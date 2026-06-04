# Model Selection By Table Shape

## Candidate Ladder

Run this ladder before claiming any advanced model is useful:

```text
LightGBM
CatBoost
shape-specific neural/foundation candidates
```

`dummy_mean`, `dummy_median`, and `ridge` may exist as code-level sanity
wrappers, but they are excluded from the planned experiment candidate set because
they do not answer the current model-architecture question.

## Dense Numeric Tables

Examples: superconductivity, SGEMM.

Recommended first:
- LightGBM and CatBoost as serious baselines.
- TabM and RealMLP as practical neural baselines.
- TabR when retrieval-like neighborhoods may help.
- TabPFN/TabICLv2 only when runtime/checkpoint constraints are acceptable.

Defer:
- TabTransformer unless categorical features are present.
- Mamba/sequence models unless the table is converted into a real sequence
  prediction problem.

## Categorical-Heavy Tables

Examples: Allstate, Mercedes-Benz.

Recommended first:
- CatBoost, because it is the most direct current wrapper for categorical-heavy
  regression.
- LightGBM with explicit categorical handling or stable encoding.
- TabM/RealMLP only after the categorical preprocessing path is verified.

Add later:
- AutoGluon as an ensemble upper-bound benchmark.
- EBM when interpretability and main-effect/interaction plots are required.

## High-Dimensional Sparse-ish Numeric Tables

Example: Santander transaction value.

Recommended first:
- LightGBM/CatBoost with careful memory limits.
- Linear baselines only if a future high-dimensional-specific experiment adds
  them deliberately.
- XGBoost as the first missing GBDT add-on wrapper.

Guardrails:
- Exclude string IDs.
- Watch memory use before running all deep wrappers.
- Consider target transform if the target distribution is heavy-tailed.

## Metric Policy

Use RMSE and MAE as primary ranking metrics. WAPE is useful for scale-normalized
comparison within a dataset. MAPE is diagnostic only, because targets can be zero
or near zero.
