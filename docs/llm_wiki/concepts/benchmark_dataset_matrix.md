# Benchmark Dataset Matrix

## Purpose

The benchmark suite should force models to face meaningfully different table
shapes. A single numeric materials dataset is not enough to choose a robust
tabular regression architecture.

## Primary Experiments

| Dataset | Why It Matters | Expected Hard Part | First Models |
|---|---|---|---|
| superconductivity | Scientific/materials regression with dense numeric engineered features | Smooth nonlinear interactions across physical descriptors | LightGBM, CatBoost, TabM, RealMLP, TabR, TabPFN/TabICLv2 |
| Allstate Claims Severity | Large insurance dataset with many categorical columns | Categorical encoding, high-cardinality effects, skewed loss target | CatBoost, LightGBM with categorical/encoded features, TabM, AutoGluon later |
| SGEMM GPU kernel performance | Large numeric HPC performance prediction | Structured parameter space, repeated timing targets, possible log-target benefit | LightGBM, CatBoost, TabM, RealMLP, XGBoost later |

## Expansion Experiments

| Dataset | Why It Matters | Expected Hard Part | First Models |
|---|---|---|---|
| Mercedes-Benz Greener Manufacturing | Small manufacturing table with high-cardinality categoricals and many anonymized columns | Overfitting, categorical handling, ID exclusion | CatBoost, LightGBM, EBM/ExtraTrees later |
| Santander transaction value | Very high-dimensional sparse-ish numeric finance table | Feature sparsity, target skew, compute/memory pressure | LightGBM, CatBoost, memory-gated TabM; linear models only if added later |

## Dataset-Level Rules

- Use each OpenML default target unless a source card says otherwise.
- Exclude row ID columns such as `id`, `ID`, and string identifiers.
- Keep per-dataset configs explicit once downloaded or converted to CSV.
- Do not use one global feature list across datasets.
- Record dataset id, OpenML URL, target, row count, feature count, missing-value
  count, and feature type distribution in every result summary.
