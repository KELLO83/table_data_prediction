# Dataset Explanation

The active project compares tabular regression models across multiple dataset
shapes. Every dataset must still satisfy the generic CSV contract:

- feature columns
- one numeric regression target
- explicit exclusions for identifiers, metadata, and leakage columns

## Primary Experiments

| Dataset | OpenML id | Domain | Shape | Target | Main Risk |
|---|---:|---|---|---|---|
| superconductivity | 44964 | materials/science | 21,263 rows, 82 numeric columns including target | `critical_temp` | dense numeric nonlinear interactions |
| Allstate_Claims_Severity | 42571 | insurance | 188,318 rows, 116 symbolic features plus numeric columns | `loss` | categorical-heavy encoding and target skew |
| SGEMM_GPU_kernel_performance | 43144 | HPC performance | 241,600 rows, 18 numeric columns | `Run1` | large numeric runtime/memory and repeated timing columns |

## Expansion Experiments

| Dataset | OpenML id | Domain | Shape | Target | Main Risk |
|---|---:|---|---|---|---|
| Mercedes_Benz_Greener_Manufacturing | 42570 | manufacturing | 4,209 rows, mixed high-cardinality categoricals and many numeric/binary columns | `y` | small sample overfitting and `ID` exclusion |
| Santander_transaction_value | 42572 | finance | 4,459 rows, 4,992 numeric features plus string `ID` | `target` | high-dimensional sparse-ish numeric features |

## Current Local Dataset

The ready-to-run local dataset is:

```text
superconductivity/openml_44964_superconductivity.csv
```

Use all 81 non-target features for final superconductivity model comparison.
Five-seed LightGBM feature validation found the full feature set strongest:

```text
all_81 RMSE mean = 9.227177
top_50_gain RMSE mean = 9.254649
corr_filtered_58 RMSE mean = 9.260771
```

## Dataset Preparation Rules

- Keep one config per dataset/model once data is downloaded or converted.
- Exclude row id columns such as `id` and `ID`.
- For SGEMM, do not use `Run2`, `Run3`, or `Run4` as inputs when predicting
  `Run1` unless the task is explicitly redefined as repeated-measurement
  prediction.
- Record OpenML id, target, row count, feature count, categorical columns,
  excluded columns, and missing-value count in every experiment note.
