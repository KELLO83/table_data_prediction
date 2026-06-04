# Source Card: OpenML superconductivity

## Metadata

- Type: OpenML regression dataset
- URL: https://www.openml.org/search?type=data&sort=runs&id=44964&status=active
- OpenML id: 44964
- Domain: materials science / superconductors
- Default target: `critical_temp`
- Local status: primary benchmark, already present as `superconductivity/openml_44964_superconductivity.csv`
- Compatibility label: controlled local benchmark

## Dataset Shape

OpenML metadata and local CSV agree on:

| Item | Value |
|---|---:|
| Rows | 21,263 |
| OpenML features including target | 82 |
| Numeric features including target | 82 |
| Symbolic features | 0 |
| Missing values | 0 |

Local target summary for `critical_temp`:

| Item | Value |
|---|---:|
| Minimum | 0.00021 |
| Maximum | 185.0 |
| Mean | 34.421219 |

## Relevance To This Project

This is the first dense numeric scientific regression benchmark. It tests
whether model families can exploit engineered physical descriptors without
categorical encoders or missing-value handling dominating the result.

## Protocol Match

| Item | Source | This Project | Match |
|---|---|---|---|
| Dataset | OpenML 44964 | local CSV copy | Yes |
| Target | `critical_temp` | `critical_temp` | Yes |
| Feature type | dense numeric | generic CSV numeric path | Yes |
| Default split | not fixed by source card | random holdout seed protocol | Project-defined |
| Metric | regression | RMSE/MAE/WAPE | Project-defined |

## Adopt / Defer / Avoid

- Adopt: all 81 features for final model comparison.
- Defer: compact `top_50_gain` only as a secondary interpretability run.
- Avoid: drawing categorical-model conclusions from this all-numeric dataset.

## Claim Boundary Impact

A win here supports numeric scientific regression only. It does not establish
robustness on categorical-heavy or high-dimensional sparse-ish tables.
