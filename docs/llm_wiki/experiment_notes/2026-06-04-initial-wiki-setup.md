# Experiment Note: Initial Wiki Setup

## Run Metadata

- Date: 2026-06-04
- Command: documentation setup and OpenML metadata inspection
- Config: none
- Suite: OpenML tabular regression benchmark planning
- Model: none
- Seed: none
- Device: none
- Results row: none

## Metadata Findings

| Dataset | OpenML id | Rows | Features / definitions | Target | Regime |
|---|---:|---:|---:|---|---|
| superconductivity | 44964 | 21,263 | 82 | `critical_temp` | dense numeric scientific |
| Allstate Claims Severity | 42571 | 188,318 | 132 definitions, 116 symbolic | `loss` | categorical-heavy insurance |
| SGEMM GPU kernel performance | 43144 | 241,600 | 18 numeric | `Run1` | large numeric HPC |
| Mercedes-Benz Greener Manufacturing | 42570 | 4,209 | 378 definitions, 8 symbolic | `y` | high-cardinality manufacturing |
| Santander transaction value | 42572 | 4,459 | 4,993 definitions, string ID plus numeric features | `target` | high-dimensional sparse-ish numeric |

## Interpretation

The benchmark suite now covers three primary regimes: dense numeric scientific
features, categorical-heavy insurance data, and large numeric performance data.
The two expansion datasets add high-cardinality categorical manufacturing and
high-dimensional sparse-ish numeric finance behavior.

## Claim Boundary

The project should be described as multi-dataset tabular regression model
comparison. Superconductivity remains the local ready-to-run dataset, but it is
not the whole benchmark objective.

## Next Action

Create per-dataset CSV/config generation for the four OpenML datasets that are
not yet local, then smoke-test the current model registry on a small row sample
before running full sweeps.
