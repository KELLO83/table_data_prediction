# LLM Wiki Index

## Scope

This wiki tracks source-grounded knowledge for the tabular regression benchmark
project. The project goal is to compare multiple regression models across
multiple table-shaped datasets with different data regimes.

Current benchmark direction:

```text
OpenML regression datasets
  -> normalize each dataset into the generic CSV contract
  -> run comparable model families under one split/metric protocol
  -> record dataset shape, leakage exclusions, runtime risk, and metrics
```

Chronological wiki changes are recorded in `LOG.md`.

## Required Local Context

- Project overview: `README.md`
- Product requirements: `docs/PRD.md`
- System architecture: `docs/SYSTEM_ARCH.md`
- Data contract: `docs/DATA_CONTRACT.md`
- Dataset benchmark plan: `docs/DATASETEXPLAIN.md`
- Variable policy: `docs/TABULAR_DATA_VARIABLES.md`
- Evaluation protocol: `docs/EVALUATION_PROTOCOL.md`
- Model tiers: `docs/MODEL_TIERS.md`
- Run risk checklist: `docs/RUN_RISK_CHECKLIST.md`
- Current superconductivity feature validation:
  `superconductivity/feature_selection_results/candidate_validation/summary.md`

## Benchmark Datasets

| Stage | Dataset | OpenML id | Shape / regime | Target | Card |
|---|---|---:|---|---|---|
| Primary | superconductivity | 44964 | 21,263 rows, dense numeric materials features | `critical_temp` | `source_cards/openml/superconductivity.md` |
| Primary | Allstate_Claims_Severity | 42571 | 188,318 rows, categorical-heavy insurance claims | `loss` | `source_cards/openml/allstate_claims_severity.md` |
| Primary | SGEMM_GPU_kernel_performance | 43144 | 241,600 rows, numeric HPC kernel parameters | `Run1` | `source_cards/openml/sgemm_gpu_kernel_performance.md` |
| Expansion | Mercedes_Benz_Greener_Manufacturing | 42570 | 4,209 rows, high-cardinality manufacturing categoricals plus many binary/numeric columns | `y` | `source_cards/openml/mercedes_benz_greener_manufacturing.md` |
| Expansion | Santander_transaction_value | 42572 | 4,459 rows, 4,992 sparse-ish numeric features plus string ID | `target` | `source_cards/openml/santander_transaction_value.md` |

## Concept Notes

| Concept | Note |
|---|---|
| Current experiment findings | `concepts/current_experiment_findings.md` |
| Benchmark dataset matrix | `concepts/benchmark_dataset_matrix.md` |
| Model selection by table shape | `concepts/model_selection_by_table_shape.md` |
| Model candidate profiles | `concepts/model_candidate_profiles_ko.md` |
| Runtime acceleration ablation protocol | `concepts/runtime_acceleration_ablation_protocol.md` |

## Experiment Notes

| Date | Run | Note |
|---|---|---|
| 2026-06-07 | Runtime acceleration baseline plan | `experiment_notes/runtime/2026-06-07-runtime-acceleration-baseline-plan.md` |
| 2026-06-04 | Initial wiki setup | `experiment_notes/2026-06-04-initial-wiki-setup.md` |

## Experiment Artifact Roots

| Scope | Folder |
|---|---|
| Runtime acceleration | `experiment_notes/artifacts/runtime/` |
| Dataset-scoped results | `experiment_notes/artifacts/datasets/` |
| Model-family diagnostics | `experiment_notes/artifacts/models/` |

## Claim Boundaries

- Do not claim a model family is generally best from one dataset.
- Do not compare results across datasets without naming the dataset, target,
  split, metric, feature exclusions, and runtime environment.
- Treat MAPE as diagnostic only when targets can be zero or near zero.
- Exclude row IDs and post-target/leakage columns explicitly in every config.
- Runtime comparisons must name AMP, SDPA, compile, matmul precision, batch size,
  device, and SDPA backend diagnostics.
