# PRD: Industrial Tabular Regression Lab

## Goal

Build a portfolio-ready dashboard for tabular regression experiments. The app
should make model quality, runtime cost, dataset shape, and runtime acceleration
settings visible after training results are produced.

## Product Positioning

Industrial Tabular Regression Lab is an experiment console for comparing
serious tabular regression models across production-like table regimes:

- dense numeric science/materials data
- categorical-heavy insurance data
- large numeric performance-prediction data
- high-cardinality manufacturing data
- ultra-wide sparse-ish financial data

The dashboard is a read-only analysis surface. Training remains in the Python
CLI pipeline.

## Primary Users

- ML engineer reviewing benchmark results
- Hiring reviewer evaluating project scope and engineering maturity
- Developer comparing runtime acceleration choices such as SDPA, AMP, and
  `torch.compile`

## Core User Journeys

1. Open the dashboard and see the current best model, metric coverage, dataset
   coverage, and runtime coverage.
2. Compare models by dataset with RMSE, MAE, WAPE, train time, and predict time.
3. Inspect each dataset's shape and why it belongs in the benchmark suite.
4. Review runtime acceleration evidence for SDPA, AMP, mixed precision, and
   compile settings.
5. Read maintained experiment notes from the LLM Wiki.

## Data Sources

| Source | Path | Behavior |
|---|---|---|
| Experiment results | `results/tabular_regression_experiments.csv` | Optional. If absent, dashboard shows planned datasets and empty-result state. |
| Dataset registry | `web/src/lib/datasets.ts` | Static metadata aligned with `docs/llm_wiki` source cards. |
| LLM Wiki | `docs/llm_wiki/**/*.md` | Read-only experiment knowledge store. |

## Frontend Scope

### Dashboard

- KPI strip: datasets, completed runs, best RMSE, fastest training run.
- Dataset coverage table.
- Leaderboard table grouped by result rows.
- Performance scatter: train time vs RMSE.
- Runtime acceleration panel for SDPA/AMP/compile settings when result rows
  contain those fields.

### Dataset Explorer

- Domain
- target
- row count
- feature count
- categorical/numeric character
- portfolio role
- selectable dataset detail panel with OpenML link, leakage/id policy, and
  recommended modeling story

### Model Registry

- Model family grouping
- Recommended datasets/regimes
- Runtime capability labels
- Status before and after model training

### Runtime Matrix

- Planned SDPA/AMP/compile ablation matrix before runs exist
- Measured runtime rows after result CSV fields exist
- Clear pending/measured state per setting

### Result Filtering

- Dataset filter
- Model-family filter
- Metric sort mode
- Filtered leaderboard and scatter chart

### LLM Wiki Viewer

- Current findings
- Runtime acceleration protocol
- Runtime baseline plan
- full-note modal or inline expanded reader

## Non-goals

- Browser-triggered model training
- User authentication
- Editing wiki notes in the browser
- Uploading arbitrary datasets
- Real-time model serving before trained artifacts exist

## Success Criteria

- `npm run build` succeeds inside `web/`.
- Dashboard renders without a results CSV.
- Dashboard automatically displays runs when `results/tabular_regression_experiments.csv` exists.
- UI includes dataset, model, metric, runtime, and wiki evidence surfaces.
- App is visually portfolio-ready without hiding missing experiment state.
- Dataset cards expose portfolio rationale and OpenML links.
- Model registry makes model-family choices explainable before training.
- Runtime matrix shows planned ablations even before runtime result rows exist.
- Wiki cards can be expanded to read the maintained note body.

## Future Extensions

- Prediction demo backed by saved model artifacts.
- Dataset-specific drilldown pages.
- Config generator for model runs.
- Result promotion helper that writes sanitized TSV files into
  `docs/llm_wiki/experiment_notes/artifacts/`.
