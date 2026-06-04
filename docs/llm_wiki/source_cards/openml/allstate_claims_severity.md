# Source Card: OpenML Allstate Claims Severity

## Metadata

- Type: OpenML regression dataset
- URL: https://www.openml.org/d/42571
- OpenML id: 42571
- Domain: insurance claim severity
- Default target: `loss`
- Row id attribute: `id`
- Local status: primary benchmark, download/CSV conversion pending
- Compatibility label: categorical-heavy benchmark

## Dataset Shape

OpenML metadata:

| Item | Value |
|---|---:|
| Rows | 188,318 |
| OpenML features excluding target | 131 |
| Feature definitions including target/id | 132 |
| Numeric features including target/id | 16 |
| Symbolic features | 116 |
| Missing values | 0 |

The OpenML description states that variables prefixed with `cat` are
categorical and variables prefixed with `cont` are continuous.

## Relevance To This Project

This is the main categorical-heavy large-table benchmark. It is required because
models that perform well on superconductivity may still fail when categorical
encoding dominates.

## Protocol Match

| Item | Source | This Project | Match |
|---|---|---|---|
| Dataset | OpenML 42571 | CSV conversion pending | Pending |
| Target | `loss` | `loss` | Yes |
| Feature type | 116 categorical, 15 continuous plus id/target accounting | generic CSV with categorical list | Needs explicit config |
| Row id | `id` | must exclude | Required |
| Metric | regression | RMSE/MAE/WAPE | Project-defined |

## Adopt / Defer / Avoid

- Adopt: CatBoost first; LightGBM only with explicit categorical or stable
  encoding policy.
- Defer: neural wrappers until the categorical preprocessing path is smoke
  tested.
- Avoid: treating `id` as a feature.

## Claim Boundary Impact

A model that wins here provides evidence for categorical-heavy production-style
tabular regression, not just dense numeric benchmarks.
