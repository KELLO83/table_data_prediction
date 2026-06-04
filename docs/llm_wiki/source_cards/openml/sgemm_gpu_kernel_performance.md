# Source Card: OpenML SGEMM GPU Kernel Performance

## Metadata

- Type: OpenML regression dataset
- URL: https://www.openml.org/d/43144
- OpenML id: 43144
- Domain: HPC kernel performance prediction
- Default target: `Run1`
- Local status: primary benchmark, download/CSV conversion pending
- Compatibility label: large numeric benchmark

## Dataset Shape

OpenML metadata:

| Item | Value |
|---|---:|
| Rows | 241,600 |
| Features including target columns | 18 |
| Numeric features | 18 |
| Symbolic features | 0 |
| Missing values | 0 |

The source description defines 14 kernel parameters and four timing columns
`Run1`, `Run2`, `Run3`, and `Run4`; `Run1` is the OpenML default target.

## Relevance To This Project

This is the large numeric performance-prediction benchmark. It tests scaling,
runtime, and whether models benefit from structured low-cardinality numeric
parameters.

## Protocol Match

| Item | Source | This Project | Match |
|---|---|---|---|
| Dataset | OpenML 43144 | CSV conversion pending | Pending |
| Target | `Run1` | `Run1` initially | Yes |
| Feature type | numeric/ordinal and binary parameters | generic numeric path | Yes |
| Special note | log runtime target may help | optional target-transform experiment | Later |

## Adopt / Defer / Avoid

- Adopt: LightGBM, CatBoost, TabM, RealMLP.
- Defer: log-target variant until the plain target baseline exists.
- Avoid: using `Run2`, `Run3`, or `Run4` as input features when predicting
  `Run1` unless explicitly running a repeated-measurement task.

## Claim Boundary Impact

This dataset tests large numeric-table efficiency. It should be reported with
runtime and memory, not only RMSE.
