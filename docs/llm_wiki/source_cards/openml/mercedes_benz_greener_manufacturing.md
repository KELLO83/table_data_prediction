# Source Card: OpenML Mercedes-Benz Greener Manufacturing

## Metadata

- Type: OpenML regression dataset
- URL: https://www.openml.org/d/42570
- OpenML id: 42570
- Domain: manufacturing process/test-time prediction
- Default target: `y`
- Row id attribute: `ID`
- Local status: expansion benchmark, download/CSV conversion pending
- Compatibility label: high-cardinality categorical expansion dataset

## Dataset Shape

OpenML metadata:

| Item | Value |
|---|---:|
| Rows | 4,209 |
| OpenML features excluding target | 377 |
| Feature definitions including target/id | 378 |
| Numeric features including target/id | 370 |
| Symbolic features | 8 |
| Missing values | 0 |

## Relevance To This Project

This expansion dataset stresses small-sample overfitting with anonymized
manufacturing features and categorical variables. It is useful after the primary
three datasets establish the baseline pipeline.

## Protocol Match

| Item | Source | This Project | Match |
|---|---|---|---|
| Dataset | OpenML 42570 | CSV conversion pending | Pending |
| Target | `y` | `y` | Yes |
| Row id | `ID` | must exclude | Required |
| Feature type | mixed categorical/numeric, many anonymized columns | generic CSV with explicit categorical list | Needs config |

## Adopt / Defer / Avoid

- Adopt: CatBoost and LightGBM as first expansion baselines.
- Defer: deep neural wrappers until overfitting controls are defined.
- Avoid: using `ID` as a feature.

## Claim Boundary Impact

This dataset checks whether a model remains useful on smaller high-dimensional
manufacturing tables, especially with categorical handling.
