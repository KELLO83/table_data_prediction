# Source Card: OpenML Santander Transaction Value

## Metadata

- Type: OpenML regression dataset
- URL: https://www.openml.org/d/42572
- OpenML id: 42572
- Domain: finance / transaction-value prediction
- Default target: `target`
- Row id attribute: `ID`
- Local status: expansion benchmark, download/CSV conversion pending
- Compatibility label: high-dimensional sparse-ish numeric expansion dataset

## Dataset Shape

OpenML metadata:

| Item | Value |
|---|---:|
| Rows | 4,459 |
| Numeric features including target | 4,992 |
| Symbolic features | 0 |
| Feature definitions including string ID | 4,993 |
| Missing values | 0 |

The first feature is string `ID`; all other feature columns and the target are
numeric in the OpenML feature definition.

## Relevance To This Project

This dataset tests high-dimensional numeric behavior and memory pressure. It is
the expansion benchmark for sparse-ish anonymized finance features.

## Protocol Match

| Item | Source | This Project | Match |
|---|---|---|---|
| Dataset | OpenML 42572 | CSV conversion pending | Pending |
| Target | `target` | `target` | Yes |
| Row id | `ID` string | must exclude | Required |
| Feature type | thousands of numeric features | generic numeric path, memory-sensitive | Yes with guardrails |

## Adopt / Defer / Avoid

- Adopt: LightGBM/CatBoost with memory-aware params.
- Add later: XGBoost and sparse-aware linear baselines.
- Defer: full deep wrapper sweep until memory and runtime are bounded.
- Avoid: using `ID` as a feature.

## Claim Boundary Impact

Good results here support high-dimensional numeric robustness. They do not imply
categorical robustness.
