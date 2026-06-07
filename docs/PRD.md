# PRD: Generic Tabular Regression Model Comparison

## Goal

CSV 기반 테이블형 데이터에서 회귀 문제를 정의하고, 여러 데이터셋과 여러
모델의 성능을 같은 프로토콜로 비교한다.

## Problem Definition

```text
X = 사용자가 지정한 feature columns
y = 사용자가 지정한 numeric target column
task = supervised regression
```

데이터셋은 하나로 고정하지 않는다. 1차 실험은 다음 OpenML 회귀 데이터셋을 사용한다.

- superconductivity, OpenML 44964, target `critical_temp`
- Allstate_Claims_Severity, OpenML 42571, target `loss`
- SGEMM_GPU_kernel_performance, OpenML 43144, target `Run1`

확장 실험은 다음 데이터셋을 사용한다.

- Mercedes_Benz_Greener_Manufacturing, OpenML 42570, target `y`
- Santander_transaction_value, OpenML 42572, target `target`

모든 데이터셋은 다음 조건을 만족해야 한다.

- CSV로 읽을 수 있는 테이블형 데이터
- 연속형 또는 수치형 회귀 타깃 존재
- 학습 시점에만 알 수 있는 누수 컬럼은 제외 가능

## Primary Interface

```powershell
.\.venv314\Scripts\python.exe train.py --config configs\example.json
```

또는:

```powershell
.\.venv314\Scripts\python.exe train.py `
  --csv "path\to\data.csv" `
  --target target `
  --features f1,f2,f3 `
  --model lightgbm
```

## Model Scope

Experiment candidates should start from serious tabular models. The codebase
still exposes simple sklearn sanity wrappers, but they are not part of the
planned model-comparison candidate set.

- `lightgbm`, `catboost`
- `realmlp`, `tabm`, `tabr`, `dcnv2`, `node`
- `ft_transformer`, `tab_transformer`, `tabnet`
- `tabpfn`, `tabiclv2`

New wrappers should be added only when they cover a dataset shape not already
well represented by this registry.

## Success Criteria

- same dataset/target/split can be run across multiple models
- same model family can be compared across multiple dataset shapes
- result rows are appended to `results/tabular_regression_experiments.csv`
- feature columns and excluded columns are logged
- metrics include RMSE, MAE, MAPE, WAPE
- no dataset-specific assumptions are required in `train.py`

## Portfolio Dashboard Extension

The frontend product surface is specified in
`docs/INDUSTRIAL_TABULAR_REGRESSION_LAB_PRD.md`.

Industrial Tabular Regression Lab is a read-only Next.js dashboard that presents
experiment results, dataset shape, runtime acceleration settings, and LLM Wiki
findings. Training remains in the Python CLI pipeline; the dashboard consumes
result artifacts after runs complete.
