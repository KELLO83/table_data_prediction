# Prediction ML

Generic CSV-based tabular regression model comparison across multiple OpenML
regression datasets.

Primary datasets:

- superconductivity, OpenML 44964, target `critical_temp`
- Allstate_Claims_Severity, OpenML 42571, target `loss`
- SGEMM_GPU_kernel_performance, OpenML 43144, target `Run1`

Expansion datasets:

- Mercedes_Benz_Greener_Manufacturing, OpenML 42570, target `y`
- Santander_transaction_value, OpenML 42572, target `target`

Use config-driven runs:

```powershell
.\.venv314\Scripts\python.exe train.py --config superconductivity\feature_selection_results\candidate_validation\recommended_config.json
```

Or direct CLI:

```powershell
.\.venv314\Scripts\python.exe train.py --csv superconductivity\openml_44964_superconductivity.csv --target critical_temp --model lightgbm
```
