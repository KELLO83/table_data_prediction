# Runtime Artifacts

Store sanitized runtime ablation evidence here.

Recommended columns for TSV artifacts:

```text
date	dataset	target	model	seed	device	rows	features	batch_size	epochs	enable_sdpa	sdpa_backend	enable_amp	amp_dtype	enable_compile	compile_mode	matmul_precision	train_time_sec	predict_time_sec	rmse	mae	wape	notes
```

Keep one artifact focused on one comparable question, for example:

- FT-Transformer AMP vs compile on superconductivity
- TabTransformer SDPA backend behavior on categorical-heavy data
- Batch-size sweep for one model and one dataset

