# Experiment Artifacts

This directory stores sanitized, machine-readable evidence copied from ignored
experiment outputs. Keep artifacts grouped by comparable scope so benchmark
claims do not mix incompatible protocols.

Do not add TSV, JSON, or visualization artifacts directly under this root.
Place each artifact under the closest scope folder and link that exact path from
the corresponding experiment note.

## Folder Map

| Folder | Scope |
|---|---|
| `runtime/` | Runtime ablation tables for SDPA, AMP, `torch.compile`, batch size, and prediction settings |
| `datasets/` | Dataset-scoped result snapshots |
| `models/` | Model-family or wrapper behavior snapshots |

## Artifact Rules

- Prefer TSV for small comparison tables copied from result CSVs.
- Prefer JSON for structured diagnostics or environment captures.
- Do not move raw training logs here unless they are trimmed and sanitized.
- Keep the artifact filename date-stamped and scoped to dataset/model/runtime.

