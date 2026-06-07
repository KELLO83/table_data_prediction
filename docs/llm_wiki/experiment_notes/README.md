# Experiment Notes

Experiment notes are grouped by evidence scope. Do not add new dated notes
directly under this folder. Root-level content is limited to this README and
`artifacts/`.

## Folder Map

| Folder | Scope |
|---|---|
| `runtime/` | AMP, mixed precision, `torch.compile`, SDPA, batch-transfer, and predict-loop ablations |
| `datasets/` | Dataset-specific runs when a result is promoted beyond raw CSV output |
| `models/` | Model-family or wrapper behavior notes that are not tied to one dataset |
| `artifacts/` | Sanitized TSV/JSON evidence snapshots grouped by comparable scope |

## Rules

- Keep each note next to notes with the same evidence scope.
- Do not preserve smoke/refactor-only notes unless they explain a maintained
  result or a runtime decision.
- Link artifacts using the matching artifact folder.
- If a new dataset or model-family scope is added, create the folder README
  before adding dated notes.
- Each note must contain one bounded next action or explicitly say no follow-up
  is needed.

