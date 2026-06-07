# LLM Wiki

The LLM Wiki is the research-memory layer for the tabular regression benchmark.
It records what was learned, where the evidence lives, and which claims are
currently allowed.

It is not the implementation spec, not the only result table, and not a place
to silently rewrite raw evidence.

## Read Order

1. `INDEX.md`
   - Main entry point for source cards, concept notes, experiment notes, and
     local project context.
2. `concepts/current_experiment_findings.md`
   - Current synthesized interpretation across maintained experiments.
3. Relevant concept note under `concepts/`
   - Dataset selection, model-family fit, runtime acceleration, and risk rules.
4. Relevant `experiment_notes/**/*.md`
   - Run-level interpretation, commands, metrics, limitations, and next action.
5. Relevant `experiment_notes/artifacts/**/*.tsv`
   - Sanitized evidence snapshots copied from result outputs when a result is
     promoted into the wiki.
6. `LOG.md`
   - Chronological record of wiki updates.

## Layer Responsibilities

| Path | Responsibility |
|---|---|
| `source_cards/` | One-card summaries of datasets, papers, tools, and model families |
| `concepts/` | Synthesized project interpretation across sources and experiments |
| `experiment_notes/` | Dataset/model/runtime scoped experiment interpretation |
| `experiment_notes/artifacts/` | Sanitized TSV/JSON evidence snapshots grouped by comparable scope |
| `templates/` | Reusable source-card and experiment-note templates |
| `INDEX.md` | Human entry point and index |
| `LOG.md` | Chronological change record |

## Comparison Boundary

Numeric comparisons must name the dataset, target, split seed, metric, feature
policy, model params, and runtime settings. If any of those differ, treat the
numbers as context rather than a winner table.

Runtime comparisons must also name:

- `device`
- `batch_size`
- `enable_sdpa`
- `enable_amp`
- `amp_dtype`
- `enable_compile`
- `compile_mode`
- `matmul_precision`
- observed SDPA backend log

