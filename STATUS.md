# second-reader run status: BLOCKED on ANTHROPIC_API_KEY

- when: 2026-09-04
- step: client setup (nothing model-facing can start)
- build: **amended architecture** — annotator model, tool-using agent grader,
  dual escalation paths, tools on/off ablation

## Error

```
ANTHROPIC_API_KEY is not set (checked environment and ./.env).
The grader agent and synthetic annotator cannot run without it.
```

Per the run contract, the pipeline stopped here rather than working around it.
No real grades exist and no real metrics have been published.

## What IS done and verified (no API needed)

| Piece | State |
|---|---|
| Data | DONE — 150 items (seed 42) + 218 peer votes from lmsys/mt_bench_human_judgments in DuckDB |
| Annotator model | DONE — 15 annotators, 8-12 items each, 2 BOILERPLATE + 2 POSITION_BIAS annotators; ground truth at both levels |
| Agent loop | DONE — tool-use loop (history / peers / full response / flag_for_review / emit_verdict), 6-call budget with forced verdict + budget_exhausted, full traces stored |
| Ablation | DONE — `--tools {on,off,both}`; both arms run over the same items |
| Tests | DONE — 32 pytest tests pass (schemas, agent termination/cap/retry, order-aware tools, injection at both scopes, escalation-aware precision/recall math) |
| Plumbing | DONE — full pipeline ran end-to-end with a tool-protocol mock client into `data/second_reader_mock.duckdb` and `results/mock/`: 150 items → 66 defective → 600 grades (2 modes × 2 orders) → routing → 126-row sweep → ablation table, tool metrics, example trace |
| Live run | BLOCKED — needs one Claude API key |

## Resume

1. Provide the key, either way:
   - `export ANTHROPIC_API_KEY=sk-ant-...`, or
   - create `second-reader/.env` containing `ANTHROPIC_API_KEY=sk-ant-...`
2. Run:

```
cd second-reader && make run
```

The pipeline is idempotent: completed submissions, grades, and traces are
kept in DuckDB and never re-run; a run that dies mid-way resumes where it
stopped.

Expected: ~130 annotator calls + ~300 single-mode runs + ~300 agent runs at
roughly 3-4 API rounds each — call it **$12-18** at current claude-sonnet-4-5
pricing, 30-45 minutes at 4 concurrent workers. On success this file is
deleted automatically and real metrics land in `results/calibration.csv` and
`results/metrics.md` (ablation table at the top).
