# second-reader run status: BLOCKED on ANTHROPIC_API_KEY

- when: 2026-09-04
- step: client setup (start of execution-order step 2, the grader smoke test)

## Error

```
ANTHROPIC_API_KEY is not set (checked environment and ./.env).
The grader and synthetic annotator cannot run without it.
```

Per the run contract, the pipeline stopped here rather than working around it.
No real grades exist and no real metrics have been published.

## What IS done and verified (no API needed)

| Step | State |
|---|---|
| 1. Data load | DONE — 300 items sampled (seed 42) from lmsys/mt_bench_human_judgments into `data/second_reader.duckdb` |
| Tests | DONE — 28 pytest tests pass (schema validation, bounded retry, defect injection, precision/recall math) |
| Plumbing | DONE — full pipeline ran end-to-end with the deterministic mock grader (`--mock`) into a separate DB (`data/second_reader_mock.duckdb`) and `results/mock/`. 300 items → 300 submissions → 75 defects → 600 grades → routing → 63-config sweep. |
| 2–5 (real) | BLOCKED — needs one Claude API key |

## Resume

1. Provide the key, either way:
   - `export ANTHROPIC_API_KEY=sk-ant-...`, or
   - create `second-reader/.env` containing `ANTHROPIC_API_KEY=sk-ant-...`
2. Run:

```
cd second-reader && .venv/bin/python -m second_reader run --items 300
```

The pipeline is idempotent: completed submissions/grades are kept in DuckDB
and never re-run. If the run dies mid-way (rate limit, network), re-running
the same command resumes where it stopped.

Expected: ~900 claude-sonnet-4-5 calls (300 annotator + 600 grader),
roughly **$5–7** at current pricing, ~10–20 minutes at 4 concurrent workers.
On success this file is deleted automatically and real metrics land in
`results/calibration.csv` and `results/metrics.md`.
