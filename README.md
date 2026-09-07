# LLM Eval Bench

A small evaluation harness for turning prompt and model changes into repeatable
numbers. It implements the Project 03 portfolio spec: fixed datasets, versioned
model and prompt runs, deterministic and judged scoring, judge calibration,
run-to-run comparison, confidence intervals, response caching, CI gating, cost
estimates, and HTML reports.

## What is included

- Dataset registry: `data/smoke_set.jsonl`, `data/standard_set.jsonl`, and
  `data/full_set.jsonl` with IDs, inputs, gold answers, tags, and difficulty.
- Deterministic scorers: exact match, contains match, and JSON validity.
- Optional LLM judge: Groq OpenAI-compatible endpoint with a pinned judge model
  and a written rubric.
- Judge calibration: 100 paired labels in
  `data/calibration_human_labels.jsonl` and
  `data/calibration_judge_labels.jsonl`.
- Runner: controlled concurrency, content-addressed response caching, full
  input/output capture, cost estimates, JSON output, SQLite run storage, and a
  static HTML report per run.
- Comparison: per-example improvements/regressions, bootstrap confidence
  intervals, and a paired bootstrap significance estimate.
- CI gate: GitHub Actions smoke suite that fails on regressions beyond a chosen
  threshold.
- Leaderboard: static HTML leaderboard across stored runs.

## Local usage

Run the deterministic local baseline:

```bash
python -m eval_bench.runner --dataset data/standard_set.jsonl --suite standard --output runs/baseline.json --report reports/baseline.html
```

Run the smoke CI gate:

```bash
python -m eval_bench.ci_gate --baseline runs/baseline.json --current runs/current.json --suite smoke --threshold 0.02
```

Compare two runs:

```bash
python -m eval_bench.compare runs/baseline.json runs/current.json
```

Measure judge-human agreement:

```bash
python -m eval_bench.agreement
```

Generate a leaderboard:

```bash
python -m eval_bench.leaderboard --output reports/leaderboard.html
```

Rebuild suite and calibration fixtures:

```bash
python scripts/build_eval_assets.py
```

## Live model runs

The default `local-gold-baseline` model is deterministic so local development
and CI do not require secrets. To evaluate Gemini instead:

```bash
set GEMINI_API_KEY=...
python -m eval_bench.runner --model gemini-3.6-flash --dataset data/smoke_set.jsonl --suite smoke
```

To use judge scoring:

```bash
set GROQ_API_KEY=...
python -m eval_bench.runner --judge --dataset data/smoke_set.jsonl --suite smoke
```

## Notes

The spec recommends Postgres for production storage. This repo uses SQLite by
default so the project is runnable without infrastructure; `eval_bench/storage.py`
keeps run metadata and per-example payloads isolated behind a small storage
module so replacing it with Postgres is straightforward.

