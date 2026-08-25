# eval_bench-
## Phase 5 — CI Gate

- Smoke test set: `data/smoke_set.jsonl`
- Gate script: `ci_gate.py`
- Workflow: `.github/workflows/eval.yml`

<!-- CI gate test trigger -->

## Note on CI mode (intentional, not an oversight)

`eval.yml` currently runs `ci_gate.py` in `--current` (cached) mode rather than
live mode. This is deliberate: the Gemini free-tier daily quota was exhausted
during development, so the gate runs against a cached result set instead of
making live API calls.

Once quota resets, swap back to live mode with:

```yaml
run: python -m eval_bench.ci_gate --baseline runs/baseline.json --suite smoke --threshold 0.02
```
