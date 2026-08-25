#!/usr/bin/env python3
"""
Phase 4 — bootstrap CI + run-diffing on real data.

SCHEMA (matches your actual runs/*.json):
  Each run file is a JSON list of objects:
  {"example_id": str, "input": str, "prediction": str, "gold": str, "score": float}

USAGE:
  python3 phase4_compare.py \
      --baseline runs/baseline.json \
      --variant runs/variant.json \
      --out runs/phase4_report.json \
      --resamples 10000 \
      --alpha 0.05
"""

import argparse
import json
import random
from pathlib import Path


def load_run(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    assert isinstance(data, list), f"{path}: expected a JSON list at top level"
    by_id = {}
    for i, row in enumerate(data):
        assert "example_id" in row, f"{path}: row {i} missing 'example_id'"
        assert "score" in row, f"{path}: row {i} missing 'score'"
        assert isinstance(row["score"], (int, float)), f"{path}: row {i} score not numeric"
        by_id[row["example_id"]] = row
    return by_id


def bootstrap_ci(scores: list, n_resamples: int = 10000, alpha: float = 0.05, seed: int = 42):
    """Bootstrap CI on the mean of a list of scores."""
    assert len(scores) > 0, "bootstrap_ci: empty score list"
    rng = random.Random(seed)
    n = len(scores)
    means = []
    for _ in range(n_resamples):
        resample = [scores[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lower_idx = int((alpha / 2) * n_resamples)
    upper_idx = int((1 - alpha / 2) * n_resamples) - 1
    point_estimate = sum(scores) / n
    return {
        "mean": point_estimate,
        "ci_low": means[lower_idx],
        "ci_high": means[upper_idx],
        "n": n,
    }


def paired_bootstrap_test(deltas: list, n_resamples: int = 10000, alpha: float = 0.05, seed: int = 43):
    """
    Paired bootstrap significance test on per-example score deltas (variant - baseline).
    Null hypothesis: true mean delta is 0.
    """
    assert len(deltas) > 0, "paired_bootstrap_test: empty deltas"
    ci = bootstrap_ci(deltas, n_resamples=n_resamples, alpha=alpha, seed=seed)
    significant = not (ci["ci_low"] <= 0 <= ci["ci_high"])
    return {
        "mean_delta": ci["mean"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "significant": significant,
        "alpha": alpha,
    }


def diff_runs(baseline: dict, variant: dict, regression_threshold: float = 0.0):
    """
    Per-example diff. regression_threshold: a drop strictly greater than this
    counts as a regression (0.0 = any drop counts).
    """
    baseline_ids = set(baseline.keys())
    variant_ids = set(variant.keys())
    common = baseline_ids & variant_ids
    only_baseline = baseline_ids - variant_ids
    only_variant = variant_ids - baseline_ids

    regressions, improvements, unchanged = [], [], []
    deltas = []

    for ex_id in sorted(common):
        b_score = baseline[ex_id]["score"]
        v_score = variant[ex_id]["score"]
        delta = v_score - b_score
        deltas.append(delta)
        row = {
            "id": ex_id,
            "input": baseline[ex_id].get("input", ""),
            "gold": baseline[ex_id].get("gold", ""),
            "baseline_score": b_score,
            "variant_score": v_score,
            "delta": delta,
            "baseline_prediction": baseline[ex_id].get("prediction", ""),
            "variant_prediction": variant[ex_id].get("prediction", ""),
        }
        if delta < -regression_threshold:
            regressions.append(row)
        elif delta > regression_threshold:
            improvements.append(row)
        else:
            unchanged.append(row)

    return {
        "common_count": len(common),
        "only_in_baseline": sorted(only_baseline),
        "only_in_variant": sorted(only_variant),
        "regressions": regressions,
        "improvements": improvements,
        "unchanged": unchanged,
        "deltas": deltas,
    }


def print_report(baseline_ci, variant_ci, sig_test, diff, threshold):
    print("=" * 60)
    print("PHASE 4 — REAL DATA COMPARISON REPORT")
    print("=" * 60)
    print(f"\nBaseline : mean={baseline_ci['mean']:.4f}  "
          f"95% CI [{baseline_ci['ci_low']:.4f}, {baseline_ci['ci_high']:.4f}]  n={baseline_ci['n']}")
    print(f"Variant  : mean={variant_ci['mean']:.4f}  "
          f"95% CI [{variant_ci['ci_low']:.4f}, {variant_ci['ci_high']:.4f}]  n={variant_ci['n']}")

    print(f"\nPaired delta (variant - baseline): {sig_test['mean_delta']:+.4f}")
    print(f"95% CI on delta: [{sig_test['ci_low']:+.4f}, {sig_test['ci_high']:+.4f}]")
    verdict = "SIGNIFICANT" if sig_test["significant"] else "NOT significant"
    direction = "improvement" if sig_test["mean_delta"] > 0 else "regression" if sig_test["mean_delta"] < 0 else "no change"
    print(f"Verdict: {verdict} ({direction} at alpha={sig_test['alpha']})")

    print(f"\nCompared on {diff['common_count']} shared examples "
          f"(threshold for flagging: >{threshold})")
    if diff["only_in_baseline"]:
        print(f"  WARNING - only in baseline (missing from variant): {diff['only_in_baseline']}")
    if diff["only_in_variant"]:
        print(f"  WARNING - only in variant (missing from baseline): {diff['only_in_variant']}")

    print(f"\nRegressions : {len(diff['regressions'])}")
    for r in sorted(diff["regressions"], key=lambda x: x["delta"])[:10]:
        print(f"  [{r['id']}] {r['baseline_score']:.2f} -> {r['variant_score']:.2f}  (delta {r['delta']:+.2f})  input: {r['input'][:60]}")

    print(f"\nImprovements: {len(diff['improvements'])}")
    for r in sorted(diff["improvements"], key=lambda x: -x["delta"])[:10]:
        print(f"  [{r['id']}] {r['baseline_score']:.2f} -> {r['variant_score']:.2f}  (delta {r['delta']:+.2f})  input: {r['input'][:60]}")

    print(f"\nUnchanged   : {len(diff['unchanged'])}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Phase 4 real-data comparison (bootstrap CI + diff)")
    parser.add_argument("--baseline", required=True, help="Path to baseline run JSON")
    parser.add_argument("--variant", required=True, help="Path to variant run JSON")
    parser.add_argument("--out", default="runs/phase4_report.json", help="Where to write the full JSON report")
    parser.add_argument("--resamples", type=int, default=10000, help="Bootstrap resample count")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default 0.05 -> 95% CI)")
    parser.add_argument("--regression-threshold", type=float, default=0.0,
                         help="Min score drop to count as a regression (default: any drop)")
    args = parser.parse_args()

    baseline = load_run(args.baseline)
    variant = load_run(args.variant)

    assert len(baseline) > 0, f"{args.baseline}: no examples loaded"
    assert len(variant) > 0, f"{args.variant}: no examples loaded"

    baseline_scores = [r["score"] for r in baseline.values()]
    variant_scores = [r["score"] for r in variant.values()]

    baseline_ci = bootstrap_ci(baseline_scores, n_resamples=args.resamples, alpha=args.alpha, seed=42)
    variant_ci = bootstrap_ci(variant_scores, n_resamples=args.resamples, alpha=args.alpha, seed=42)

    diff = diff_runs(baseline, variant, regression_threshold=args.regression_threshold)
    assert len(diff["deltas"]) > 0, "No shared examples between baseline and variant — cannot run paired test"

    sig_test = paired_bootstrap_test(diff["deltas"], n_resamples=args.resamples, alpha=args.alpha, seed=43)

    print_report(baseline_ci, variant_ci, sig_test, diff, args.regression_threshold)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "baseline_ci": baseline_ci,
        "variant_ci": variant_ci,
        "paired_significance_test": sig_test,
        "diff": {
            "common_count": diff["common_count"],
            "only_in_baseline": diff["only_in_baseline"],
            "only_in_variant": diff["only_in_variant"],
            "regressions": diff["regressions"],
            "improvements": diff["improvements"],
            "unchanged_count": len(diff["unchanged"]),
        },
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
