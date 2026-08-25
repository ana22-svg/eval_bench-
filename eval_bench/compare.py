import json
import numpy as np
from dataclasses import asdict
from eval_bench.runner import ExampleResult


def bootstrap_ci(scores: list[float], n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float, float]:
    """Returns (mean, lower_bound, upper_bound) via bootstrap resampling."""
    scores = np.array(scores)
    boot_means = [np.random.choice(scores, size=len(scores), replace=True).mean() for _ in range(n_boot)]
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return scores.mean(), lower, upper


def paired_significance(scores_a: list[float], scores_b: list[float]) -> float:
    """Paired bootstrap test for whether run B differs from run A.
    Returns p-value-ish stat: fraction of resamples where B doesn't beat A."""
    diffs = np.array(scores_b) - np.array(scores_a)
    n_boot = 5000
    boot_diffs = [np.random.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n_boot)]
    boot_diffs = np.array(boot_diffs)
    return float((boot_diffs <= 0).mean())  # smaller = more confident B > A


def save_run(results: list[ExampleResult], path: str):
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


def load_run(path: str) -> list[ExampleResult]:
    with open(path) as f:
        return [ExampleResult(**r) for r in json.load(f)]


def diff_runs(run_a: list[ExampleResult], run_b: list[ExampleResult]):
    a_by_id = {r.example_id: r for r in run_a}
    b_by_id = {r.example_id: r for r in run_b}

    regressions, improvements = [], []
    for eid in a_by_id:
        if eid not in b_by_id:
            continue
        ra, rb = a_by_id[eid], b_by_id[eid]
        if rb.score < ra.score:
            regressions.append((eid, ra, rb))
        elif rb.score > ra.score:
            improvements.append((eid, ra, rb))

    scores_a = [a_by_id[e].score for e in a_by_id if e in b_by_id]
    scores_b = [b_by_id[e].score for e in a_by_id if e in b_by_id]

    mean_a, lo_a, hi_a = bootstrap_ci(scores_a)
    mean_b, lo_b, hi_b = bootstrap_ci(scores_b)
    p = paired_significance(scores_a, scores_b)

    print(f"Run A: {mean_a:.3f} [{lo_a:.3f}, {hi_a:.3f}]")
    print(f"Run B: {mean_b:.3f} [{lo_b:.3f}, {hi_b:.3f}]")
    print(f"Confidence B > A: {(1 - p):.1%}")
    print(f"\nRegressions ({len(regressions)}):")
    for eid, ra, rb in regressions:
        print(f"  {eid}: {ra.score} -> {rb.score} | old={ra.prediction!r} new={rb.prediction!r}")
    print(f"\nImprovements ({len(improvements)}):")
    for eid, ra, rb in improvements:
        print(f"  {eid}: {ra.score} -> {rb.score}")

    return regressions, improvements
