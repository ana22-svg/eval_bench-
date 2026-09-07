import json
import random
import statistics
from dataclasses import asdict
from eval_bench.runner import ExampleResult


def bootstrap_ci(scores: list[float], n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float, float]:
    """Returns (mean, lower_bound, upper_bound) via bootstrap resampling."""
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    boot_means = []
    for _ in range(n_boot):
        sample = [random.choice(scores) for _ in scores]
        boot_means.append(statistics.fmean(sample))
    boot_means.sort()
    lower_index = round((alpha / 2) * (n_boot - 1))
    upper_index = round((1 - alpha / 2) * (n_boot - 1))
    return statistics.fmean(scores), boot_means[lower_index], boot_means[upper_index]


def paired_significance(scores_a: list[float], scores_b: list[float]) -> float:
    """Paired bootstrap test for whether run B differs from run A.
    Returns p-value-ish stat: fraction of resamples where B doesn't beat A."""
    diffs = [b - a for a, b in zip(scores_a, scores_b)]
    if not diffs:
        return 1.0
    n_boot = 5000
    not_better = 0
    for _ in range(n_boot):
        sample_mean = statistics.fmean(random.choice(diffs) for _ in diffs)
        if sample_mean <= 0:
            not_better += 1
    return not_better / n_boot  # smaller = more confident B > A


def save_run(results: list[ExampleResult], path: str):
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


def load_run(path: str) -> list[ExampleResult]:
    with open(path) as f:
        return [ExampleResult(**_compatible_row(r)) for r in json.load(f)]


def _compatible_row(row: dict) -> dict:
    fields = ExampleResult.__dataclass_fields__
    return {name: row.get(name, field.default) for name, field in fields.items()}


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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("current")
    args = parser.parse_args()
    diff_runs(load_run(args.baseline), load_run(args.current))


if __name__ == "__main__":
    main()
