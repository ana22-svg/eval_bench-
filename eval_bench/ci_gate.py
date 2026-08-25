import argparse
import sys
from eval_bench.dataset import load_dataset
from eval_bench.runner import run_suite, summarize
from eval_bench.compare import load_run, diff_runs, bootstrap_ci, save_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.02, help="Max allowed score drop")
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--suite", type=str, default="smoke", choices=["smoke", "standard", "full"])
    parser.add_argument("--current", type=str, default=None, help="Path to pre-computed current run (skips live model calls)")
    args = parser.parse_args()

    dataset_path = {
        "smoke": "data/smoke_set.jsonl",
        "standard": "data/standard_set.jsonl",
        "full": "data/full_set.jsonl",
    }[args.suite]

    if args.current:
        current_results = load_run(args.current)
    else:
        examples = load_dataset(dataset_path)
        current_results = run_suite(examples)
        save_run(current_results, "runs/current.json")

    baseline_results = load_run(args.baseline)
    current_mean, _, _ = bootstrap_ci([r.score for r in current_results])
    baseline_mean, _, _ = bootstrap_ci([r.score for r in baseline_results])

    print(f"Baseline: {baseline_mean:.3f} | Current: {current_mean:.3f}")

    if baseline_mean - current_mean > args.threshold:
        print(f"❌ REGRESSION: score dropped by {baseline_mean - current_mean:.3f} "
              f"(threshold: {args.threshold})")
        diff_runs(baseline_results, current_results)
        sys.exit(1)

    print("✅ No regression detected")


if __name__ == "__main__":
    main()
