import argparse
import sys
from eval_bench.dataset import load_dataset
from eval_bench.runner import run_suite, summarize_run
from eval_bench.compare import load_run, diff_runs, bootstrap_ci
from eval_bench.compare import save_run
from eval_bench.report import write_html_report
from eval_bench.storage import save_run_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.02, help="Max allowed score drop")
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--current", type=str, default="runs/current.json")
    parser.add_argument("--suite", type=str, default="smoke", choices=["smoke", "standard", "full"])
    parser.add_argument("--model", type=str, default="local-gold-baseline")
    parser.add_argument("--scorer", type=str, default="exact_match")
    parser.add_argument("--prompt-version", type=str, default="answer-short-v1")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--judge", action="store_true")
    args = parser.parse_args()

    dataset_path = {
        "smoke": "data/smoke_set.jsonl",
        "standard": "data/standard_set.jsonl",
        "full": "data/full_set.jsonl",
    }[args.suite]

    examples = load_dataset(dataset_path)
    current_results = run_suite(
        examples,
        model=args.model,
        scorer_name=args.scorer,
        prompt_version=args.prompt_version,
        concurrency=args.concurrency,
        use_judge=args.judge,
    )
    save_run(current_results, args.current)
    summary = summarize_run(current_results, args.suite, dataset_path, args.model, args.prompt_version)
    write_html_report(summary, current_results, f"reports/{args.suite}.html")
    save_run_record(summary["run_id"], args.suite, summary["dataset_version"], summary, current_results)

    baseline_results = load_run(args.baseline)
    current_mean, _, _ = bootstrap_ci([r.score for r in current_results])
    baseline_mean, _, _ = bootstrap_ci([r.score for r in baseline_results])

    print(f"Baseline: {baseline_mean:.3f} | Current: {current_mean:.3f}")

    if baseline_mean - current_mean > args.threshold:
        print(f"REGRESSION: score dropped by {baseline_mean - current_mean:.3f} "
              f"(threshold: {args.threshold})")
        diff_runs(baseline_results, current_results)
        sys.exit(1)

    print(f"Cost estimate: ${summary['cost_usd']:.6f}")
    print("No regression detected")


if __name__ == "__main__":
    main()
