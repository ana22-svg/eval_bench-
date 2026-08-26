import argparse
import sys
from eval_bench.dataset import load_dataset
from eval_bench.runner import run_suite, summarize
from eval_bench.compare import load_run, diff_runs, bootstrap_ci, save_run

# Gemini 3.6 Flash pricing, USD per million tokens (introductory rate through
# Dec 31, 2026 — doubles to $1.50 / $7.50 on Jan 1, 2027; update if it changes).
GEMINI_INPUT_RATE = 0.75 / 1_000_000
GEMINI_OUTPUT_RATE = 3.75 / 1_000_000


def compute_cost(results) -> tuple[float, int, int]:
    """Returns (cost_usd, total_input_tokens, total_output_tokens)."""
    in_tok = sum(getattr(r, "input_tokens", 0) for r in results)
    out_tok = sum(getattr(r, "output_tokens", 0) for r in results)
    cost = in_tok * GEMINI_INPUT_RATE + out_tok * GEMINI_OUTPUT_RATE
    return cost, in_tok, out_tok


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

    cost, in_tok, out_tok = compute_cost(current_results)
    print(f"Cost: ${cost:.4f} ({in_tok:,} in / {out_tok:,} out tokens)")

    if baseline_mean - current_mean > args.threshold:
        print(f"❌ REGRESSION: score dropped by {baseline_mean - current_mean:.3f} "
              f"(threshold: {args.threshold})")
        diff_runs(baseline_results, current_results)
        sys.exit(1)

    print("✅ No regression detected")


if __name__ == "__main__":
    main()
