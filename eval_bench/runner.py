import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

from .cache import cached_response
from .costs import estimate_cost
from .dataset import Example, dataset_version, load_dataset
from .judge import call_judge
from .scorers import SCORERS


@dataclass
class ExampleResult:
    example_id: str
    input: str
    prediction: str
    gold: Optional[str]
    score: float
    scorer: str = "exact_match"
    judge_score: Optional[int] = None
    reasoning: Optional[str] = None
    latency_ms: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    model: str = "local-gold-baseline"
    prompt_version: str = "answer-short-v1"


def call_model(prompt: str, model: str, fallback_gold: Optional[str] = None) -> str:
    """Calls the configured system under test.

    The local baseline keeps CI deterministic. Set --model gemini-3.6-flash and
    GEMINI_API_KEY to evaluate a live model.
    """
    if model == "local-gold-baseline":
        return fallback_gold or ""
    if model == "gemini-3.6-flash":
        from google import genai

        if "GEMINI_API_KEY" not in os.environ:
            raise RuntimeError("GEMINI_API_KEY is required for gemini-3.6-flash")
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model=model,
            contents=f"Answer in as few words as possible, no punctuation, no explanation. Question: {prompt}",
        )
        return response.text.strip()
    raise ValueError(f"Unsupported model: {model}")


def _score_example(
    ex: Example,
    model: str,
    scorer_name: str,
    prompt_version: str,
    cache_dir: str,
    use_judge: bool,
) -> ExampleResult:
    start = time.perf_counter()
    prediction, was_cached = cached_response(
        cache_dir,
        model,
        prompt_version,
        ex.input,
        lambda: call_model(ex.input, model=model, fallback_gold=ex.gold),
    )
    elapsed_ms = round((time.perf_counter() - start) * 1000)
    cost = estimate_cost(model, ex.input, prediction)

    scorer = SCORERS[scorer_name]
    score = scorer(prediction, ex.gold or "") if ex.gold else 0.0
    judge_score = None
    reasoning = None
    if use_judge and ex.gold:
        judged = call_judge(ex.input, prediction, ex.gold)
        judge_score = judged.score
        reasoning = judged.reasoning
        if judged.score is not None:
            score = judged.score / 5

    return ExampleResult(
        example_id=ex.id,
        input=ex.input,
        prediction=prediction,
        gold=ex.gold,
        score=score,
        scorer="judge" if use_judge else scorer_name,
        judge_score=judge_score,
        reasoning=reasoning,
        latency_ms=elapsed_ms,
        cost_usd=cost.estimated_cost_usd,
        cached=was_cached,
        model=model,
        prompt_version=prompt_version,
    )


def run_suite(
    examples: list[Example],
    model: str = "local-gold-baseline",
    scorer_name: str = "exact_match",
    prompt_version: str = "answer-short-v1",
    cache_dir: str = ".cache/responses",
    concurrency: int = 4,
    use_judge: bool = False,
) -> list[ExampleResult]:
    if scorer_name not in SCORERS:
        raise ValueError(f"Unknown scorer {scorer_name!r}. Available: {sorted(SCORERS)}")
    if use_judge and "GROQ_API_KEY" not in os.environ:
        raise RuntimeError("GROQ_API_KEY is required when --judge is enabled")

    results_by_id = {}
    workers = max(1, concurrency)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_score_example, ex, model, scorer_name, prompt_version, cache_dir, use_judge): ex.id
            for ex in examples
        }
        for future in as_completed(futures):
            result = future.result()
            results_by_id[result.example_id] = result
    return [results_by_id[ex.id] for ex in examples]


def summarize(results: list[ExampleResult]) -> float:
    if not results:
        return 0.0
    return sum(r.score for r in results) / len(results)


def summarize_run(results: list[ExampleResult], suite: str, dataset_path: str, model: str, prompt_version: str) -> dict:
    return {
        "run_id": f"{suite}-{int(time.time())}",
        "suite": suite,
        "dataset_version": dataset_version(dataset_path),
        "model": model,
        "prompt_version": prompt_version,
        "example_count": len(results),
        "mean_score": summarize(results),
        "cost_usd": sum(r.cost_usd for r in results),
        "cached_examples": sum(1 for r in results if r.cached),
    }


def main() -> None:
    import argparse
    from .compare import save_run
    from .report import write_html_report
    from .storage import save_run_record

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/sample_set.jsonl")
    parser.add_argument("--suite", default="standard")
    parser.add_argument("--model", default="local-gold-baseline")
    parser.add_argument("--scorer", default="exact_match", choices=sorted(SCORERS))
    parser.add_argument("--prompt-version", default="answer-short-v1")
    parser.add_argument("--cache-dir", default=".cache/responses")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--output", default="runs/current.json")
    parser.add_argument("--report", default="reports/current.html")
    args = parser.parse_args()

    examples = load_dataset(args.dataset)
    print(f"Dataset version: {dataset_version(args.dataset)}")
    results = run_suite(
        examples,
        model=args.model,
        scorer_name=args.scorer,
        prompt_version=args.prompt_version,
        cache_dir=args.cache_dir,
        concurrency=args.concurrency,
        use_judge=args.judge,
    )
    for r in results:
        print(f"{r.example_id}: pred={r.prediction!r} gold={r.gold!r} score={r.score}")
    summary = summarize_run(results, args.suite, args.dataset, args.model, args.prompt_version)
    save_run(results, args.output)
    write_html_report(summary, results, args.report)
    save_run_record(summary["run_id"], args.suite, summary["dataset_version"], summary, results)
    print(f"\nOverall score: {summary['mean_score']:.2%}")
    print(f"Estimated cost: ${summary['cost_usd']:.6f}")
    print(f"Saved {len(results)} results to {args.output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
