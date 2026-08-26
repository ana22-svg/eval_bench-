from dataclasses import dataclass
from .dataset import Example
from .scorers import exact_match


@dataclass
class ExampleResult:
    example_id: str
    input: str
    prediction: str
    gold: str | None
    score: float
    input_tokens: int = 0
    output_tokens: int = 0


import os
import time
from google import genai

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def call_model(prompt: str) -> tuple[str, int, int]:
    """Calls Gemini 3.6 Flash (free tier). Returns (text, input_tokens, output_tokens)."""
    response = _get_client().models.generate_content(
        model="gemini-3.6-flash",
        contents=f"Answer in as few words as possible, no punctuation, no explanation. Question: {prompt}",
    )
    usage = response.usage_metadata
    in_tok = getattr(usage, "prompt_token_count", 0) or 0
    out_tok = getattr(usage, "candidates_token_count", 0) or 0
    return response.text.strip(), in_tok, out_tok


def run_suite(examples: list[Example]) -> list[ExampleResult]:
    results = []
    for i, ex in enumerate(examples):
        prediction, in_tok, out_tok = call_model(ex.input)
        score = exact_match(prediction, ex.gold) if ex.gold else 0.0
        results.append(ExampleResult(ex.id, ex.input, prediction, ex.gold, score, in_tok, out_tok))
        if i < len(examples) - 1:
            time.sleep(13)
    return results


def summarize(results: list[ExampleResult]) -> float:
    if not results:
        return 0.0
    return sum(r.score for r in results) / len(results)


if __name__ == "__main__":
    import json
    from .dataset import load_dataset, dataset_version

    examples = load_dataset("data/sample_set.jsonl")
    print(f"Dataset version: {dataset_version('data/sample_set.jsonl')}")
    results = run_suite(examples)
    for r in results:
        print(f"{r.example_id}: pred={r.prediction!r} gold={r.gold!r} score={r.score}")
    print(f"\nOverall score: {summarize(results):.2%}")

    output = [
        {
            "example_id": r.example_id,
            "input": r.input,
            "prediction": r.prediction,
            "gold": r.gold,
            "score": r.score,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
        }
        for r in results
    ]
    with open("runs/baseline.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(output)} results to runs/baseline.json")
