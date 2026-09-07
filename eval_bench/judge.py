import json
import os
from dataclasses import dataclass
from typing import Optional

JUDGE_MODEL = "openai/gpt-oss-120b"  # pin this — never "latest"

RUBRIC = """You are grading a model's answer against a reference answer.
Score from 1-5 using this rubric:
5 - Fully correct, matches the reference in substance.
4 - Mostly correct, minor omission or imprecision.
3 - Partially correct, misses a key element.
2 - Largely incorrect but shows some relevant understanding.
1 - Incorrect or irrelevant.

Respond ONLY with JSON: {"score": <int 1-5>, "reasoning": "<one sentence>"}
"""

@dataclass
class JudgeResult:
    score: Optional[int]
    reasoning: str
    judge_model: str


def _client():
    if "GROQ_API_KEY" not in os.environ:
        raise RuntimeError("GROQ_API_KEY is required for judge scoring")
    from openai import OpenAI

    return OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )


def call_judge(question: str, prediction: str, gold: str) -> JudgeResult:
    last_err = None
    for attempt in range(2):
        resp = _client().chat.completions.create(
            model=JUDGE_MODEL,
            max_tokens=400,
            messages=[
                {"role": "system", "content": RUBRIC},
                {"role": "user", "content": f"Question: {question}\nReference answer: {gold}\nModel answer: {prediction}"},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
        try:
            parsed = json.loads(raw)
            return JudgeResult(score=int(parsed["score"]), reasoning=parsed["reasoning"], judge_model=JUDGE_MODEL)
        except json.JSONDecodeError as e:
            last_err = e
            print(f"  [judge retry {attempt + 1}] malformed JSON, retrying... ({e})")
    print(f"  [judge] giving up after retries, defaulting to score=None. Raw: {raw!r}")
    return JudgeResult(score=None, reasoning=f"JSON parse failed: {last_err}", judge_model=JUDGE_MODEL)
