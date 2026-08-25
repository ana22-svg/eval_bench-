import json
import os
from dataclasses import dataclass

JUDGE_MODEL = "claude-sonnet-5"  # pin this — never "latest"

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
    score: int
    reasoning: str
    judge_model: str


def call_judge(question: str, prediction: str, gold: str) -> JudgeResult:
    """Calls the judge model. Fill in the actual API call for your SDK."""
    import anthropic  # pip install anthropic --break-system-packages

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=200,
        system=RUBRIC,
        messages=[{
            "role": "user",
            "content": f"Question: {question}\nReference answer: {gold}\nModel answer: {prediction}"
        }],
    )
    raw = msg.content[0].text.strip()
    parsed = json.loads(raw)
    return JudgeResult(score=parsed["score"], reasoning=parsed["reasoning"], judge_model=JUDGE_MODEL)


import random

def call_judge_debiased(question: str, prediction: str, gold: str) -> JudgeResult:
    """Randomizes which answer is labeled 'A' vs 'B' to strip position bias,
    and truncates long answers so length alone can't win points."""
    a, b = prediction, gold
    swapped = random.random() < 0.5
    if swapped:
        a, b = b, a
    # pass a/b into your prompt as "Answer A" / "Answer B" instead of
    # "model answer" / "reference answer", then unswap the verdict after.
    ...
