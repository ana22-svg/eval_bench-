import json
import re


def normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", value.lower()).split())


def exact_match(prediction: str, gold: str) -> float:
    """Simple deterministic scorer: 1.0 if normalized strings match, else 0.0."""
    return 1.0 if normalize_text(prediction) == normalize_text(gold) else 0.0


def contains_match(prediction: str, gold: str) -> float:
    """Partial deterministic scorer for short factual answers."""
    pred = normalize_text(prediction)
    expected = normalize_text(gold)
    return 1.0 if expected and expected in pred else 0.0


def json_schema_validity(prediction: str) -> float:
    """Checks whether the model output is valid JSON."""
    try:
        json.loads(prediction)
    except json.JSONDecodeError:
        return 0.0
    return 1.0


SCORERS = {
    "exact_match": exact_match,
    "contains": contains_match,
    "json_valid": lambda prediction, gold=None: json_schema_validity(prediction),
}
