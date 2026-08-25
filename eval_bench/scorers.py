def exact_match(prediction: str, gold: str) -> float:
    """Simple deterministic scorer: 1.0 if normalized strings match, else 0.0"""
    return 1.0 if prediction.strip().lower() == gold.strip().lower() else 0.0
