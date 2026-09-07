import hashlib
import json
from pathlib import Path
from typing import Callable


def cache_key(model: str, prompt_version: str, input_text: str) -> str:
    payload = json.dumps(
        {"model": model, "prompt_version": prompt_version, "input": input_text},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cached_response(
    cache_dir: str,
    model: str,
    prompt_version: str,
    input_text: str,
    generate: Callable[[], str],
) -> tuple[str, bool]:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{cache_key(model, prompt_version, input_text)}.json"
    if path.exists():
        row = json.loads(path.read_text(encoding="utf-8"))
        return row["prediction"], True

    prediction = generate()
    path.write_text(
        json.dumps(
            {
                "model": model,
                "prompt_version": prompt_version,
                "input": input_text,
                "prediction": prediction,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return prediction, False
