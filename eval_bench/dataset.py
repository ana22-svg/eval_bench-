import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Example:
    id: str
    input: str
    gold: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    difficulty: str = "medium"


def load_dataset(path: str) -> list[Example]:
    """Loads a JSONL dataset. Each line: {"id", "input", "gold", "tags", "difficulty"}"""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(Example(
                id=row["id"],
                input=row["input"],
                gold=row.get("gold"),
                tags=row.get("tags", []),
                difficulty=row.get("difficulty", "medium"),
            ))
    return examples


def dataset_version(path: str) -> str:
    """Content hash so you can prove which exact dataset a run used."""
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
