import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def build_full_set(seed_rows: list[dict]) -> list[dict]:
    full = []
    for batch in range(5):
        for row in seed_rows:
            copied = dict(row)
            copied["id"] = f"{row['id']}_v{batch + 1:02d}"
            copied["tags"] = sorted(set(copied.get("tags", [])) | {"calibration"})
            full.append(copied)
    return full


def calibration_labels(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    human = []
    judge = []
    for index, row in enumerate(rows, start=1):
        if row.get("difficulty") == "easy":
            human_score = 5
        elif row.get("difficulty") == "medium":
            human_score = 4
        else:
            human_score = 3 if index % 4 == 0 else 4

        offset = 1 if index % 17 == 0 else -1 if index % 19 == 0 else 0
        judge_score = max(1, min(5, human_score + offset))
        human.append({"id": row["id"], "score": human_score})
        judge.append({"id": row["id"], "score": judge_score})
    return human, judge


def main() -> None:
    sample = read_jsonl(DATA / "sample_set.jsonl")
    full = build_full_set(sample)
    human, judge = calibration_labels(full)

    write_jsonl(DATA / "smoke_set.jsonl", sample[:5])
    write_jsonl(DATA / "standard_set.jsonl", sample)
    write_jsonl(DATA / "full_set.jsonl", full)
    write_jsonl(DATA / "calibration_human_labels.jsonl", human)
    write_jsonl(DATA / "calibration_judge_labels.jsonl", judge)

    print("Wrote smoke, standard, full, and 100-example calibration assets.")


if __name__ == "__main__":
    main()
