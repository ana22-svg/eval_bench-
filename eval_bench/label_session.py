"""
Interactive labeling session: reads model predictions from a saved run
(runs/baseline.json), gets a judge score from Groq, then asks you for a
human 1-5 score. Appends to human_labels.jsonl and judge_labels.jsonl so
you can stop and resume anytime — it skips examples already labeled.
"""
import json
import os
from eval_bench.dataset import load_dataset
from eval_bench.judge import call_judge

DATASET_PATH = "data/sample_set.jsonl"
BASELINE_PATH = "runs/baseline.json"
HUMAN_LABELS_PATH = "data/human_labels.jsonl"
JUDGE_LABELS_PATH = "data/judge_labels.jsonl"


def already_labeled(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def load_predictions(path: str) -> dict[str, str]:
    with open(path) as f:
        rows = json.load(f)
    return {row["example_id"]: row["prediction"] for row in rows}


def append_label(path: str, example_id: str, score: int):
    with open(path, "a") as f:
        f.write(json.dumps({"id": example_id, "score": score}) + "\n")


def main():
    examples = load_dataset(DATASET_PATH)
    done = already_labeled(HUMAN_LABELS_PATH)
    predictions = load_predictions(BASELINE_PATH)

    missing = [ex.id for ex in examples if ex.id not in predictions]
    if missing:
        print(f"Skipping {len(missing)} example(s) with no baseline prediction: {missing}")
        print(f"(Run runner.py against the full dataset to fill these in.)\n")

    remaining = [ex for ex in examples if ex.id not in done and ex.id in predictions]
    print(f"{len(remaining)} of {len(examples)} examples left to label.\n")

    for ex in remaining:
        prediction = predictions[ex.id]
        judge_result = call_judge(ex.input, prediction, ex.gold or "")

        print("=" * 60)
        print(f"ID: {ex.id}")
        print(f"Question: {ex.input}")
        print(f"Reference answer: {ex.gold}")
        print(f"Model answer: {prediction}")
        print(f"Judge score: {judge_result.score}  (reasoning: {judge_result.reasoning})")

        while True:
            raw = input("Your score (1-5, or 's' to skip, 'q' to quit): ").strip().lower()
            if raw == "q":
                print("Stopping — progress saved, rerun anytime to continue.")
                return
            if raw == "s":
                break
            if raw in {"1", "2", "3", "4", "5"}:
                append_label(HUMAN_LABELS_PATH, ex.id, int(raw))
                append_label(JUDGE_LABELS_PATH, ex.id, judge_result.score)
                break
            print("Enter 1-5, 's' to skip, or 'q' to quit.")

    print("\nAll examples labeled.")


if __name__ == "__main__":
    main()
