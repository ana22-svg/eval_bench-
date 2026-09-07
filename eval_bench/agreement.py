import argparse
import json


def judge_human_agreement(judge_scores: list[int], human_scores: list[int]) -> float:
    """Both lists must be same length and same example order.
    weights='linear' penalizes a judge that's off by 1 less than off by 4."""
    if len(judge_scores) != len(human_scores):
        raise ValueError("judge_scores and human_scores must have the same length")
    if not judge_scores:
        return 0.0

    labels = sorted(set(judge_scores) | set(human_scores))
    max_distance = max(labels) - min(labels) or 1
    matrix = {(a, b): 0 for a in labels for b in labels}
    judge_counts = {label: 0 for label in labels}
    human_counts = {label: 0 for label in labels}

    for judge, human in zip(judge_scores, human_scores):
        matrix[(judge, human)] += 1
        judge_counts[judge] += 1
        human_counts[human] += 1

    n = len(judge_scores)

    def weight(a: int, b: int) -> float:
        return abs(a - b) / max_distance

    observed = sum(weight(a, b) * count for (a, b), count in matrix.items()) / n
    expected = 0.0
    for a in labels:
        for b in labels:
            expected += weight(a, b) * judge_counts[a] * human_counts[b] / (n * n)
    if expected == 0:
        return 1.0
    return 1 - observed / expected


def load_scores(path: str) -> dict[str, int]:
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["id"]: json.loads(line)["score"] for line in f if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", default="data/calibration_human_labels.jsonl")
    parser.add_argument("--judge", default="data/calibration_judge_labels.jsonl")
    args = parser.parse_args()

    human = load_scores(args.human)
    judge = load_scores(args.judge)

    ids = sorted(set(human) & set(judge))
    kappa = judge_human_agreement([judge[i] for i in ids], [human[i] for i in ids])
    print(f"Judge-human agreement (kappa): {kappa:.3f} over {len(ids)} examples")


if __name__ == "__main__":
    main()
