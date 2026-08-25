from sklearn.metrics import cohen_kappa_score  # pip install scikit-learn --break-system-packages


def judge_human_agreement(judge_scores: list[int], human_scores: list[int]) -> float:
    """Both lists must be same length and same example order.
    weights='linear' penalizes a judge that's off by 1 less than off by 4."""
    return cohen_kappa_score(judge_scores, human_scores, weights="linear")


if __name__ == "__main__":
    # Example: after hand-labeling 100 examples, load both score lists
    import json

    with open("data/human_labels.jsonl") as f:
        human = {json.loads(l)["id"]: json.loads(l)["score"] for l in f}
    with open("data/judge_labels.jsonl") as f:
        judge = {json.loads(l)["id"]: json.loads(l)["score"] for l in f}

    ids = sorted(set(human) & set(judge))
    kappa = judge_human_agreement([judge[i] for i in ids], [human[i] for i in ids])
    print(f"Judge-human agreement (kappa): {kappa:.3f} over {len(ids)} examples")
    # Rule of thumb: <0.4 poor, 0.4-0.6 moderate, 0.6-0.8 good, >0.8 excellent
