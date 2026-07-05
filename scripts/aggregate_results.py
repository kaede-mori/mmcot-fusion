"""Aggregate experiment results into a LaTeX-ready summary.

Usage: uv run python scripts/aggregate_results.py exp_a=experiments/<answer_dir1> exp_b=...
Reads each run's predictions_ans_test.json, prints per-category accuracies and
pairwise McNemar tests against the first run.
"""

import json
import sys

sys.path.insert(0, "src")
from mmcot_fusion.evaluation.metrics import mcnemar_test  # noqa: E402


def load(path):
    with open(f"{path}/predictions_ans_test.json") as f:
        return json.load(f)


def main():
    runs = {}
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        runs[name] = load(path)

    keys = [
        "acc_natural", "acc_social", "acc_language",
        "acc_has_text", "acc_has_image", "acc_no_context",
        "acc_grade_1_6", "acc_grade_7_12", "acc_average",
    ]
    header = "run        " + "  ".join(k.replace("acc_", "") for k in keys)
    print(header)
    for name, data in runs.items():
        row = "  ".join(str(data["scores"]["answer"][k]) for k in keys)
        print(f"{name:10s} {row}")

    names = list(runs)
    if len(names) >= 2:
        base = runs[names[0]]["per_question_correct"]
        qids = sorted(base)
        for other in names[1:]:
            comp = runs[other]["per_question_correct"]
            result = mcnemar_test(
                [base[q] for q in qids], [comp[q] for q in qids]
            )
            print(f"\nMcNemar {names[0]} vs {other}: {result}")


if __name__ == "__main__":
    main()
