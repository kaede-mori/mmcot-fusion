"""Recompute the template-overlap and answer-leakage statistics of the report.

Reports: number of distinct lectures in the ScienceQA training split, verbatim
lecture/solution recurrence from train to test, and the answer-string leakage
rates of human vs. GPT-4o teacher rationales.

Usage: uv run python scripts/template_stats.py
Requires data/scienceqa/*.json (mmcot-fusion download) and, for the leakage
comparison, teacher_rationales/share4o_gpt4o_train.jsonl.
"""

import json
import os


def main():
    problems = json.load(open("data/scienceqa/problems.json"))
    splits = json.load(open("data/scienceqa/pid_splits.json"))
    train, test = splits["train"], splits["test"]

    train_lectures = {problems[q]["lecture"] for q in train if problems[q]["lecture"]}
    train_solutions = {problems[q]["solution"] for q in train if problems[q]["solution"]}
    print(f"distinct lectures in train: {len(train_lectures)}")

    with_lec = [q for q in test if problems[q]["lecture"]]
    seen = sum(1 for q in with_lec if problems[q]["lecture"] in train_lectures)
    print(f"test lectures verbatim in train: {seen}/{len(with_lec)} "
          f"({100*seen/len(with_lec):.1f}%)")

    with_sol = [q for q in test if problems[q]["solution"]]
    seen_s = sum(1 for q in with_sol if problems[q]["solution"] in train_solutions)
    print(f"test solutions verbatim in train: {seen_s}/{len(with_sol)} "
          f"({100*seen_s/len(with_sol):.1f}%)")

    teacher_path = "teacher_rationales/share4o_gpt4o_train.jsonl"
    if os.path.exists(teacher_path):
        teacher = {}
        for line in open(teacher_path):
            r = json.loads(line)
            teacher[r["qid"]] = r["rationale"]

        def leaks(text, p):
            return p["choices"][p["answer"]].lower() in text.lower()

        h_leak = h_n = t_leak = t_n = 0
        for q in train:
            p = problems[q]
            gold = (p["lecture"] + " " + p["solution"]).strip()
            if gold:
                h_n += 1
                h_leak += leaks(gold, p)
            if q in teacher:
                t_n += 1
                t_leak += leaks(teacher[q], p)
        print(f"answer string in human rationale: {h_leak}/{h_n} ({100*h_leak/h_n:.1f}%)")
        print(f"answer string in GPT-4o rationale: {t_leak}/{t_n} ({100*t_leak/t_n:.1f}%)")


if __name__ == "__main__":
    main()
