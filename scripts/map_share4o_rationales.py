"""Map the public Share4oReasoning GPT-4o rationales onto ScienceQA train qids.

Downloads sft/sqa/cot.{image,text}.none.jsonl from the Apache-2.0 dataset
released by LLaVA-Reasoner (arXiv:2410.16198), strips the trailing
"### Answer: X" verdict so Stage 1 remains a pure rationale generator, and
writes teacher_rationales/share4o_gpt4o_train.jsonl keyed by ScienceQA qid.

Usage: uv run python scripts/map_share4o_rationales.py
Requires data/scienceqa/pid_splits.json (see docs/REPRODUCING.md).
"""

import json
import os
import re
import urllib.request

BASE = "https://huggingface.co/datasets/Share4oReasoning/sft_data/resolve/main/sft/sqa"
OUT = "teacher_rationales/share4o_gpt4o_train.jsonl"


def main():
    records = {}
    for name in ("cot.image.none.jsonl", "cot.text.none.jsonl"):
        path = f"/tmp/{name}"
        if not os.path.exists(path):
            print("downloading", name)
            urllib.request.urlretrieve(f"{BASE}/{name}", path)
        for line in open(path):
            d = json.loads(line)
            idx = int(d["id"].split("_")[1])  # index into the ScienceQA train split
            text = d["conversations"][-1]["value"].strip()
            text = re.sub(r"\s*###\s*Answer:\s*[A-H]\.?\s*$", "", text).strip()
            records[idx] = text

    splits = json.load(open("data/scienceqa/pid_splits.json"))
    train = splits["train"]
    os.makedirs("teacher_rationales", exist_ok=True)
    written = 0
    with open(OUT, "w") as f:
        for i, qid in enumerate(train):
            if i in records:
                f.write(json.dumps({"qid": qid, "rationale": records[i]}) + "\n")
                written += 1
    print(f"wrote {written}/{len(train)} rationales -> {OUT} "
          f"({len(train) - written} uncovered questions keep the human rationale)")


if __name__ == "__main__":
    main()
