"""Prepare M3CoT (Chen et al., ACL 2024) in ScienceQA-compatible layout.

Downloads LightChen2333/M3CoT from HuggingFace, converts problems into the
ScienceQA problems.json schema (rationale -> solution), and extracts ViT
features (vit_large_patch32_384, matching cooelf/vision_features) for all
images.

Outputs:
  data_m3cot/scienceqa/problems.json, pid_splits.json
  vision_features_m3cot/vit.npy, name_map.json

Usage: uv run --group experiments python scripts/prepare_m3cot.py [--device mps]
"""

import argparse
import json
import os

import numpy as np
import torch

OPTIONS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--data-out", default="data_m3cot")
    parser.add_argument("--features-out", default="vision_features_m3cot")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    from datasets import load_dataset
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    ds = load_dataset("LightChen2333/M3CoT")

    problems = {}
    pid_splits = {"train": [], "val": [], "test": []}
    images = []  # (qid, PIL image)
    for split_hf, split in (("train", "train"), ("validation", "val"), ("test", "test")):
        for ex in ds[split_hf]:
            qid = ex["id"]
            answer_idx = OPTIONS.index(ex["answer"]) if isinstance(ex["answer"], str) else int(ex["answer"])
            problems[qid] = {
                "question": ex["question"],
                "choices": ex["choices"],
                "answer": answer_idx,
                "hint": "",
                "lecture": "",
                "solution": ex["rationale"] or "",
                "split": split,
                "image": "image.png" if ex["image"] is not None else None,
                "domain": ex.get("domain", ""),
                "topic": ex.get("topic", ""),
            }
            pid_splits[split].append(qid)
            if ex["image"] is not None:
                images.append((qid, ex["image"]))

    os.makedirs(os.path.join(args.data_out, "scienceqa"), exist_ok=True)
    with open(os.path.join(args.data_out, "scienceqa/problems.json"), "w") as f:
        json.dump(problems, f)
    with open(os.path.join(args.data_out, "scienceqa/pid_splits.json"), "w") as f:
        json.dump(pid_splits, f)
    print(
        f"problems: train={len(pid_splits['train'])} val={len(pid_splits['val'])} "
        f"test={len(pid_splits['test'])}, images={len(images)}"
    )

    # ---- ViT feature extraction (same model as cooelf/vision_features) ----
    os.makedirs(args.features_out, exist_ok=True)
    feat_path = os.path.join(args.features_out, "vit.npy")
    map_path = os.path.join(args.features_out, "name_map.json")
    if os.path.exists(feat_path) and os.path.exists(map_path):
        print("features already extracted, skipping")
        return

    device = torch.device(args.device)
    model = timm.create_model("vit_large_patch32_384", pretrained=True, num_classes=0)
    model.eval().to(device)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    name_map = {}
    features = np.zeros((len(images), 145, 1024), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(images), args.batch_size):
            chunk = images[start : start + args.batch_size]
            batch = torch.stack(
                [transform(img.convert("RGB")) for _, img in chunk]
            ).to(device)
            out = model.forward_features(batch)
            features[start : start + len(chunk)] = out.float().cpu().numpy()
            for offset, (qid, _) in enumerate(chunk):
                name_map[qid] = start + offset
            if start % (args.batch_size * 20) == 0:
                print(f"{start}/{len(images)}")

    np.save(feat_path, features)
    with open(map_path, "w") as f:
        json.dump(name_map, f)
    print(f"saved {features.shape} -> {feat_path}")


if __name__ == "__main__":
    main()
