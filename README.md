# mmcot-fusion

A faithful reimplementation of Multimodal Chain-of-Thought
(MM-CoT, [Zhang et al., TMLR 2024](https://arxiv.org/abs/2302.00923)) with a
pluggable cross-modal fusion interface, used for a controlled re-examination
of what actually drives its performance.

## TL;DR of the study

Starting from a faithful reproduction (ScienceQA 85.95 with the official
checkpoint, 85.48 retrained, vs. 85.31 in the paper; M3CoT 44.22 vs. 44.85),
we tested the two most natural improvement hypotheses under a controlled,
same-budget protocol (20 epochs, identical seed/data, single H100):

1. Better fusion? No. A 2x2 grid over gate design (sigmoid vs. zero-init
   tanh) x attention heads (1 vs. 8) yields no significant differences
   (McNemar p >= 0.24 on ScienceQA; p >= 0.46 fine-tuned on M3CoT). Learned
   tanh gates stay nearly closed (|tanh(alpha)| <= 0.08) — the model itself
   declines the visual signal, and zeroing image features costs only 2.6 pt.
2. Better teacher rationales? No. Replacing the human rationales with
   frontier-VLM-generated ones significantly hurts accuracy.

Why: ScienceQA's rationale supervision is template retrieval — the dataset
contains only 255 unique lectures, and 99.9% of test-question lectures appear
verbatim in the training split. Stage-1 "rationale generation" memorises
templates (training loss 0.005 vs. 0.14 with diverse teacher rationales), and
free-form teacher text breaks that shortcut. Out-of-domain (M3CoT zero-shot),
the in-domain advantage of template-fit rationales largely disappears.

Practical takeaway: before improving the architecture or the supervision of a
multimodal CoT system, verify the benchmark actually exercises them.

## Fusion interface

| fusion key   | description |
|--------------|-------------|
| `sigmoid_1h` | original MM-CoT: single-head cross-attention + position-wise sigmoid gate |
| `sigmoid_mh` | 8 heads, sigmoid gate |
| `tanh_1h`    | single head, zero-init tanh gate |
| `tanh_mh`    | 8 heads + zero-initialised scalar tanh gate (Flamingo-style) |

```python
from mmcot_fusion import T5ForMultimodalGeneration

model = T5ForMultimodalGeneration.from_pretrained(
    "declare-lab/flan-alpaca-base", patch_size=(145, 1024), fusion="tanh_mh"
)
```

## Reproduce

Everything below runs with `uv` (Python 3.11 is provisioned automatically;
all dependency versions are pinned in `uv.lock`, including
`transformers==4.30.0`).

```bash
# 0. Environment and tests
uv sync
uv run pytest tests/ -q                    # 14 unit tests

# 1. Data, vision features, official checkpoints (~10 GB)
uv run mmcot-fusion download               # ScienceQA + DETR features + base checkpoints
# ViT features: the distributed vit.npy is INCOMPATIBLE with the released
# checkpoint (see docs/REPRODUCING.md, lesson 1). Convert vit.pth instead:
uv run python - <<'PY'
import numpy as np, torch
from huggingface_hub import hf_hub_download
t = torch.load(hf_hub_download("cooelf/vision_features", "vit.pth"),
               map_location="cpu", mmap=True)
out = np.lib.format.open_memmap("vision_features/vit.npy", mode="w+",
                                dtype=np.float32, shape=tuple(t.shape))
for i in range(0, t.shape[0], 512):
    out[i:i+512] = t[i:i+512].float().numpy()
PY

# 2. Checkpoint-level reproduction (expects 85.95 on the ScienceQA test set)
bash scripts/e0_eval_official_ckpt.sh

# 3. Controlled retraining (full official recipe, 20 epochs; ~2.3 h on one H100)
uv run mmcot-fusion run --user-msg baseline --fusion sigmoid_1h \
  --model declare-lab/flan-alpaca-base --img-type vit \
  --caption-file data/instruct_captions.json \
  --prompt-format QCM-E --output-len 512 --lr 8e-5 --epoch 20 --bs 8 \
  --skip-val-generation
# then the answer stage with --prompt-format QCMG-A --output-len 64 and
# --test-le <rationale_run>/predictions_ans_test.json
# (PBS job scripts for all runs are under scripts/hpc/)

# 4. Teacher rationales for the distillation run (public GPT-4o data)
uv run python scripts/map_share4o_rationales.py
# train with --teacher-rationales teacher_rationales/share4o_gpt4o_train.jsonl
```

Analyses (McNemar tests, image-blind evaluation, M3CoT transfer) and the
compute environments used (one NVIDIA H100 for training; an Apple M1 Pro
laptop suffices for the checkpoint evaluation) are documented in
`docs/REPRODUCING.md`, together with the pitfalls we hit along the way.

Teacher rationales come from the public
[Share4oReasoning/sft_data](https://huggingface.co/datasets/Share4oReasoning/sft_data)
dataset released by LLaVA-Reasoner (GPT-4o generated, Apache-2.0);
`scripts/map_share4o_rationales.py` maps them onto ScienceQA question ids
and strips the answer verdicts.

## Results

The `results/` directory contains the raw evaluation artifacts (per-question
predictions, correctness maps, training logs, run configs) for every number in
the report, with a mapping table in `results/README.md`. Pairwise statistics
can be recomputed with `scripts/aggregate_results.py`.

## Attribution

- Model architecture, prompt construction, data handling, and evaluation
  protocol are adapted from
  [amazon-science/mm-cot](https://github.com/amazon-science/mm-cot)
  (Apache-2.0, Zhang et al., TMLR 2024); per-file attribution is in the
  source headers and `NOTICE`.
- ScienceQA (Lu et al., NeurIPS 2022) is used under CC BY-NC-SA 4.0.
- M3CoT (Chen et al., ACL 2024) is obtained via HuggingFace
  (`LightChen2333/M3CoT`).
- Teacher rationales: Share4oReasoning/sft_data (Zhang et al., ACL 2025,
  Apache-2.0), used in accordance with its published license.

## License

Apache-2.0
