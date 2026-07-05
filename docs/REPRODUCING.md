# Reproducing the experiments

Complete command list for reproducing every result in the report. See the
README for the quick-start version; this file adds the analysis commands,
the compute environments, and the pitfalls we hit along the way.

## 1. Setup

```bash
uv sync                      # Python 3.11 venv, pinned deps (transformers==4.30.0)
uv run pytest tests/ -q      # 14 unit tests

# ScienceQA json + DETR features + official base checkpoints (~2.5 GB)
uv run mmcot-fusion download
# Image captions: data/instruct_captions.json from the official
# amazon-science/mm-cot repository (data/ directory).

# ViT features: convert vit.pth (the distributed vit.npy is incompatible
# with the released checkpoint — see Pitfalls #1):
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
```

## 2. Checkpoint-level reproduction

```bash
bash scripts/e0_eval_official_ckpt.sh
# expected: 85.95 average accuracy on the 4,241-question ScienceQA test set
```

## 3. Controlled retraining grid (fusion 2x2)

One run per fusion variant, all under the full official recipe
(20 epochs, effective batch 8, lr 8e-5, seed 42, fp32):

```bash
for FUSION in sigmoid_1h sigmoid_mh tanh_1h tanh_mh; do
  qsub -N $FUSION -v TAG=$FUSION,FUSION=$FUSION scripts/hpc/job_run.sh
done
# scripts/hpc/job_run.sh runs both stages (QCM->E rationale, QCMG->A answer).
# On a workstation, run the two `mmcot-fusion run` commands inside it directly.
```

## 4. Teacher-rationale distillation

```bash
uv run python scripts/map_share4o_rationales.py   # public GPT-4o rationales
qsub -N distill -v TAG=distill,FUSION=sigmoid_1h,TEACHER=teacher_rationales/share4o_gpt4o_train.jsonl scripts/hpc/job_run.sh
```

## 5. Analyses

```bash
# Aggregation + paired McNemar tests
uv run python scripts/aggregate_results.py \
  base=experiments/<baseline_answer_dir> variant=experiments/<variant_answer_dir>

# Image-blind evaluation (zeroed vision features at inference)
uv run mmcot-fusion run --user-msg blind --evaluate-dir <rationale_run> --blind \
  --fusion sigmoid_1h --img-type vit --caption-file data/instruct_captions.json \
  --prompt-format QCM-E --output-len 512 --skip-val-generation
# then the answer stage with --blind and --test-le on the blind rationales.

# M3CoT: prepare data/features, then fine-tune or transfer-evaluate
uv run --group experiments python scripts/prepare_m3cot.py
qsub -N m3cot -v TAG=m3cot,FUSION=sigmoid_1h scripts/hpc/job_m3cot.sh

# Loss-curve figure
uv run --group dev python scripts/make_loss_figure.py
```

## Compute environments

| | Checkpoint reproduction | Controlled grids |
|---|---|---|
| Hardware | Apple M1 Pro laptop (16 GB) | 1x NVIDIA H100 (80 GB) |
| Backend | MPS, fp32 | CUDA 13.0, fp32 |
| Python / torch / transformers | 3.11 / 2.12.1 / 4.30.0 (pinned in uv.lock) | same |
| Throughput | — | ~4-6 it/s at batch 8; one full two-stage run in ~4 h |

Total compute for all reported experiments: roughly 35 GPU-hours.

## Pitfalls

1. The distributed `vit.npy` has shape (11208, 577, 768) in fp16 and is
   incompatible with the released checkpoint, whose image projection expects
   768x1024 inputs. The features that match the checkpoint are in `vit.pth`
   (11208, 145, 1024, fp32). Always convert `vit.pth`.
2. The original repository's default evaluation uses teacher-forced argmax
   decoding, which both leaks the gold rationale prefix and would require
   holding ~280 GB of logits at ScienceQA scale. We always evaluate with
   free-running generation (`predict_with_generate`).
3. transformers 4.30 compatibility: the stock `DataCollatorForSeq2Seq`
   cannot handle the custom `image_ids` field (we stack it separately), and
   T5's `prepare_inputs_for_generation` silently drops custom inputs during
   generation (we override it).
4. Generation output ids are padded with -100 when the evaluation loop
   concatenates variable-length batches; replace with the pad token id
   before decoding.
5. On Lustre filesystems, install packages with `UV_LINK_MODE=copy`
   (hardlink installs can silently lose files), and do not let compute
   nodes re-sync the environment — call `.venv/bin/...` directly.
6. Training on a 16 GB MPS machine requires batch 2 + gradient accumulation
   + gradient checkpointing, and contention with other applications can
   double step time. The H100 path is strongly recommended for training.
