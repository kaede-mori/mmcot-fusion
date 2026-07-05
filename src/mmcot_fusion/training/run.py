"""Two-stage MM-CoT training / inference pipeline.

Adapted from https://github.com/amazon-science/mm-cot (Apache-2.0)
(``main.py``), refactored into a callable API with a pluggable fusion
module, MPS/CUDA/CPU support and JSON result artefacts that include
per-question correctness (for paired significance testing).

Stage 1 (rationale generation): prompt ``QCM-LE``, output_len 512.
Stage 2 (answer inference):     prompt ``QCMG-A``, output_len 64, consuming
the rationales predicted by stage 1 for val/test.

Note on the evaluation protocol: the original repository defaults to
teacher-forced argmax decoding for prediction, which materialises the full
logit tensor (impractically large on this scale) and leaks gold prefixes.
We therefore always evaluate with free-running generation
(``predict_with_generate=True``).
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import re
import time
from typing import Optional

import numpy as np
import torch
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
)

from ..data.scienceqa import IMG_SHAPE, ScienceQADatasetImg, load_image_features, load_scienceqa
from ..evaluation.metrics import extract_answer, get_scores
from ..models.t5_multimodal import T5ForMultimodalGeneration

DEFAULT_OPTIONS = ["A", "B", "C", "D", "E", "F", "G", "H"]


class MultimodalDataCollator(DataCollatorForSeq2Seq):
    """DataCollatorForSeq2Seq that stacks the extra ``image_ids`` feature.

    ``tokenizer.pad`` (transformers >=4.30) cannot convert custom float
    tensor features, so the image features are stacked separately.
    """

    def __call__(self, features, return_tensors=None):
        image_ids = torch.stack(
            [feature.pop("image_ids") for feature in features]
        )
        batch = super().__call__(features, return_tensors=return_tensors)
        batch["image_ids"] = image_ids
        return batch


@dataclasses.dataclass
class RunConfig:
    # paths
    data_root: str = "data"
    features_root: str = "vision_features"
    output_dir: str = "experiments"
    caption_file: Optional[str] = None
    # model
    model: str = "allenai/unifiedqa-t5-base"
    fusion: str = "sigmoid_1h"
    img_type: str = "detr"
    # stage
    prompt_format: str = "QCM-LE"  # QCM-LE (rationale) or QCMG-A (answer)
    input_len: int = 512
    output_len: int = 512
    eval_le: Optional[str] = None
    test_le: Optional[str] = None
    # optimisation
    epoch: int = 20
    lr: float = 5e-5
    bs: int = 8
    eval_bs: int = 8
    grad_accum: int = 2
    seed: int = 42
    # control
    evaluate_dir: Optional[str] = None  # skip training, evaluate this checkpoint
    orig_ckpt: bool = False  # evaluate_dir holds an official MM-CoT release
    user_msg: str = "run"
    train_subset: Optional[int] = None  # stratified subsampling fallback
    gradient_checkpointing: bool = False
    blind: bool = False  # zero out image features (image-blind analysis)
    # JSONL of {"qid", "rationale"}: replaces the human rationale as the
    # training target (distillation from a frontier VLM teacher)
    teacher_rationales: Optional[str] = None
    # skip rationale generation for the val split (only needed when a later
    # stage evaluates on val, which our final-eval protocol does not)
    skip_val_generation: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_save_dir(cfg: RunConfig) -> str:
    if cfg.evaluate_dir is not None:
        save_dir = os.path.join(
            cfg.output_dir,
            f"eval_{cfg.user_msg}_{os.path.basename(os.path.normpath(cfg.evaluate_dir))}_{cfg.prompt_format}",
        )
    else:
        model_name = cfg.model.replace("/", "-")
        save_dir = (
            f"{cfg.output_dir}/{cfg.user_msg}_{model_name}_{cfg.img_type}_{cfg.fusion}"
            f"_{cfg.prompt_format}_lr{cfg.lr}_bs{cfg.bs * cfg.grad_accum}"
            f"_op{cfg.output_len}_ep{cfg.epoch}_seed{cfg.seed}"
        )
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def build_model(cfg: RunConfig):
    patch_size = IMG_SHAPE[cfg.img_type]
    if cfg.evaluate_dir is not None:
        if cfg.orig_ckpt:
            model = T5ForMultimodalGeneration.from_mmcot_checkpoint(
                cfg.evaluate_dir, patch_size=patch_size
            )
        else:
            model = T5ForMultimodalGeneration.from_pretrained(
                cfg.evaluate_dir, patch_size=patch_size, fusion=cfg.fusion
            )
            _verify_fusion_matches_checkpoint(model, cfg.evaluate_dir, cfg.fusion)
        tokenizer = AutoTokenizer.from_pretrained(cfg.evaluate_dir)
    else:
        model = T5ForMultimodalGeneration.from_pretrained(
            cfg.model, patch_size=patch_size, fusion=cfg.fusion
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    return model, tokenizer


def _verify_fusion_matches_checkpoint(model, checkpoint_dir, fusion):
    """Fail fast if --fusion does not match the trained checkpoint.

    from_pretrained silently random-initialises missing fusion parameters and
    drops unexpected ones, which produces garbage evaluations."""
    import os

    state = torch.load(
        os.path.join(checkpoint_dir, "pytorch_model.bin"), map_location="meta", weights_only=True
    )
    ckpt_keys = {k for k in state if k.startswith("fusion.")}
    model_keys = {f"fusion.{k}" for k, _ in model.fusion.named_parameters()}
    if ckpt_keys != model_keys:
        raise ValueError(
            f"--fusion {fusion} does not match checkpoint {checkpoint_dir}: "
            f"checkpoint fusion keys {sorted(ckpt_keys)} vs model {sorted(model_keys)}"
        )


def _subset(qids, n, problems, seed):
    """Stratified (by subject) subsample of training question ids."""
    if n is None or n >= len(qids):
        return qids
    rng = random.Random(seed)
    by_subject = {}
    for qid in qids:
        by_subject.setdefault(problems[qid]["subject"], []).append(qid)
    frac = n / len(qids)
    subset = []
    for subject_qids in by_subject.values():
        k = max(1, round(len(subject_qids) * frac))
        subset.extend(rng.sample(subject_qids, k))
    rng.shuffle(subset)
    return subset[:n]


def _decode(tokenizer, predict_results):
    """Decode generated ids and label ids, replacing the -100 padding that
    the evaluation loop inserts when concatenating variable-length batches."""
    preds_ids = np.where(
        predict_results.predictions != -100, predict_results.predictions, tokenizer.pad_token_id
    )
    label_ids = np.where(
        predict_results.label_ids != -100, predict_results.label_ids, tokenizer.pad_token_id
    )
    preds = tokenizer.batch_decode(preds_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    targets = tokenizer.batch_decode(label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return preds, targets


def run(cfg: RunConfig) -> dict:
    set_seed(cfg.seed)
    save_dir = make_save_dir(cfg)
    with open(os.path.join(save_dir, "run_config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    problems, qids = load_scienceqa(cfg.data_root, cfg.caption_file)
    if cfg.teacher_rationales:
        # the trailing verdict sentence is stripped so stage 1 remains a pure
        # rationale generator (the answer must not leak into stage-2 input)
        verdict = re.compile(r"\s*The answer is \([A-Z]\)\.?\s*$")
        n = 0
        with open(cfg.teacher_rationales) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec["qid"] in problems:
                    problems[rec["qid"]]["lecture"] = ""
                    problems[rec["qid"]]["solution"] = verdict.sub("", rec["rationale"]).strip()
                    n += 1
        print(f"teacher rationales applied to {n} problems")
    name_maps, image_features = load_image_features(cfg.features_root, cfg.img_type)

    train_qids = _subset(qids["train"], cfg.train_subset, problems, cfg.seed)
    val_qids, test_qids = qids["val"], qids["test"]
    print(f"problems: train={len(train_qids)} val={len(val_qids)} test={len(test_qids)}")

    model, tokenizer = build_model(cfg)
    print("model parameters:", model.num_parameters())
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    common = dict(
        problems=problems,
        name_maps=name_maps,
        tokenizer=tokenizer,
        source_len=cfg.input_len,
        target_len=cfg.output_len,
        image_features=image_features,
        img_type=cfg.img_type,
        prompt_format=cfg.prompt_format,
        use_caption=cfg.caption_file is not None,
    )
    train_set = ScienceQADatasetImg(qids=train_qids, **common)
    eval_set = ScienceQADatasetImg(qids=val_qids, test_le=cfg.eval_le, **common)
    test_set = ScienceQADatasetImg(qids=test_qids, test_le=cfg.test_le, **common)
    if cfg.blind:
        # image-blind analysis: every question falls back to zero features
        for ds in (train_set, eval_set, test_set):
            ds.feature_index = [-1] * len(ds.feature_index)

    datacollator = MultimodalDataCollator(tokenizer)

    training_args = Seq2SeqTrainingArguments(
        save_dir,
        do_train=cfg.evaluate_dir is None,
        do_eval=False,
        evaluation_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.bs,
        per_device_eval_batch_size=cfg.eval_bs,
        gradient_accumulation_steps=cfg.grad_accum,
        weight_decay=0.01,
        num_train_epochs=cfg.epoch,
        predict_with_generate=True,
        generation_max_length=cfg.output_len,
        report_to="none",
        seed=cfg.seed,
        # transformers 4.30 + accelerate: the eval dataloader moves batches to
        # the accelerate device (mps) while the model stays on args.device;
        # setting this aligns both on mps
        use_mps_device=torch.backends.mps.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=datacollator,
        tokenizer=tokenizer,
    )

    if cfg.evaluate_dir is None:
        start = time.time()
        trainer.train()
        model.config.use_cache = True  # re-enable KV cache for generation
        trainer.save_model(save_dir)
        with open(os.path.join(save_dir, "train_log.json"), "w") as f:
            json.dump(
                {"train_seconds": time.time() - start, "log_history": trainer.state.log_history},
                f,
                indent=2,
            )

    # ---- test-set prediction (free-running generation) ----
    predict_results = trainer.predict(test_dataset=test_set, max_length=cfg.output_len)
    preds, targets = _decode(tokenizer, predict_results)

    output = {"save_dir": save_dir}
    if cfg.prompt_format.endswith("-A"):
        results_ans, results_rationale, results_reference = {}, {}, {}
        per_question = {}
        num_fail = 0
        rng = random.Random(cfg.seed)
        options = DEFAULT_OPTIONS
        for idx, qid in enumerate(test_qids):
            pred, ref = preds[idx], targets[idx]
            extracted = extract_answer(pred)
            if extracted != "FAILED" and extracted in options:
                pred_idx = options.index(extracted)
            else:
                num_fail += 1
                pred_idx = rng.choice(range(len(options)))
            results_ans[str(qid)] = pred_idx
            results_rationale[str(qid)] = pred
            results_reference[str(qid)] = ref
            per_question[str(qid)] = bool(pred_idx == problems[qid]["answer"])

        accuracy = 100.0 * sum(per_question.values()) / len(per_question)
        try:
            # detailed per-subject breakdown (ScienceQA schema only)
            scores = get_scores(
                results_ans,
                results_rationale,
                results_reference,
                os.path.join(cfg.data_root, "scienceqa/problems.json"),
            )
        except (KeyError, FileNotFoundError):
            scores = {"answer": {"acc_average": "{:.2f}".format(accuracy)}}
        output.update(
            {
                "num_fail": num_fail,
                "scores": scores,
                "per_question_correct": per_question,
                "preds": [p.strip() for p in preds],
                "labels": targets,
            }
        )
        out_file = os.path.join(save_dir, "predictions_ans_test.json")
        print("test scores:", json.dumps(scores["answer"], indent=2))
    else:
        output.update({"preds": [p.strip() for p in preds], "labels": targets})
        out_file = os.path.join(save_dir, "predictions_ans_test.json")

    with open(out_file, "w") as f:
        json.dump(output, f, indent=4)

    # rationale stage also generates rationales for the val set (consumed by stage 2)
    if cfg.prompt_format in ("QCM-LE", "QCM-E") and not cfg.skip_val_generation:
        predict_results = trainer.predict(test_dataset=eval_set, max_length=cfg.output_len)
        preds, targets = _decode(tokenizer, predict_results)
        eval_output = {"preds": [p.strip() for p in preds], "labels": targets}
        with open(os.path.join(save_dir, "predictions_ans_eval.json"), "w") as f:
            json.dump(eval_output, f, indent=4)

    return output
