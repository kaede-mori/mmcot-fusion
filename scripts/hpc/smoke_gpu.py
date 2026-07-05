"""GPU smoke test for the eg (trial) queue: verifies the full pipeline
(dataset, collator, fusion model, Trainer train step, generation) on CUDA
with a tiny subset. Target runtime: under 10 minutes on a 1/7 MIG slice.
"""

import os
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments

from mmcot_fusion.data.scienceqa import ScienceQADatasetImg, load_image_features, load_scienceqa
from mmcot_fusion.models.t5_multimodal import T5ForMultimodalGeneration
from mmcot_fusion.training.run import MultimodalDataCollator

assert torch.cuda.is_available(), "CUDA not available"
print("GPU:", torch.cuda.get_device_name(0))

problems, qids = load_scienceqa("data", "data/instruct_captions.json")
name_maps, feats = load_image_features("vision_features", "vit")
tok = AutoTokenizer.from_pretrained("declare-lab/flan-alpaca-base")

train_ds = ScienceQADatasetImg(
    problems, qids["train"][:16], name_maps, tok, 512, 512, feats, "vit", "QCM-E", use_caption=True
)
test_ds = ScienceQADatasetImg(
    problems, qids["test"][:8], name_maps, tok, 512, 512, feats, "vit", "QCM-E", use_caption=True
)

for fusion in ("sigmoid_1h", "tanh_mh"):
    model = T5ForMultimodalGeneration.from_pretrained(
        "declare-lab/flan-alpaca-base", patch_size=(145, 1024), fusion=fusion
    )
    args = Seq2SeqTrainingArguments(
        f"/tmp/smoke_{fusion}",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        learning_rate=8e-5,
        predict_with_generate=True,
        generation_max_length=512,
        report_to="none",
        save_strategy="no",
        logging_steps=4,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=MultimodalDataCollator(tok),
        tokenizer=tok,
    )
    t0 = time.time()
    trainer.train()
    t_train = time.time() - t0
    t0 = time.time()
    res = trainer.predict(test_ds, max_length=512)
    t_gen = time.time() - t0
    import numpy as np

    ids = np.where(res.predictions != -100, res.predictions, tok.pad_token_id)
    sample = tok.decode(ids[0], skip_special_tokens=True)
    print(
        f"[{fusion}] train 8 microsteps: {t_train:.1f}s ({t_train/8:.2f}s/step bs8) | "
        f"gen 16 samples: {t_gen:.1f}s | sample: {sample[:80]!r}"
    )

print("SMOKE_OK")
