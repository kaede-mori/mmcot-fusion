#!/bin/sh
#PBS -q sg
#PBS -l select=1:ngpus=1
#PBS -l walltime=06:00:00
#PBS -W group_list=YOUR_GROUP
#PBS -j oe

# M3CoT fine-tuning with a given fusion. qsub -v TAG=m1,FUSION=sigmoid_1h
cd ${PBS_O_WORKDIR}
export PATH=$HOME/.local/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
EPOCH=${EPOCH:-20}
SEED=${SEED:-42}
set -e

.venv/bin/mmcot-fusion run \
  --user-msg ${TAG}-rationale --fusion ${FUSION} \
  --model declare-lab/flan-alpaca-base --img-type vit \
  --data-root data_m3cot --features-root vision_features_m3cot \
  --prompt-format QCM-E --output-len 512 --lr 8e-5 --skip-val-generation \
  --epoch ${EPOCH} --seed ${SEED} --bs 8 --grad-accum 1 --eval-bs 32

RAT_DIR=$(ls -dt experiments/${TAG}-rationale_*ep${EPOCH}_seed${SEED} | head -1)

.venv/bin/mmcot-fusion run \
  --user-msg ${TAG}-answer --fusion ${FUSION} \
  --model declare-lab/flan-alpaca-base --img-type vit \
  --data-root data_m3cot --features-root vision_features_m3cot \
  --prompt-format QCMG-A --output-len 64 --lr 8e-5 \
  --epoch ${EPOCH} --seed ${SEED} --bs 8 --grad-accum 1 --eval-bs 32 \
  --test-le "${RAT_DIR}/predictions_ans_test.json"
echo "RUN_${TAG}_DONE"
