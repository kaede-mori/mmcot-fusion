#!/bin/sh
#PBS -q eg
#PBS -l select=1:ngpus=1
#PBS -l walltime=00:30:00
#PBS -W group_list=YOUR_GROUP
#PBS -j oe

cd ${PBS_O_WORKDIR}
export PATH=$HOME/.local/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,memory.total --format=csv
.venv/bin/python -u scripts/hpc/smoke_gpu.py
