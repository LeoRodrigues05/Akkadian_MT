#!/bin/bash
#SBATCH --job-name=bilstm-akk
#SBATCH --output=logs/bilstm_%j.out
#SBATCH --error=logs/bilstm_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -e
mkdir -p logs

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

conda activate leo-akkadian-1
cd /home/leo.rodrigues/Akkadian_MT

python -u scripts/04_train_bilstm.py \
    --train-csv scripts/data/augmented_train_split.csv \
    --val-csv   scripts/data/augmented_val_split.csv \
    --output-dir checkpoints/bilstm

echo "Job finished: $(date)"
