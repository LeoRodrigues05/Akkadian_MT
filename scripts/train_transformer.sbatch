#!/bin/bash
#SBATCH --job-name=transformer-aug
#SBATCH --output=logs/transformer_%j.out
#SBATCH --error=logs/transformer_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -e
mkdir -p logs

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source /home/leo.rodrigues/Akkadian_MT/.venv/bin/activate
cd /home/leo.rodrigues/Akkadian_MT

python -u scripts/05_train_transformer.py \
    --train-csv scripts/data/augmented_train_split.csv \
    --val-csv   scripts/data/augmented_val_split.csv \
    --output-dir checkpoints/transformer-augmented

echo "Job finished: $(date)"
