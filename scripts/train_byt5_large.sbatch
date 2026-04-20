#!/bin/bash
#SBATCH --job-name=byt5-large-sft
#SBATCH --output=logs/byt5_large_%j.out
#SBATCH --error=logs/byt5_large_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

# ── Adjust these to your cluster ──────────────────────────────────────────
# #SBATCH --partition=gpu          # uncomment and set your partition
# #SBATCH --account=your-account   # uncomment if needed

set -e

# Create log directory
mkdir -p logs

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate leo-akkadian-1

cd /home/leo.rodrigues/Akkadian_MT

# ── Option A: SFT only (faster, ~6-10 hours) ─────────────────────────────
# Train ByT5-large directly on augmented parallel data
python -u scripts/15_train_byt5_large.py \
    --stage sft \
    --model google/byt5-large \
    --train-csv scripts/data/augmented_train_split.csv \
    --val-csv scripts/data/augmented_val_split.csv \
    --sft-epochs 15 \
    --sft-lr 3e-4 \
    --batch-size 2 \
    --label-smoothing 0.1

# ── Option B: CPT + SFT (longer, ~12-20 hours, potentially better) ───────
# Uncomment below and comment out Option A to use two-stage training
# python -u scripts/15_train_byt5_large.py \
#     --stage both \
#     --model google/byt5-large \
#     --train-csv scripts/data/augmented_train_split.csv \
#     --val-csv scripts/data/augmented_val_split.csv \
#     --cpt-epochs 5 \
#     --sft-epochs 15 \
#     --cpt-lr 1e-4 \
#     --sft-lr 3e-4 \
#     --batch-size 2 \
#     --label-smoothing 0.1

echo "Job finished: $(date)"
