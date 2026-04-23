# Akkadian → English Machine Translation

End-to-end pipeline for translating Akkadian transliterations into English. Includes
two character-level baselines (BiLSTM, Transformer) and ByT5-base / ByT5-large
fine-tuned with continued pre-training (CPT) and supervised fine-tuning (SFT) on an
augmented parallel corpus.

---

## 1. Results summary

All metrics are on the held-out validation split.

| Model                 | Trained on       | BLEU  | chrF++ | geo_mean |
|-----------------------|------------------|------:|-------:|---------:|
| Transformer (scratch) | aligned          |  8.55 |  28.06 |   15.49  |
| BiLSTM + attn         | aligned          | 17.45 |  36.17 |   25.12  |
| ByT5-base SFT         | augmented        | 33.27 |  49.75 |   40.68  |
| ByT5-large SFT        | augmented        |  see [checkpoints/byt5-large-sft/eval_results.json](checkpoints/byt5-large-sft/eval_results.json) |||

> The two baseline checkpoints in [checkpoints/bilstm/](checkpoints/bilstm/) and
> [checkpoints/transformer/](checkpoints/transformer/) were trained on the original
> `aligned_*` splits, **not** on the augmented data. To re-train them on the
> augmented corpus, see [§5](#5-train-baselines-on-the-augmented-data).

---

## 2. Prerequisites

- Linux / macOS, Python **3.10+**
- NVIDIA GPU strongly recommended:
  - BiLSTM / Transformer: any modern GPU with ≥8 GB VRAM
  - ByT5-base SFT: ≥16 GB VRAM
  - ByT5-large SFT: ≥40 GB VRAM (A100 / H100), or use gradient checkpointing
- ~10 GB free disk for data + checkpoints
- Optional: a Kaggle account + `~/.kaggle/kaggle.json` to re-download the raw corpus
- Optional: `OPENAI_API_KEY` for the LLM-based alignment step (`11_llm_alignment.py`)

---

## 3. Setup

```bash
git clone <repo-url> Akkadian_MT
cd Akkadian_MT

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The project root must be importable so the `preprocess` module resolves; running
all scripts from the repo root (as shown below) takes care of this.

---

## 4. Reproduce the data pipeline

The cleaned splits are already committed under [scripts/data/](scripts/data/). To
regenerate everything from scratch:

```bash
# 1. Pull the raw Kaggle dataset
python scripts/01_data_setup.py

# 2. Sentence-align the raw transliteration / translation pairs
python scripts/02_sentence_alignment.py

# 3-4. (optional) Mine extra published texts and integrate the eBL lexicon
python scripts/09_mine_published_texts.py
python scripts/10_lexicon_integration.py

# 5. (optional) LLM-assisted re-alignment of noisy pairs   (needs OPENAI_API_KEY)
python scripts/11_llm_alignment.py

# 6. Quality-filter the aligned pairs
python scripts/12_quality_filter.py

# 7. Build the augmented training set + 90/10 split
python scripts/14_data_augmentation.py
```

After step 7 you will have:

- `scripts/data/aligned_train_split.csv`, `aligned_val_split.csv` — clean baseline data
- `scripts/data/augmented_train_split.csv`, `augmented_val_split.csv` — augmented data used for the ByT5 runs

---

## 5. Train baselines on the augmented data

The baseline scripts default to `aligned_*_split.csv`; pass the augmented splits
explicitly to retrain on the new data. Use `--output-dir` to write to a fresh
folder so the existing `checkpoints/{bilstm,transformer}/` checkpoints aren't
overwritten.

```bash
# BiLSTM seq2seq (≈1–2 h on a single GPU)
python scripts/04_train_bilstm.py \
    --train-csv  scripts/data/augmented_train_split.csv \
    --val-csv    scripts/data/augmented_val_split.csv \
    --output-dir checkpoints/bilstm-augmented

# Vanilla Transformer (≈1–2 h on a single GPU)
python scripts/05_train_transformer.py \
    --train-csv  scripts/data/augmented_train_split.csv \
    --val-csv    scripts/data/augmented_val_split.csv \
    --output-dir checkpoints/transformer-augmented
```

Outputs (`best_model.pt`, `eval_results.json`) land in the directory passed via
`--output-dir`.

---

## 6. Train ByT5

### ByT5-base (SFT only)

```bash
python scripts/03_train_byt5.py
```

### ByT5-large (CPT + SFT or SFT only)

The dedicated script [scripts/15_train_byt5_large.py](scripts/15_train_byt5_large.py)
defaults to the augmented splits.

```bash
# SFT only (recommended — faster, strong results)
python scripts/15_train_byt5_large.py \
    --stage sft \
    --model google/byt5-large \
    --train-csv scripts/data/augmented_train_split.csv \
    --val-csv   scripts/data/augmented_val_split.csv \
    --sft-epochs 15 \
    --sft-lr 3e-4 \
    --batch-size 2 \
    --label-smoothing 0.1

# Two-stage: CPT on monolingual Akkadian, then SFT
python scripts/15_train_byt5_large.py --stage both --model google/byt5-large
```

For SLURM clusters use the provided batch file (edit the partition / account
lines first):

```bash
sbatch scripts/train_byt5_large.sbatch
```

The best checkpoint is written to `checkpoints/byt5-large-sft/best_model/`.

---

## 7. Evaluate & compare

```bash
# ByT5 (HF) checkpoint
python scripts/08_evaluate_byt5.py \
    --model-dir checkpoints/byt5-large-sft/best_model \
    --val-csv   scripts/data/augmented_val_split.csv

# BiLSTM / Transformer (.pt) checkpoints
python scripts/07_evaluate.py

# Side-by-side comparison across all available checkpoints
python scripts/compare_models.py
```

For a higher-quality (slower) decoding pass over the test set:

```bash
python scripts/16_mbr_decode.py \
    --model-dir checkpoints/byt5-large-sft/best_model
```

---

## 8. Repository layout

```
scripts/
  01_data_setup.py … 16_mbr_decode.py   # numbered pipeline steps
  train_byt5_large.sbatch               # SLURM job for ByT5-large
  data/                                 # cleaned CSV splits (committed)
checkpoints/
  bilstm/  transformer/                 # baseline .pt + eval_results.json
  byt5-base/  byt5-base-sft/  byt5-large-sft/
documentation/                          # design notes & action plans
kaggle_notebook.ipynb                   # inference / submission notebook
preprocess.py                           # shared text-cleaning helpers
requirements.txt
```

---

## 9. Troubleshooting

- **`ModuleNotFoundError: preprocess`** — run scripts from the repo root.
- **CUDA OOM on ByT5-large** — lower `--batch-size`, keep gradient checkpointing on,
  or switch to `--stage sft` only.
- **`kagglehub` auth error** in `01_data_setup.py` — place valid Kaggle credentials
  at `~/.kaggle/kaggle.json` (`chmod 600`).
- **Existing baseline checkpoints overwritten** — both baseline scripts write to a
  fixed `checkpoints/<model>/` directory; copy `best_model.pt` aside before
  re-training on a new dataset.
