# Akkadian MT Pipeline — Script Reference

## Execution Order

```
01_data_setup.py → 02_sentence_alignment.py → [03 / 04 / 05 in parallel] → 06_inference.py → 07_evaluate.py
```

---

## Data Pipeline (run once, in order)

### 01_data_setup.py
**Purpose:** Downloads competition data via `kagglehub`, discovers all CSVs, copies them into a local `data/` folder, then prints shape/columns/samples of each file for exploration.
- **Run:** First, run once.
- **Input:** Kaggle API credentials.
- **Output:** `data/` directory with `train.csv`, `test.csv`, `sample_submission.csv`, `Sentences_Oare_FirstWord_LinNum.csv`, and any other competition CSVs.

### 02_sentence_alignment.py
**Purpose:** The critical data step. `train.csv` has *document-level* pairs (full tablet transliterations paired with full English translations), but the test set is *sentence-level*. This script splits each document into sentence-level (Akkadian, English) pairs using `Sentences_Oare_FirstWord_LinNum.csv` line numbers + English sentence boundary detection (`". " + capital letter`). Applies `preprocess.py` cleaning to both sides, then saves train/val splits.
- **Run:** After 01, before any training script.
- **Input:** `data/train.csv`, `data/Sentences_Oare_FirstWord_LinNum.csv`.
- **Output:**
  - `data/aligned_train.csv` — all aligned pairs
  - `data/aligned_train_split.csv` — 90% training split
  - `data/aligned_val_split.csv` — 10% validation split

---

## Model Training (run independently; each requires aligned data from step 02)

### 03_train_byt5.py
**Purpose:** **Primary baseline.** Fine-tunes `google/byt5-base` (pretrained byte-level seq2seq model) using HuggingFace `Seq2SeqTrainer`. ByT5 operates on raw bytes — no custom tokenizer needed. This is the model intended for Kaggle submission.
- **Config:** 15 epochs, lr=3e-4, batch_size=16, gradient_accumulation=4, Adafactor optimizer, fp16.
- **Metrics:** Computes BLEU, chrF++, and geometric mean on the validation set using sacrebleu.
- **Input:** `data/aligned_train_split.csv`, `data/aligned_val_split.csv`.
- **Output:** `checkpoints/byt5-base/best_model/` (model + tokenizer), `checkpoints/byt5-base/eval_results.json`.
- **Requirements:** GPU with sufficient VRAM (fp16 enabled).

### 04_train_bilstm.py
**Purpose:** **Academic comparison only.** Trains a character-level BiLSTM encoder (2-layer, hidden=512, embedding=256) + LSTM decoder with Bahdanau attention from scratch. Uses teacher forcing (ratio=0.5). Expected to score poorly (~5–15 on the competition metric) — exists to demonstrate that pretraining matters for low-resource MT.
- **Config:** 50 epochs, Adam optimizer, lr=1e-3, batch_size=64.
- **Input:** `data/aligned_train_split.csv`, `data/aligned_val_split.csv`.
- **Output:** `checkpoints/bilstm/best_model.pt`, `checkpoints/bilstm/eval_results.json`.

### 05_train_transformer.py
**Purpose:** **Academic comparison only.** Trains a small vanilla Transformer (4 layers, 4 heads, d_model=256, d_ff=512) from scratch with character-level tokenization and a warmup learning rate schedule. Expected score ~8–18. Same purpose as BiLSTM — shows that architecture alone isn't enough without pretraining on a low-resource language.
- **Config:** 50 epochs, Adam (β₁=0.9, β₂=0.98), warmup=2000 steps, batch_size=64.
- **Input:** `data/aligned_train_split.csv`, `data/aligned_val_split.csv`.
- **Output:** `checkpoints/transformer/best_model.pt`, `checkpoints/transformer/eval_results.json`.

---

## Inference & Evaluation

### 06_inference.py
**Purpose:** Loads the best ByT5 checkpoint, reads `test.csv`, applies `clean_transliteration()` to each input, runs beam search decoding (num_beams=4, max_length=512), and writes predictions to `submission.csv`. Works locally for testing. Paths are configurable via environment variables (`MODEL_PATH`, `TEST_CSV`, `OUTPUT_CSV`).
- **Input:** `checkpoints/byt5-base/best_model/`, `data/test.csv`.
- **Output:** `submission.csv` with columns `id,translation`.
- **Note:** The local `test.csv` is dummy data; real test data only appears during Kaggle notebook execution.

### 07_evaluate.py
**Purpose:** Reads the `eval_results.json` from all three model checkpoint directories and prints a formatted comparison table showing BLEU, chrF++, geometric mean, and parameter count for each baseline. No training or inference — purely aggregation and display.
- **Input:** `checkpoints/*/eval_results.json`.
- **Output:** `results_comparison.json`, printed table to stdout.

---

## Standalone Modules

### preprocess.py
**Purpose:** **Shared library** imported by `02_sentence_alignment.py`, `06_inference.py`, and the kaggle submission script. Contains two functions:

- **`clean_transliteration(text)`** — Normalizes Akkadian transliteration strings:
  - Unicode NFKC normalization
  - Ḫ/ḫ → H/h conversion (competition convention)
  - Subscript digits ₀–₉ → 0–9
  - Determinative normalization ({d}, {ki}, {m}, etc.)
  - Gap standardization ([x x], [...], … → `<gap>`)
  - Scribal notation removal (!, ?, *, #, half-brackets ˹˺, square brackets)
  - Alternate romanization: sz→š, s,→ṣ, t,→ṭ
  - Accented vowel index notation: a2→á, a3→à, e2→é, etc.

- **`clean_translation(text)`** — Normalizes English translation strings:
  - Scholarly annotation removal: (fem. plur.), (sic), (lit.), etc.
  - Bracket cleaning
  - Gap marker alignment
  - Fraction normalization: 1/2→½, 1/3→⅓, etc.
  - Roman numeral month conversion: Month XII → Month 12

Has a `__main__` block with test cases for manual verification.

### kaggle_submission.py
**Purpose:** **Self-contained Kaggle notebook version** of `06_inference.py`. Duplicates the preprocessing logic inline (no imports from `preprocess.py`) so it works as a single file on Kaggle with no internet access. Points to Kaggle-specific paths (`/kaggle/input/...`). This is what you paste into a Kaggle notebook for actual competition submission.

- **Kaggle setup:** Upload `checkpoints/byt5-base/best_model/` as a Kaggle dataset, then reference it.
- **Output:** `/kaggle/working/submission.csv`.

---

## Folder Structure

```
Akkadian_MT/
├── data/                        # Created by 01_data_setup.py
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   ├── Sentences_Oare_FirstWord_LinNum.csv
│   ├── aligned_train.csv        # Created by 02
│   ├── aligned_train_split.csv  # Created by 02
│   └── aligned_val_split.csv    # Created by 02
├── checkpoints/                 # Created by training scripts
│   ├── byt5-base/
│   ├── bilstm/
│   └── transformer/
├── scripts/
│   ├── 01_data_setup.py
│   ├── 02_sentence_alignment.py
│   ├── 03_train_byt5.py
│   ├── 04_train_bilstm.py
│   ├── 05_train_transformer.py
│   ├── 06_inference.py
│   └── 07_evaluate.py
├── documentation/
│   └── PIPELINE_GUIDE.md        # This file
├── preprocess.py
├── kaggle_submission.py
├── EXECUTION_PLAN.md
└── LLM_AGENT_QUERY.md
```
