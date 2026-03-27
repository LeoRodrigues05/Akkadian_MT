# Akkadian → English Machine Translation — Experiment Documentation

## 1. Data Preprocessing & Loading

### 1.1 Raw Data

The pipeline begins with `01_data_setup.py`, which downloads 10 CSVs (~218 MB) from the Kaggle  
**Deep Past Initiative** competition via `kagglehub`:

| File | Purpose |
|------|---------|
| `train.csv` | Document-level transliteration–translation pairs with `oare_id` |
| `test.csv` | Transliterations to translate for submission |
| `sentences.csv` | Sentence-level annotations linked via `text_uuid` |
| `words.csv`, `signs.csv`, … | Additional linguistic annotations (unused) |

### 1.2 Sentence Alignment (`02_sentence_alignment.py`)

Training data is document-level; each row may contain an entire tablet.  
We align it to sentence-level pairs for effective training:

1. **Join**: `train.oare_id == sentences.text_uuid` — matches 253 documents.  
2. **Segment by `first_word_number`**: Sentences in `sentences.csv` carry a 1-based  
   `first_word_number` indicating where each sentence begins in the document transliteration.  
   We split the document at these word boundaries, assigning each segment to its corresponding  
   English sentence.  
3. **Proportional fallback**: When `first_word_number` is missing or yields segments of zero  
   length, we instead distribute transliteration words proportionally by the character length  
   of each English sentence.  
4. **Train/Val split**: 90/10 random split → **5,098 train / 598 val** aligned pairs.  
5. **Output**: `aligned_train_split.csv` and `aligned_val_split.csv` with columns  
   `oare_id`, `transliteration`, `translation`.

### 1.3 Text Normalization (`preprocess.py`)

Both transliteration and translation go through cleaning before model input.

#### `clean_transliteration(text)`

| Step | Transformation |
|------|---------------|
| Unicode normalize | NFKC normalization |
| Special chars | `Ḫ` → `H`, `ḫ` → `h` |
| Subscript digits | `₀–₉` → `0–9` via `str.maketrans` |
| Gap markers | `[x x]`, `[...]`, `…`, `[broken]`, `[damaged]` → `<gap>` |
| Half-brackets | Remove `˹ ⸢ ˺ ⸣` |
| Square brackets | Remove enclosing brackets, keep content: `[text]` → `text` |
| Angle brackets | Remove non-gap angle brackets: `<text>` → `text` |
| Noise chars | Remove `! ? * #` |
| Sibilant | `sz` → `š`, `SZ` → `Š` |
| Emphatics | `s,` → `ṣ`, `t,` → `ṭ` (and uppercase) |
| Vowel accents | `a2` → `á`, `a3` → `à`, etc. (16 patterns) |
| Collapse gaps | Multiple consecutive `<gap>` → single `<gap>` |
| Whitespace | Collapse and strip |

#### `clean_translation(text)`

| Step | Transformation |
|------|---------------|
| Unicode normalize | NFKC normalization |
| Scholarly annotations | Remove `(...)`, commentary text |
| Fractions | `1/2` → `half`, `1/3` → `third`, etc. |
| Roman numeral months | `Month I` → `Month 1`, etc. |
| Whitespace | Collapse and strip |

### 1.4 Tokenization Strategies

| Model | Strategy | Details |
|-------|----------|---------|
| **ByT5** | Byte-level (built-in) | ByT5 operates on raw UTF-8 bytes; no external tokenizer needed. `MAX_INPUT_LENGTH = MAX_TARGET_LENGTH = 512` bytes. |
| **BiLSTM** | Character-level (`CharVocab`) | Custom vocabulary built from all unique characters in train+val. Special tokens: `<pad>=0, <sos>=1, <eos>=2, <unk>=3`. `MAX_LEN = 300` characters. |
| **Transformer** | Character-level (`CharVocab`) | Same character-level vocab as BiLSTM. `MAX_LEN = 300` characters. |

---

## 2. Model Specifications

### 2.1 ByT5-base (Primary Model)

| Property | Value |
|----------|-------|
| **Architecture** | Encoder-decoder Transformer (T5 variant) |
| **Pre-trained model** | `google/byt5-base` |
| **Parameters** | ~580M |
| **Input representation** | UTF-8 byte sequences |
| **Fine-tuning epochs** | 15 |
| **Batch size** | 4 (effective 64 with gradient_accumulation_steps=16) |
| **Learning rate** | 3e-4 |
| **Optimizer** | AdamW (PyTorch) |
| **LR schedule** | Linear warmup (warmup_ratio=0.05) |
| **Precision** | bf16 (mixed precision) |
| **Max sequence length** | 512 (input and target) |
| **Best model selection** | Lowest eval_loss |
| **Inference** | Beam search (num_beams=4, max_length=512) |
| **Framework** | HuggingFace `Seq2SeqTrainer` |

### 2.2 BiLSTM Seq2Seq with Bahdanau Attention

| Property | Value |
|----------|-------|
| **Architecture** | Bidirectional LSTM encoder + LSTM decoder with Bahdanau attention |
| **Parameters** | 17,343,613 (~17.3M) |
| **Embedding dim** | 256 |
| **Hidden dim** | 512 |
| **Encoder layers** | 2 (bidirectional → 1024-dim outputs) |
| **Decoder layers** | 2 |
| **Dropout** | 0.3 |
| **Attention** | Bahdanau (additive): `v · tanh(W_q · h_dec + W_k · h_enc)` |
| **Decoder output** | `fc_out(concat(rnn_out, context, embedding))` → vocab logits |
| **Teacher forcing** | 0.5 (during training) |
| **Batch size** | 64 |
| **Epochs** | 50 |
| **Learning rate** | 1e-3 |
| **Optimizer** | Adam |
| **Max sequence length** | 300 characters |
| **Best model selection** | Highest geometric mean (BLEU × chrF++) |
| **Inference** | Greedy decoding |
| **Encoder details** | Bidirectional LSTM → `fc_h` and `fc_c` linear projections with tanh to combine forward/backward states |

### 2.3 Vanilla Transformer (from scratch)

| Property | Value |
|----------|-------|
| **Architecture** | Standard Transformer encoder-decoder (`nn.Transformer`) |
| **Parameters** | 5,358,717 (~5.4M) |
| **d_model** | 256 |
| **d_ff** | 512 |
| **Attention heads** | 4 |
| **Encoder layers** | 4 |
| **Decoder layers** | 4 |
| **Dropout** | 0.1 |
| **Positional encoding** | Sinusoidal (fixed) |
| **Weight init** | Xavier uniform for all params with dim > 1 |
| **Batch size** | 64 |
| **Epochs** | 50 |
| **Learning rate** | 1e-4 (peaked by warmup scheduler) |
| **Optimizer** | Adam (β₁=0.9, β₂=0.98, ε=1e-9) |
| **LR schedule** | Noam warmup: `d_model^(-0.5) · min(step^(-0.5), step · warmup^(-1.5))` with warmup_steps=2000 |
| **Max sequence length** | 300 characters |
| **Gradient clipping** | max_norm = 1.0 |
| **Best model selection** | Highest geometric mean (BLEU × chrF++) |
| **Inference** | Greedy autoregressive decoding |

---

## 3. Results (Validation Set — 598 samples)

| Model | BLEU | chrF++ | Geo Mean | Parameters | Best Epoch |
|-------|------|--------|----------|------------|------------|
| **ByT5-base** | *(pending — run `08_evaluate_byt5.py`)* | — | — | ~580M | 15 (eval_loss=0.5267) |
| **BiLSTM Seq2Seq** | 17.45 | 36.17 | 25.12 | 17.3M | 45/50 |
| **Vanilla Transformer** | 8.55 | 28.06 | 15.49 | 5.4M | 50/50 |

### Key Observations

- **BiLSTM outperforms the from-scratch Transformer** on both BLEU (+8.9) and chrF++ (+8.1).  
  The Bahdanau attention mechanism and higher dropout (0.3 vs 0.1) likely helped with the  
  small dataset (5,098 training pairs).
- **Transformer underfitting**: Best epoch = 50 (final epoch) suggests the model had not fully  
  converged. More epochs or larger d_model might improve results.
- **ByT5** benefits from massive pre-training; its evaluation loss (0.5267) is promising but  
  generation-based metrics (BLEU/chrF++) need to be computed via `08_evaluate_byt5.py`.

---

## 4. File Inventory

| File | Description |
|------|-------------|
| `preprocess.py` | Text normalization functions |
| `scripts/01_data_setup.py` | Download competition data via kagglehub |
| `scripts/02_sentence_alignment.py` | Document → sentence alignment |
| `scripts/03_train_byt5.py` | ByT5-base fine-tuning |
| `scripts/04_train_bilstm.py` | BiLSTM Seq2Seq training |
| `scripts/05_train_transformer.py` | Vanilla Transformer training |
| `scripts/06_inference.py` | ByT5 test-set inference → `submission.csv` |
| `scripts/07_evaluate.py` | Multi-model comparison table |
| `scripts/08_evaluate_byt5.py` | ByT5 BLEU/chrF++ evaluation on val set |
| `kaggle_notebook.ipynb` | Kaggle submission notebook (all 3 models) |
| `checkpoints/byt5-base/` | ByT5 checkpoint + `eval_results.json` |
| `checkpoints/bilstm/` | BiLSTM checkpoint + `eval_results.json` |
| `checkpoints/transformer/` | Transformer checkpoint + `eval_results.json` |

---

## 5. Reproduction

```bash
# 1. Download data
python scripts/01_data_setup.py

# 2. Align sentences
python scripts/02_sentence_alignment.py

# 3. Train all models (submit to SLURM or run locally with GPU)
python scripts/03_train_byt5.py
python scripts/04_train_bilstm.py
python scripts/05_train_transformer.py

# 4. Evaluate ByT5 (BLEU/chrF++)
python scripts/08_evaluate_byt5.py

# 5. Compare all models
python scripts/07_evaluate.py

# 6. Generate Kaggle submission
python scripts/06_inference.py
```

### Environment

- Python 3.11 (Miniconda)
- PyTorch with CUDA 12.4
- HuggingFace Transformers
- sacrebleu
- NVIDIA RTX 5000 Ada Generation (32 GB VRAM)
