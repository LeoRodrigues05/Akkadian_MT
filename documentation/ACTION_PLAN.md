# Akkadian MT — Improvement Action Plan

## Current State (Baseline Results)

| # | Model | BLEU | chrF++ | Geo Mean | Kaggle Private | Params |
|---|-------|------|--------|----------|----------------|--------|
| 1 | ByT5-base | 19.46 | 37.71 | 27.09 | 28.15 | 580M |
| 2 | BiLSTM Seq2Seq | 17.45 | 36.17 | 25.12 | 18.92 | 17.3M |
| 3 | Vanilla Transformer | 8.55 | 28.06 | 15.49 | 14.07 | 5.4M |

**Training data:** ~5,098 sentence-level pairs (from 253 explicitly aligned + proportional fallback on the rest).  
**Gap to medal zone:** The top competition solutions achieved 40–43 geometric mean using the **same ByT5 architecture** but with ~20k clean pairs. The gap is almost entirely data-driven.

---

## Implementation Overview

All new scripts are numbered 09–16 and follow the existing pipeline convention. The execution order below reflects dependencies between phases.

### Execution Order

```
Phase 1 (Data Mining & Normalization)
  09_mine_published_texts.py      →  monolingual_akkadian.csv
  10_lexicon_integration.py       →  lexicon_drills.csv, proper_noun_lookup.json
  [preprocess.py fixes applied]   →  improved normalization

Phase 2 (Alignment & Extraction)
  11_llm_alignment.py             →  llm_aligned_train.csv  (requires API key)
  13_extract_publications.py      →  ocr_extracted_pairs.csv
  12_quality_filter.py            →  filtered_train.csv

Phase 3 (Augmentation)
  14_data_augmentation.py         →  augmented_train_split.csv, augmented_val_split.csv

Phase 4 (Model Training)
  15_train_byt5_large.py --stage both  →  byt5-large-cpt/, byt5-large-sft/

Phase 5 (Inference)
  06_inference.py                 →  submission.csv  (beam search, improved params)
  16_mbr_decode.py                →  submission_mbr.csv  (MBR decoding)
```

---

## Phase 1 — Data Mining & Normalization Fixes

### 1.1 Normalization improvements (`preprocess.py` — already applied)

| Fix | Description |
|-----|-------------|
| Hamza/Ayin deletion | ʾ (U+02BE), ʿ (U+02BF), ʼ (U+02BC) → removed |
| Ḫ/ḫ → H/h | Consistent with test data convention |
| Subscript x | ₓ → x (was missing from subscript map) |
| Double angle brackets | `<< >>` (errant signs) → removed entirely |
| Big gap markers | `{large break}`, `[… …]`, `…` → `<big_gap>` |
| All 17 determinatives | Added: {mul}, {e₂}, {dub}, {id₂}, {mušen}, {kuš}, {u₂}, {lu₂}, etc. |
| Gap deduplication | Mixed gap sequences collapsed to `<big_gap>` |

### 1.2 Mine published_texts.csv (`09_mine_published_texts.py`)

- **Input:** 7,953 texts in `published_texts.csv`, of which 6,388 are NOT in `train.csv`
- **Output:** `monolingual_akkadian.csv` — cleaned transliterations for CPT and pseudo-labeling
- **Impact:** Provides monolingual Akkadian for continued pre-training (Stage 1 of two-stage training)

### 1.3 Lexicon integration (`10_lexicon_integration.py`)

- **Input:** `OA_Lexicon_eBL.csv` (39k entries) + `eBL_Dictionary.csv` (19k entries)
- **Output:**
  - `lexicon_drills.csv` — synthetic word-level transliteration→English pairs
  - `proper_noun_lookup.json` — PN/GN form → normalized name mapping
- **Impact:** Dictionary drills augment vocabulary coverage; proper noun lookup can post-process model outputs

### 1.4 Beam search improvements (`06_inference.py`, `kaggle_submission.py` — already applied)

| Parameter | Before | After |
|-----------|--------|-------|
| `num_beams` | 4 | 8 |
| `no_repeat_ngram_size` | — | 3 |
| `length_penalty` | — | 1.0 |

Expected impact: +0.5–1 point from reduced repetitive output.

---

## Phase 2 — LLM-Assisted Sentence Alignment (Highest Impact)

### 2.1 LLM alignment (`11_llm_alignment.py`)

**Problem:** Only 253 out of ~1,500 documents have explicit sentence alignment via the CSV. The remaining ~1,300 use noisy proportional fallback, which distributes Akkadian words by English character length — often producing misaligned pairs.

**Solution:**
- Send each unaligned document (transliteration + translation) to an LLM (Gemini free tier or GPT-4o-mini)
- LLM outputs JSON with aligned sentence pairs
- Quality-check: length ratios, completeness, deduplication
- Resume support via checkpoint file

**Requirements:** `GEMINI_API_KEY` or `OPENAI_API_KEY` environment variable.

**Expected impact:** Transform ~1,300 noisy proportional alignments into clean sentence-level pairs. This was the single biggest differentiator between top solutions (20k+ clean pairs) and baselines (~5k noisy pairs).

### 2.2 OCR extraction (`13_extract_publications.py`)

- **Input:** `publications.csv` (216k pages of OCR from ~880 PDFs)
- **Strategy:** Detect translation blocks via section headers, cross-reference tablet identifiers to match transliterations from `published_texts.csv`
- **Output:** `ocr_extracted_pairs.csv`
- **Note:** Extraction quality is inherently noisy; best used alongside LLM alignment

### 2.3 Quality filtering (`12_quality_filter.py`)

Applied to any aligned CSV:
1. **Length ratio filter** — reject pairs with src/tgt ratio > 6× or < 0.1×
2. **Gap ratio filter** — reject pairs where > 70% of source is gap tokens
3. **Language detection** — basic check that target looks like English
4. **Deduplication** — exact + near-duplicate removal (keep longer translations)
5. **Repetition filter** — reject translations with > 80% repeated words

---

## Phase 3 — Data Augmentation

### 3.1 Combined augmentation (`14_data_augmentation.py`)

Merges data from all sources:

| Source | Description | Est. Pairs |
|--------|-------------|-----------|
| Base aligned | Original + LLM-aligned sentence pairs | 8k–15k |
| OCR extracted | From publications.csv | 500–2k |
| Dictionary drills | Word-level from lexicon | 5k–15k |
| Template pairs | Formulaic patterns (debt, seals, letters) | 500–2k |
| Pseudo-labels | ByT5 translations of monolingual texts | 2k–5k |

**Key design decision:** Validation set contains ONLY real human-translated pairs (no synthetic data) for fair evaluation.

**Pseudo-labeling:** Uses best current ByT5 model to translate monolingual Akkadian texts from `published_texts.csv`. Only high-confidence outputs are kept.

---

## Phase 4 — Model Scaling & Two-Stage Training

### 4.1 Two-stage training (`15_train_byt5_large.py`)

**Stage 1 — Continued Pre-Training (CPT):**
- Fine-tune ByT5 on ALL available Akkadian text using a denoising objective (T5-style span corruption)
- This teaches the model Akkadian language patterns without needing parallel data
- Uses monolingual data from `09_mine_published_texts.py` + source side of training data
- 5 epochs, cosine LR, gradient checkpointing

**Stage 2 — Supervised Fine-Tuning (SFT):**
- Start from the CPT checkpoint
- Fine-tune on expanded clean parallel data from Phase 3
- 15 epochs, label smoothing 0.1, cosine LR with warmup
- Best model selection by validation loss

### 4.2 Model scaling

| Model | Params | VRAM (bf16) | Fits RTX 5000 Ada? |
|-------|--------|-------------|---------------------|
| byt5-base | 580M | ~4 GB | ✅ Easily |
| byt5-large | 1.2B | ~8 GB | ✅ With grad ckpt |
| byt5-xl | 3.7B | ~20 GB | ⚠️ Tight, needs DeepSpeed |

**Recommendation:** Start with `byt5-large`. All top-3 competition solutions used `byt5-large` or `byt5-xl`.

```bash
# CPT then SFT with byt5-large
python scripts/15_train_byt5_large.py --stage both --model google/byt5-large --batch-size 2

# SFT only (skip CPT, faster)
python scripts/15_train_byt5_large.py --stage sft --model google/byt5-large
```

---

## Phase 5 — Inference Optimization

### 5.1 Improved beam search (already applied)

Beam width 8, `no_repeat_ngram_size=3`, `length_penalty=1.0`.

### 5.2 MBR Decoding (`16_mbr_decode.py`)

- Generate 20 candidates per input via diverse sampling (top-k=50, top-p=0.95, temp=0.7)
- Add beam search result for stability
- Score each candidate against all others using sentence-level geometric mean of BLEU and chrF++
- Select highest-scoring candidate

**Expected impact:** +1–3 points. More expensive at inference (~20× generation cost) but consistently improves quality.

```bash
python scripts/16_mbr_decode.py --n-samples 20 --model checkpoints/byt5-large-sft/best_model
```

---

## Projected Score Trajectory

| Phase | Est. Geo Mean | Key Driver | Training Data |
|-------|--------------|-----------|---------------|
| Current baseline | 28.15 | 5k pairs, byt5-base | 5,098 |
| After Phase 1 | 29–30 | Better normalization + beam search | 5,098 |
| After Phase 2 | 33–36 | LLM-aligned 10–15k clean pairs | 10k–15k |
| After Phase 3 | 36–39 | Augmented to 20k+ pairs | 15k–25k |
| After Phase 4 | 39–42 | byt5-large + CPT→SFT | 15k–25k |
| After Phase 5 | 40–43 | MBR decoding | 15k–25k |

---

## Quick-Start Commands

```bash
# Phase 1: Data mining (no GPU needed)
python scripts/09_mine_published_texts.py
python scripts/10_lexicon_integration.py

# Phase 2: LLM alignment (needs API key)
export GEMINI_API_KEY="your-key-here"
python scripts/11_llm_alignment.py

# Phase 2: OCR extraction + quality filter
python scripts/13_extract_publications.py
python scripts/12_quality_filter.py --input data/llm_aligned_train.csv

# Phase 3: Augmentation
python scripts/14_data_augmentation.py --base-data data/filtered_train.csv

# Phase 4: Train byt5-large (GPU required)
python scripts/15_train_byt5_large.py --stage both --model google/byt5-large

# Phase 5: MBR inference
python scripts/16_mbr_decode.py --model checkpoints/byt5-large-sft/best_model
```

---

## Key Decisions

1. **Data over architecture.** Top solutions all used vanilla ByT5 with no architectural changes. The 12+ point gap is entirely data quality/quantity.
2. **BiLSTM/Transformer baselines are frozen.** They serve the mid-term report but won't improve competition standing.
3. **byt5-large over byt5-xl.** 1.2B fits on RTX 5000 Ada (32 GB); 3.7B requires multi-GPU or aggressive quantization.
4. **Clean val set.** Validation set contains only real human-translated pairs — no synthetic data — for fair evaluation.
5. **Competition ended March 2026.** Late submissions still accepted. Goal is final report and demonstrating the data-centric approach.

---

## File Inventory (New Scripts)

| Script | Purpose | Phase |
|--------|---------|-------|
| `09_mine_published_texts.py` | Extract monolingual Akkadian from published_texts.csv | 1 |
| `10_lexicon_integration.py` | Dictionary drills + proper noun lookup | 1 |
| `11_llm_alignment.py` | LLM-assisted sentence alignment | 2 |
| `12_quality_filter.py` | Data quality filtering | 2 |
| `13_extract_publications.py` | OCR extraction from publications.csv | 2 |
| `14_data_augmentation.py` | Merge all data sources + augmentation | 3 |
| `15_train_byt5_large.py` | ByT5-large with CPT→SFT two-stage training | 4 |
| `16_mbr_decode.py` | Minimum Bayes Risk decoding | 5 |

## Modified Files

| File | Changes |
|------|---------|
| `preprocess.py` | Added glottal char deletion, subscript x, big_gap, double angle brackets, all 17 determinatives |
| `scripts/06_inference.py` | Beam width 4→8, no_repeat_ngram_size=3, updated normalization |
| `kaggle_submission.py` | Same beam search + normalization updates as 06_inference.py |
