# Akkadian MT — Improvement Plan

## Current Baseline

| Model | BLEU | chrF++ | Kaggle Public | Kaggle Private |
|-------|------|--------|---------------|----------------|
| ByT5-base | 19.46 | 37.71 | 25.96 | 28.15 |
| BiLSTM | 17.45 | 36.17 | 17.80 | 18.92 |
| Transformer | 8.55 | 28.06 | 12.71 | 14.07 |

**Top competition scores** (for reference):
- 1st place: 41.5 public / 43.2 private (ensemble of 11 × byt5-xl)
- 2nd place: 42.4 public / 41.0 private (single byt5-large)
- 3rd place: ~41.3 public / 41.4 private (byt5-large + byt5-xl ensemble)

The gap from our best (28.15) to medal zone (~40+) is **entirely explained by data**. All three winners used vanilla ByT5 with no architectural changes — every point of improvement came from building larger, cleaner training corpora.

---

## Priority 1: Leverage Unused Competition Data (High Impact, Low Effort)

### 1A. Mine `published_texts.csv` for Additional Training Pairs

`published_texts.csv` contains ~13,456 published texts, many with `transliteration_orig` and some with `AICC_translation` fields. Currently **completely unused** in our pipeline.

**Action items:**
1. Load `published_texts.csv`, filter rows that have both a non-empty transliteration and translation
2. Apply `clean_transliteration()` and `clean_translation()` to the extracted pairs
3. Deduplicate against existing `aligned_train.csv` by `oare_id`
4. Add as supplementary training data

**Expected impact:** The 2nd-place winner explicitly noted that `published_texts.csv` contains ~8k tablets with transliterations that can be used for pseudo-labeling or LLM-assisted translation. Even extracting the already-translated subset could double our training data.

### 1B. Use `publications.csv` OCR Text

The 3rd-place winner extracted ~20k sentence pairs from `publications.csv` OCR text.

**Action items:**
1. Parse `publications.csv` for texts containing transliteration-translation pairs
2. Use regex or heuristic sentence splitting to extract parallel pairs
3. Clean and deduplicate

### 1C. Integrate `eBL_Dictionary.csv` and `OA_Lexicon_eBL.csv`

Currently zero references to these dictionary/lexicon files anywhere in the codebase.

**Action items:**
1. Build a word-level Akkadian→English lookup from `eBL_Dictionary.csv` (`word` → `definition`)
2. Use `OA_Lexicon_eBL.csv` for morphological normalization (mapping surface forms to lemmas)
3. Use the dictionary to create vocabulary drill training pairs (see Priority 3B)

---

## Priority 2: Improve Sentence Alignment Quality (High Impact, Medium Effort)

### Current Problem

Only 253 out of ~2,700 documents have explicit sentence anchors via `first_word_number`. The remaining ~90% use **proportional fallback** — a noisy heuristic that introduces misaligned training pairs.

### 2A. LLM-Assisted Sentence Alignment

Both 1st and 2nd-place winners used LLMs to produce clean sentence-level alignments. The 2nd-place winner's three-stage pipeline:

1. **Sentence Breaker**: For documents with full translations but no sentence segmentation — send full Akkadian + full English to an LLM, have it produce sentence-level aligned pairs
2. **Translation Fixer**: For documents with corrupt/misaligned sentence fragments — use LLM to re-anchor and fix
3. **(Optional) Translation Generator**: For untranslated transliterations — have LLM generate translations (2nd place found this hurt quality; 1st and 3rd place found it helpful with careful prompting)

**Action items:**
1. Create script `09_llm_sentence_alignment.py`
2. For each document in `train.csv`, send the full transliteration + full translation to an LLM (Gemini API is free-tier viable) 
3. Prompt the LLM to split into sentence-level aligned pairs
4. Quality-filter: drop pairs where transliteration/translation length ratio is extreme (>100 char difference per 2nd-place finding)
5. Replace the proportional fallback data with LLM-aligned data

**Expected impact:** The 2nd-place winner went from ~6k pairs to ~20k+ sentence pairs just from the official data using this approach. This alone could push BLEU from ~28 to ~35+ on the private LB.

### 2B. Cross-Fold Error Detection (from 1st-place)

After training an initial model, use it to detect training errors:
1. Train with 4-fold CV
2. Each fold's model infers on its training fold  
3. If the prediction is far from the training label (`geo_metric < 20`), flag as error
4. Re-extract or remove flagged samples

---

## Priority 3: Data Augmentation (High Impact, Medium-High Effort)

### 3A. Pseudo-Labeling / Backtranslation

Use your fine-tuned ByT5 model to generate translations for untranslated texts:

1. Take transliterations from `published_texts.csv` that have no English translation
2. Run inference with your best ByT5 checkpoint
3. Filter pseudo-labels by confidence (e.g., beam search score threshold)
4. Add high-confidence pseudo-labeled pairs to training

**Why this works:** The 3rd-place ablations showed pseudo-labeled data (27.7k sentences) improved from 40.8→41.2 public BLEU. Even with a weaker model, pseudo-labeling on unseen transliterations adds distributional coverage.

### 3B. Dictionary-Based Synthetic Data (from 1st and 3rd place)

Both 1st and 3rd-place winners generated synthetic training pairs using dictionaries:

**Vocabulary Drills (3rd place):**
1. Extract ~1k lemmas from `eBL_Dictionary.csv`  
2. For each (lemma, definition) pair, use an LLM to generate synthetic transliteration-translation sentence pairs
3. Focus on lemmas NOT appearing in the current training set (coverage gap analysis)
4. Generate ~30-90k synthetic pairs

**Slot-Fill Templates (3rd place):**
1. Extract formula patterns from training data (debt contracts, witness lists, etc.)
2. Create templates with typed slots: `{AMOUNT} KÙ.BABBAR ṣa-ru-pá-am i-ṣé-er {PN1} {PN2} i-šu`
3. Fill programmatically from `onomasticon.csv` names and known commodity terms
4. Can generate 100k+ pairs cheaply

### 3C. Grammar-Rule Synthetic Data (from 3rd place)

Extract grammar rules from OA grammar references and use LLMs to apply transformations:
- Swap verb tense, person, mood, clause structure
- ~50k pairs from ~500 rules

### 3D. Document-Level Augmentation (from 2nd place)

Concatenate consecutive sentence-level pairs into document-level training examples up to a 768-byte ceiling:
- Provides inter-sentence context during training
- Natural augmentation via variable chunking boundaries
- Matches realistic test-set input lengths

---

## Priority 4: Model & Training Improvements (Medium Impact, Low-Medium Effort)

### 4A. Scale Up: ByT5-large or ByT5-xl

All three winners used **byt5-large** or **byt5-xl**. None used byt5-base for their final submissions.

| Model | Params | Notes |
|-------|--------|-------|
| byt5-base | 580M | Our current model |
| byt5-large | 1.2B | Used by 2nd and 3rd place; fits in ~24GB with bf16 |
| byt5-xl | 3.7B | Used by 1st place; needs 8×H20 or quantization |

**With RTX 5000 Ada (32GB):**
- `byt5-large` should fit with gradient checkpointing + bf16 + reduced batch size
- `byt5-xl` likely too large without multi-GPU or quantized training

**Action items:**
1. Try `byt5-large` with: `gradient_checkpointing=True`, `bf16=True`, `per_device_batch_size=2`, `gradient_accumulation_steps=32`
2. Use Adafactor optimizer (memory-efficient, no momentum state) instead of AdamW
3. The 1st-place winner: "larger models are better... the prerequisite is having sufficient and clean training data"

### 4B. Training Recipe Refinements

From the winners:

| Parameter | Current | Recommended |
|-----------|---------|-------------|
| Epochs | 15 | 3-5 (with more data) |
| Batch size (effective) | 64 | 128-256 |
| LR | 3e-4 | 7e-5 (for larger models) |
| Optimizer | AdamW | Adafactor |
| Metric selection | eval_loss | eval_loss (confirmed by 1st place) |
| Max input length | 512 | 768 |
| group_by_length | False | True (reduces padding waste) |

### 4C. Two-Stage Training: CPT → FT (from 3rd place)

1. **Stage 1 — Continued Pre-Training (CPT):** Train on ALL data (synthetic + real + pseudo-labeled) with long warmup + constant LR. This teaches OA fundamentals.
2. **Stage 2 — Fine-Tuning (FT):** Starting from the CPT checkpoint, fine-tune on only high-quality scholar-translated data. Use cosine decay LR.

3rd-place ablation: CPT→FT improved from 39.9 → 40.6 on public LB (+0.7 BLEU).

### 4D. Inference Improvements

1. **Beam search tuning:** Increase from `num_beams=4` to `num_beams=8`
2. **MBR Decoding (from 1st place):** Generate multiple candidates via beam search + temperature sampling, then select the one with highest average similarity to all others using chrF++ + BLEU + Jaccard
3. **Repeated n-gram cleaning:** Post-process to remove repeated 2-7 grams from outputs

---

## Priority 5: Text Normalization Improvements (Low-Medium Impact, Low Effort)

### 5A. Align Normalization with Winners

From the 2nd-place winner's normalization rules:

| Rule | Our Status |
|------|------------|
| `sz` → `š` | ✅ Done |
| `ḫ/Ḫ` → `h/H` | ✅ Done |
| Subscript vowels (a2→á, a3→à) | ✅ Done |
| Hamza variants (ʾ, ʿ, ʼ) → deleted | ❌ Missing |
| Higher subscripts (₄₅₆…) → plain ASCII | ❌ Partially — we convert ₀-₉ but may miss edge cases |
| Determinative standardization to `{d}`, `{ki}` | ✅ Done |

**Action items:**
1. Add hamza/ayin deletion to `clean_transliteration()`
2. Verify edge cases in subscript handling
3. Strip parenthetical supplementary text from translations (mentioned by 1st place)

---

## Recommended Execution Order

Given the constraints (single RTX 5000 Ada 32GB, student project), prioritize by **impact ÷ effort**:

### Phase 1 — Quick Wins (1-2 days)
1. **Mine `published_texts.csv`** for already-translated pairs (Priority 1A)
2. **Fix text normalization** gaps (Priority 5A) 
3. **Increase beam search** to 8, add n-gram repetition cleaning (Priority 4D)
4. **Add `group_by_length=True`** to training args

### Phase 2 — Data Quality (3-5 days)
5. **LLM sentence alignment** on `train.csv` documents (Priority 2A)  
   - Use Gemini API (free tier: 1500 req/day) or local Qwen model
6. **Extract pairs from `publications.csv`** OCR text (Priority 1B)
7. **Quality-filter** training data by length ratio

### Phase 3 — Data Augmentation (3-5 days)
8. **Dictionary vocab drills** using `eBL_Dictionary.csv` (Priority 3B)
9. **Pseudo-labeling** untranslated texts in `published_texts.csv` (Priority 3A)
10. **Slot-fill template augmentation** (Priority 3B)
11. **Document-level augmentation** — concatenate sentence pairs (Priority 3D)

### Phase 4 — Model Scaling (2-3 days)
12. **Train byt5-large** with expanded dataset (Priority 4A)
13. **Try two-stage CPT→FT** training (Priority 4C)
14. **MBR decoding** for final inference (Priority 4D)

### Projected Score Trajectory

| Phase | Est. Private Score | Key Driver |
|-------|-------------------|------------|
| Current | 28.2 | 5k pairs, byt5-base |
| After Phase 1 | 29-30 | Better normalization + inference |
| After Phase 2 | 33-36 | Cleaner + more training data |
| After Phase 3 | 36-39 | Augmented data + pseudo-labels |
| After Phase 4 | 39-41 | byt5-large + two-stage training |

---

## Key Takeaway from Winners

> "This competition is a data bottleneck problem." — 2nd place
> 
> "Data quality dictates everything." — 1st place
> 
> "The private leaderboard rewards clean, diverse data much more than architectural tricks." — 2nd place

All three winners used **vanilla ByT5 with no architectural modifications**. The entire performance gap is data-driven. Focus effort on building a larger, cleaner training corpus before scaling models.
