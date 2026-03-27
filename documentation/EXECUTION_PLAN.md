# Plan: Akkadian MT Baselines & Execution Strategy

## TL;DR
Build a progression of Akkadian→English MT baselines for the Deep Past Kaggle competition (late submission mode). Start with the simplest viable approach (fine-tune ByT5-base on cleaned train.csv), NOT BiLSTM or scratch Transformers. Data preprocessing is the #1 bottleneck—not model architecture. Every top solution used pretrained seq2seq models (ByT5 dominated), with data quality as the primary differentiator.

---

## Competition Summary
- **Task**: Translate transliterated Akkadian → English
- **Metric**: Geometric mean of BLEU and chrF++ (micro-averaged, SacreBLEU)
- **Submission**: Kaggle notebook (no internet), 9hr limit, `submission.csv` with columns `id,translation`
- **Data**: train.csv (~1500 docs, document-level aligned), test.csv (~4000 sentences, sentence-level)
- **Key challenge**: Training data is document-level but evaluation is sentence-level
- **Supplemental data**: published_texts.csv (~8k transliterations, no translations), publications.csv (~880 scholarly PDFs as OCR), OA_Lexicon_eBL.csv, eBL_Dictionary.csv, Sentences_Oare_FirstWord_LinNum.csv

## Critical Insight from Competition Results
**BiLSTM and from-scratch Transformers are NOT viable baselines for this task.** All top solutions (1st–33rd) used pretrained models, overwhelmingly ByT5. The best single models scored ~40-43 on the metric. The data quality was the dominant factor, not model architecture.

---

## Steps

### Phase 1: Data Preprocessing & Sentence Alignment (CRITICAL — blocks all later phases)

1. **Download and explore competition data** — Load train.csv, test.csv, sample_submission.csv, Sentences_Oare_FirstWord_LinNum.csv into pandas. Examine data distributions, text lengths, formatting.

2. **Implement text normalization pipeline** for transliterations:
   - Unicode NFKC normalization
   - Convert Ḫ/ḫ → H/h (competition convention)
   - Convert subscript numbers ₀-₉ → 0-9
   - Normalize determinatives to curly-bracket format: {d}, {ki}, {mi}, etc.
   - Standardize gap markers: [x] → `<gap>`, [...] / `...` → `<gap>`, deduplicate sequential gaps
   - Remove modern scribal notations: !, ?, /, :, < >, ˹ ˺, [ ] (keep text inside)
   - Convert accented vowels: a2→á, a3→à, etc. (if not already in diacritic form)
   - Handle sz→š, s,→ṣ, t,→ṭ conversions from alternate sources

3. **Implement translation cleaning pipeline**:
   - Remove scholarly annotations: (fem. plur.), (sic), (lit.), etc.
   - Clean brackets and editorial marks
   - Normalize gap markers in translations to match transliterations
   - Normalize fractions to Unicode: 1/2→½, 1/3→⅓, etc.
   - Replace Roman numeral months: Month XII → Month 12

4. **Create sentence-level training pairs** from document-level train.csv:
   - Use Sentences_Oare_FirstWord_LinNum.csv as alignment aid
   - Split each document into sentence-level (transliteration, translation) pairs
   - For initial baseline: use LLM (GPT-4o/Gemini) to perform sentence alignment (this is what ALL top teams did)
   - Alternatively: rule-based splitting using English sentence boundaries, then manually verify a sample

5. **Build validation set**: Hold out ~10-15% of aligned sentence pairs for local evaluation using sacrebleu geometric mean of BLEU + chrF++.

### Phase 2: Baseline Models (can run in parallel once data is ready)

6. **Baseline 1 — ByT5-base fine-tune (RECOMMENDED FIRST BASELINE)**
   - Model: `google/byt5-base` (HuggingFace)
   - Why: Byte-level tokenization handles Akkadian diacritics/special chars natively; no subword tokenizer issues
   - Training: HuggingFace Seq2SeqTrainer, 10-15 epochs, lr=3e-4, batch_size=16, Adafactor optimizer
   - Input format: raw transliteration text → English translation
   - Expected score: ~30-35 (based on competition ablations showing baseline ByT5 at ~34)
   - This IS the strongest simple baseline and should be done FIRST

7. **Baseline 2 — BiLSTM Seq2Seq with Attention (academic comparison only)**
   - Build character-level or BPE tokenizer for Akkadian
   - Encoder: 2-layer BiLSTM, hidden=512
   - Decoder: 2-layer LSTM with Bahdanau attention
   - Teacher forcing + scheduled sampling
   - Expected score: ~5-15 (very weak due to tiny dataset and no pretraining)
   - Purpose: demonstrate that pretraining is essential for low-resource MT

8. **Baseline 3 — Vanilla Transformer from scratch (academic comparison)**
   - Small Transformer (4 layers, 4 heads, d_model=256)
   - Character-level tokenization
   - Expected score: ~8-18 (better than BiLSTM but much worse than pretrained)
   - Purpose: show that even the right architecture needs pretraining in low-resource settings

9. **Baseline 4 — mT5-base or NLLB-200 fine-tune (pretrained multilingual comparison)**
   - Alternative pretrained model comparison point
   - mT5 uses SentencePiece (subword) — will struggle with Akkadian diacritics
   - NLLB-200 designed for low-resource languages but doesn't include Akkadian
   - Expected score: ~25-30 (worse than ByT5 due to tokenizer mismatch)

### Phase 3: Improvements (sequential, each builds on previous)

10. **Improvement 1 — Better data: CPT → SFT pipeline** (*depends on step 6*)
    - Stage 1 (CPT): Continue pre-training ByT5 on Akkadian transliteration text (all of published_texts.csv transliterations, no translations needed) for 3 epochs
    - Stage 2 (SFT): Fine-tune CPT checkpoint on sentence-level translation pairs
    - Based on 3rd place ablation: CPT→SFT gives +0.7 over direct SFT (40.6 vs 39.9)

11. **Improvement 2 — Pseudo-labeling** (*depends on step 10*)
    - Use best model to generate translations for unlabeled published_texts.csv
    - Retrain on combined real + pseudo-labeled data
    - Based on 3rd place ablation: +0.6 improvement (41.2 vs 40.6)

12. **Improvement 3 — PDF data extraction** (*parallel with step 10*)
    - Use Gemini/GPT-4o to extract transliteration-translation pairs from Kültepe Tablets PDFs
    - All top teams did this — it was the single biggest score improvement
    - Based on 2nd place: ~60k additional sentence pairs from PDF extraction

13. **Improvement 4 — Scale up model** (*depends on step 10*)
    - Move from ByT5-base → ByT5-large → ByT5-xl
    - Based on 1st place: larger models give significant improvements when data is clean
    - ByT5-large can train on single A100; ByT5-xl needs 8xH20 or equivalent

14. **Improvement 5 — MBR Decoding for inference** (*depends on any trained model*)
    - Generate multiple candidates per input (beam search + sampling at different temperatures)
    - Select best candidate using chrF++/BLEU/Jaccard consensus
    - Based on 1st place: meaningful improvement over greedy/beam-only decoding

15. **Optional: LLM-based approach** (*parallel with Phase 2*)
    - Fine-tune Qwen2.5-7B or 14B with LoRA on sentence-level data
    - Based on 25th place: Qwen2.5 with LoRA + 55k pairs → competitive results
    - More compute-intensive but shows decoder-only LLMs can work

### Phase 4: Submission & Evaluation

16. **Create Kaggle submission notebook**
    - Pre-upload trained model weights to Kaggle as a dataset
    - Load model, run inference on test.csv transliterations
    - Write predictions to submission.csv with columns `id,translation`
    - Verify notebook runs within 9hr limit with no internet

17. **Compute local metrics and write results**
    - Use sacrebleu to compute BLEU, chrF++, and geometric mean on validation set
    - Compare all baselines and improvements in a table
    - Analyze failure cases: named entities, gap handling, long sequences

---

## Relevant Files & Resources

### Competition Data Files
- `train.csv` — ~1500 docs with `oare_id`, `transliteration`, `translation` (document-level)
- `test.csv` — ~4000 sentences with `id`, `text_id`, `line_start`, `line_end`, `transliteration`
- `sample_submission.csv` — format template
- `Sentences_Oare_FirstWord_LinNum.csv` — sentence alignment aid (first word, line numbers)
- `published_texts.csv` — ~8000 transliterations without translations
- `publications.csv` — ~880 scholarly PDFs as OCR text
- `OA_Lexicon_eBL.csv` — Akkadian word forms with dictionary entries
- `eBL_Dictionary.csv` — Complete Akkadian dictionary

### Key Libraries
- `transformers` (HuggingFace) — Seq2SeqTrainer, AutoModelForSeq2SeqLM
- `sacrebleu` — BLEU, chrF++ scoring
- `ctranslate2` — Optimized inference (used by 1st place for speed)
- `torch` / `pytorch` — General DL
- `sentencepiece` — Tokenizer for mT5/NLLB baselines

### Model References
- `google/byt5-base`, `google/byt5-large`, `google/byt5-xl` — Primary model family
- `google/madlad400-3b-mt` — Alternative seq2seq (10th place ensemble)
- `Qwen/Qwen2.5-14B` — LLM baseline alternative (14th place)

---

## Verification
1. Run sacrebleu on validation set for each baseline — compute BLEU, chrF++, geometric mean
2. Submit to Kaggle (late submission) to get official score
3. Compare baseline table: BiLSTM vs Transformer-scratch vs ByT5-base vs mT5-base
4. Ablate CPT vs no-CPT on ByT5-base
5. Verify submission.csv format matches sample_submission.csv exactly

---

## Decisions
- **BiLSTM and vanilla Transformer are kept as academic baselines** to demonstrate the importance of pretraining for low-resource MT, but ByT5-base is the recommended starting point for actual competitive performance
- **ByT5 is the primary model family** based on overwhelming evidence from competition results (1st, 2nd, 3rd, 6th, 10th, 33rd all used it)
- **Data preprocessing is Phase 1** because every top solution identified data quality as the dominant factor
- **Sentence-level alignment is mandatory** because test set is sentence-level while training is document-level
- Competition deadline passed (March 23, 2026) but late submissions are accepted

## References
1. Xie et al. (2023) — "Translating Akkadian to English with Neural Machine Translation" (PNAS Nexus) — First Akkadian NMT paper using Transformer
2. ByT5: Xue et al. (2022) — "ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models" (TACL)
3. MADLAD-400: Kudugunta et al. (2024) — "MADLAD-400: A Multilingual And Document-Level Large Audited Dataset"
4. MBR Decoding: Eikema & Aziz (2022) — "Sampling-Based Minimum Bayes Risk Decoding for Neural Machine Translation"
5. 1st place solution: "Data Quality Dictates Everything" — ByT5-xl ensemble, MBR, 43.2 private
6. 2nd place solution: "Data-Centric Akkadian NMT" — ByT5-large, LLM sentence alignment, 41.0 private
7. 3rd place solution: "Synthetic Data to Teach OA Fundamentals" — ByT5-large/xl, CPT→SFT, synthetic drills, Qwen3-8B reward model
8. 10th place solution: "Seq2Seq + CPT + Pseudo-Labeling" — ByT5 + MADLAD ensemble, 39.9 private
9. 14th place solution: "LLM + Online Learning" — Qwen3-14B + ByT5 iterative pseudo-labeling
10. 25th place solution: "Post-training Qwen2.5 32B/72B" — LLM fine-tuning with LoRA
11. chrF++: Popović (2017) — "chrF++: words helping character n-grams" (WMT)
12. BLEU: Papineni et al. (2002) — "BLEU: a Method for Automatic Evaluation of Machine Translation" (ACL)
