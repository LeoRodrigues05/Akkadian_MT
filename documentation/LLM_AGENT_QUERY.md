# LLM Agent Query — Build Akkadian MT Baselines

```
You are building baseline machine translation models for the Deep Past Kaggle competition (Akkadian → English). The workspace is at: /home/leo.rodrigues/Akkadian_MT

TASK: Build a complete pipeline from data preprocessing to Kaggle submission for translating Akkadian transliterations to English.

STEP 1 — DATA SETUP:
- Download competition data using: kagglehub.competition_download('deep-past-initiative-machine-translation')
- Load train.csv, test.csv, sample_submission.csv, Sentences_Oare_FirstWord_LinNum.csv into pandas

STEP 2 — TEXT NORMALIZATION:
Create a preprocessing module (preprocess.py) with two functions:
- clean_transliteration(text): Apply Unicode NFKC, convert Ḫ/ḫ→H/h, subscripts ₀-₉→0-9, normalize determinatives to {d}/{ki}/etc, standardize gaps to <gap>, remove scribal notations (!?/: and brackets), convert sz→š s,→ṣ t,→ṭ, convert a2→á a3→à e2→é etc.
- clean_translation(text): Remove scholarly annotations like (fem. plur.)/(sic)/(lit.), normalize fractions to Unicode (1/2→½), convert Roman months to Arabic, align <gap> markers, clean brackets.

STEP 3 — SENTENCE ALIGNMENT:
The training data in train.csv is document-level (full tablet transliterations paired with full English translations). The test set is sentence-level. You MUST split training documents into sentence-level pairs.
- Use Sentences_Oare_FirstWord_LinNum.csv as an alignment aid
- Split English translations at sentence boundaries (periods followed by capital letters)
- Align corresponding Akkadian segments using line numbers from the CSV
- Save aligned pairs to a new CSV: aligned_train.csv with columns [oare_id, transliteration, translation]
- Hold out 10% as validation set

STEP 4 — BASELINE 1: ByT5-base (PRIMARY):
- Model: google/byt5-base from HuggingFace
- Use Seq2SeqTrainer with Seq2SeqTrainingArguments
- Training config: epochs=15, learning_rate=3e-4, per_device_train_batch_size=16, per_device_eval_batch_size=16, predict_with_generate=True, generation_max_length=512, fp16=True, evaluation_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="eval_loss", gradient_accumulation_steps=4, optim="adafactor"
- Compute BLEU and chrF++ on validation set using sacrebleu
- Save model checkpoint

STEP 5 — BASELINE 2: BiLSTM Seq2Seq (ACADEMIC COMPARISON):
- Build character-level vocabulary from training data
- Implement Encoder (2-layer BiLSTM, hidden=512, embedding=256) and Decoder (2-layer LSTM with Bahdanau attention)
- Train with teacher forcing, Adam optimizer, lr=1e-3, 50 epochs
- Evaluate on validation set

STEP 6 — BASELINE 3: Vanilla Transformer from scratch (ACADEMIC COMPARISON):
- Character-level tokenization
- Small Transformer: 4 layers, 4 heads, d_model=256, d_ff=512
- Train with Adam + warmup scheduler, 50 epochs
- Evaluate on validation set

STEP 7 — KAGGLE SUBMISSION:
Create an inference notebook (inference.py / .ipynb) that:
- Loads the best ByT5 model checkpoint (pre-uploaded as a Kaggle dataset)
- Reads test.csv
- Applies clean_transliteration() to each test transliteration
- Runs model.generate() with num_beams=4, max_length=512
- Writes predictions to submission.csv with columns id,translation
- Ensure submission.csv matches sample_submission.csv format exactly

STEP 8 — EVALUATION & COMPARISON:
Create a results comparison table showing BLEU, chrF++, and geometric mean for each baseline.

IMPORTANT NOTES:
- The competition metric is geometric mean of BLEU and chrF++ (use sacrebleu library)
- ByT5 uses byte-level tokenization — do NOT build a custom tokenizer for it
- Install: pip install transformers datasets sacrebleu sentencepiece accelerate kagglehub
- The test.csv provided locally is DUMMY data; real test data only appears during Kaggle notebook execution
- All model files must be pre-uploaded to Kaggle as datasets (no internet during submission)
```
