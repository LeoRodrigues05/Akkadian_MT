"""
Kaggle Submission Notebook — Akkadian → English MT
This is the self-contained script to run on Kaggle (no internet mode).

Setup on Kaggle:
1. Upload your best ByT5 checkpoint folder as a Kaggle dataset
   (e.g., "byt5-akkadian-checkpoint")
2. The dataset will appear at /kaggle/input/byt5-akkadian-checkpoint/
3. Create a new notebook, paste this code, add the dataset, and submit
"""
import os
import re
import unicodedata
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Update these paths for your Kaggle setup
# ═══════════════════════════════════════════════════════════════════════════
MODEL_PATH = "/kaggle/input/byt5-akkadian-checkpoint/best_model"
TEST_CSV = "/kaggle/input/deep-past-initiative-machine-translation/test.csv"
OUTPUT_CSV = "/kaggle/working/submission.csv"

BATCH_SIZE = 16
NUM_BEAMS = 8
MAX_LENGTH = 512
NO_REPEAT_NGRAM_SIZE = 3
LENGTH_PENALTY = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════════════
# PREPROCESSING (self-contained)
# ═══════════════════════════════════════════════════════════════════════════
_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₓ", "0123456789x")
_GLOTTAL_CHARS = "\u02BE\u02BF\u02BC\u02BB\u0027\u2018\u2019"
_VOWEL_ACCENTS = [
    (re.compile(r'a2\b'), 'á'), (re.compile(r'a3\b'), 'à'),
    (re.compile(r'e2\b'), 'é'), (re.compile(r'e3\b'), 'è'),
    (re.compile(r'i2\b'), 'í'), (re.compile(r'i3\b'), 'ì'),
    (re.compile(r'u2\b'), 'ú'), (re.compile(r'u3\b'), 'ù'),
]


def clean_transliteration(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("Ḫ", "H").replace("ḫ", "h")
    for ch in _GLOTTAL_CHARS:
        text = text.replace(ch, "")
    text = text.translate(_SUBSCRIPT_MAP)
    text = re.sub(r'\{large break\}', '<big_gap>', text, flags=re.IGNORECASE)
    text = re.sub(r'\[…\s*…\]', '<big_gap>', text)
    text = re.sub(r'…', '<big_gap>', text)
    text = re.sub(r'\[(?:x\s*)+\]', '<gap>', text)
    text = re.sub(r'\[\.{2,}\]', '<gap>', text)
    text = re.sub(r'\.{3,}', '<gap>', text)
    text = re.sub(r'\[broken\]', '<gap>', text, flags=re.IGNORECASE)
    text = re.sub(r'[˹⸢˺⸣]', '', text)
    text = re.sub(r'<<[^>]*>>', '', text)
    text = re.sub(r'\[([^\]]*)\]', r'\1', text)
    text = re.sub(r'<(?!gap>|big_gap>)([^>]*)>', r'\1', text)
    text = re.sub(r'[!?*#]', '', text)
    text = text.replace("sz", "š").replace("SZ", "Š")
    text = re.sub(r's,', 'ṣ', text)
    text = re.sub(r't,', 'ṭ', text)
    for p, r in _VOWEL_ACCENTS:
        text = p.sub(r, text)
    text = re.sub(r'(<big_gap>\s*){2,}', '<big_gap> ', text)
    text = re.sub(r'(<gap>\s*){2,}', '<gap> ', text)
    text = re.sub(r'(<(?:big_)?gap>\s*){2,}', '<big_gap> ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()
print("Model loaded.")

# Load test data
test_df = pd.read_csv(TEST_CSV)
print(f"Test samples: {len(test_df)}")

# Find columns
translit_col = next((c for c in test_df.columns if "translit" in c.lower()), test_df.columns[-1])
id_col = "id" if "id" in test_df.columns else test_df.columns[0]

# Clean
test_df["clean"] = test_df[translit_col].apply(clean_transliteration)
texts = test_df["clean"].tolist()

# Inference
predictions = []
for i in range(0, len(texts), BATCH_SIZE):
    batch = [t if t else "<gap>" for t in texts[i:i + BATCH_SIZE]]
    inputs = tokenizer(batch, max_length=MAX_LENGTH, truncation=True,
                       padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, num_beams=NUM_BEAMS,
                                 max_length=MAX_LENGTH, early_stopping=True,
                                 no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                                 length_penalty=LENGTH_PENALTY)
    predictions.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    if (i // BATCH_SIZE + 1) % 25 == 0:
        print(f"  {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

# Write submission
submission = pd.DataFrame({"id": test_df[id_col], "translation": predictions})
submission["translation"] = submission["translation"].fillna("")
submission.to_csv(OUTPUT_CSV, index=False)
print(f"Submission saved: {OUTPUT_CSV} ({len(submission)} rows)")
print(submission.head())
