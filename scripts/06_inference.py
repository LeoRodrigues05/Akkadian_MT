"""
STEP 7 — Kaggle Submission Inference Script
Loads best ByT5 checkpoint, runs inference on test.csv, writes submission.csv.

This script can be run locally for testing, or converted to a Kaggle notebook.
On Kaggle: the model checkpoint must be pre-uploaded as a Kaggle dataset.
"""
import os
import sys
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Inline preprocessing (so this script is self-contained for Kaggle) ─────
import re
import unicodedata

_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

_VOWEL_ACCENT_PATTERNS = [
    (re.compile(r'a2\b'), 'á'), (re.compile(r'a3\b'), 'à'),
    (re.compile(r'e2\b'), 'é'), (re.compile(r'e3\b'), 'è'),
    (re.compile(r'i2\b'), 'í'), (re.compile(r'i3\b'), 'ì'),
    (re.compile(r'u2\b'), 'ú'), (re.compile(r'u3\b'), 'ù'),
    (re.compile(r'A2\b'), 'Á'), (re.compile(r'A3\b'), 'À'),
    (re.compile(r'E2\b'), 'É'), (re.compile(r'E3\b'), 'È'),
    (re.compile(r'I2\b'), 'Í'), (re.compile(r'I3\b'), 'Ì'),
    (re.compile(r'U2\b'), 'Ú'), (re.compile(r'U3\b'), 'Ù'),
]


def clean_transliteration(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("Ḫ", "H").replace("ḫ", "h")
    text = text.translate(_SUBSCRIPT_MAP)
    text = re.sub(r'\[(?:x\s*)+\]', '<gap>', text)
    text = re.sub(r'\[\.{2,}\]', '<gap>', text)
    text = re.sub(r'\.{3,}', '<gap>', text)
    text = re.sub(r'…', '<gap>', text)
    text = re.sub(r'\[broken\]', '<gap>', text, flags=re.IGNORECASE)
    text = re.sub(r'\[damaged\]', '<gap>', text, flags=re.IGNORECASE)
    text = re.sub(r'[˹⸢˺⸣]', '', text)
    text = re.sub(r'\[([^\]]*)\]', r'\1', text)
    text = re.sub(r'<(?!gap>)([^>]*)>', r'\1', text)
    text = re.sub(r'[!?*#]', '', text)
    text = text.replace("sz", "š").replace("SZ", "Š")
    text = re.sub(r's,', 'ṣ', text)
    text = re.sub(r'S,', 'Ṣ', text)
    text = re.sub(r't,', 'ṭ', text)
    text = re.sub(r'T,', 'Ṭ', text)
    for pattern, replacement in _VOWEL_ACCENT_PATTERNS:
        text = pattern.sub(replacement, text)
    text = re.sub(r'(<gap>\s*){2,}', '<gap> ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Configuration ──────────────────────────────────────────────────────────
# Local paths (change these for Kaggle)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")

# On Kaggle, the model is uploaded as a dataset at a path like:
# /kaggle/input/byt5-akkadian-checkpoint/best_model/
# Locally, it's in our checkpoints directory
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(PROJECT_DIR, "checkpoints", "byt5-base", "best_model")
)

# On Kaggle: /kaggle/input/deep-past-initiative-machine-translation/test.csv
# Locally: data/test.csv
TEST_CSV = os.environ.get(
    "TEST_CSV",
    os.path.join(SCRIPT_DIR, "data", "test.csv")
)

OUTPUT_CSV = os.environ.get(
    "OUTPUT_CSV",
    os.path.join(PROJECT_DIR, "submission.csv")
)

BATCH_SIZE = 16
NUM_BEAMS = 4
MAX_LENGTH = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"Device: {DEVICE}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Test CSV: {TEST_CSV}")
    print(f"Output: {OUTPUT_CSV}")

    # ── Load model ─────────────────────────────────────────────────────────
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")

    # ── Load test data ─────────────────────────────────────────────────────
    test_df = pd.read_csv(TEST_CSV)
    print(f"Test samples: {len(test_df)}")
    print(f"Test columns: {list(test_df.columns)}")

    # Find the transliteration column
    translit_col = None
    for col in test_df.columns:
        if "translit" in col.lower():
            translit_col = col
            break
    if translit_col is None:
        # Fallback: use the last column that isn't 'id'
        non_id_cols = [c for c in test_df.columns if c.lower() != "id"]
        translit_col = non_id_cols[-1] if non_id_cols else test_df.columns[-1]
    print(f"Using transliteration column: '{translit_col}'")

    # Find the ID column
    id_col = "id" if "id" in test_df.columns else test_df.columns[0]

    # ── Clean transliterations ─────────────────────────────────────────────
    test_df["clean_translit"] = test_df[translit_col].apply(clean_transliteration)

    # ── Run inference in batches ───────────────────────────────────────────
    all_predictions = []
    texts = test_df["clean_translit"].tolist()

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_texts = [t if t else "<gap>" for t in batch_texts]  # handle empty

        inputs = tokenizer(
            batch_texts,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_beams=NUM_BEAMS,
                max_length=MAX_LENGTH,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_predictions.extend(decoded)

        if (i // BATCH_SIZE + 1) % 10 == 0:
            print(f"  Processed {min(i + BATCH_SIZE, len(texts))}/{len(texts)} samples")

    print(f"Generated {len(all_predictions)} predictions")

    # ── Create submission ──────────────────────────────────────────────────
    submission = pd.DataFrame({
        "id": test_df[id_col],
        "translation": all_predictions,
    })

    # Ensure no NaN translations
    submission["translation"] = submission["translation"].fillna("")

    submission.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSubmission saved to: {OUTPUT_CSV}")
    print(f"Shape: {submission.shape}")
    print(f"\nFirst 5 predictions:")
    for i, row in submission.head(5).iterrows():
        print(f"  [{row['id']}]: {row['translation'][:100]}...")

    # ── Verify format ──────────────────────────────────────────────────────
    sample_sub_path = os.path.join(SCRIPT_DIR, "data", "sample_submission.csv")
    if os.path.exists(sample_sub_path):
        sample = pd.read_csv(sample_sub_path)
        print(f"\nFormat check:")
        print(f"  Expected columns: {list(sample.columns)}")
        print(f"  Got columns:      {list(submission.columns)}")
        print(f"  Expected rows:    {len(sample)}")
        print(f"  Got rows:         {len(submission)}")
        if list(sample.columns) == list(submission.columns):
            print("  ✓ Column format matches!")
        else:
            print("  ✗ Column mismatch — check sample_submission.csv")
    else:
        print("\n(sample_submission.csv not found for format verification)")

    print("\n✓ Inference complete.")


if __name__ == "__main__":
    main()
