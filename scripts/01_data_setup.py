"""
STEP 1 — Data Setup
Downloads competition data and performs initial exploration.
"""
import os
import kagglehub
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Download competition data ──────────────────────────────────────────────
print("Downloading competition data...")
path = kagglehub.competition_download("deep-past-initiative-machine-translation")
print(f"Data downloaded to: {path}")

# ── Discover CSV files ─────────────────────────────────────────────────────
csv_files = []
for root, dirs, files in os.walk(path):
    for f in files:
        if f.endswith(".csv"):
            csv_files.append(os.path.join(root, f))
print(f"\nFound {len(csv_files)} CSV files:")
for f in csv_files:
    print(f"  {f}")

# ── Copy CSVs into our data/ directory for convenience ─────────────────────
import shutil
for src in csv_files:
    dst = os.path.join(DATA_DIR, os.path.basename(src))
    shutil.copy2(src, dst)
    print(f"Copied: {os.path.basename(src)}")

# ── Load and explore key files ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("DATA EXPLORATION")
print("=" * 60)

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
print(f"\ntrain.csv: {train.shape}")
print(f"Columns: {list(train.columns)}")
print(train.head(2).to_string())
print(f"\nTransliteration length stats:\n{train['transliteration'].str.len().describe()}")
print(f"\nTranslation length stats:\n{train['translation'].str.len().describe()}")

test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
print(f"\ntest.csv: {test.shape}")
print(f"Columns: {list(test.columns)}")
print(test.head(2).to_string())

sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
print(f"\nsample_submission.csv: {sample_sub.shape}")
print(f"Columns: {list(sample_sub.columns)}")
print(sample_sub.head(2).to_string())

sentences = pd.read_csv(os.path.join(DATA_DIR, "Sentences_Oare_FirstWord_LinNum.csv"))
print(f"\nSentences_Oare_FirstWord_LinNum.csv: {sentences.shape}")
print(f"Columns: {list(sentences.columns)}")
print(sentences.head(5).to_string())

# Check for other CSV files
for f in csv_files:
    basename = os.path.basename(f)
    if basename not in ["train.csv", "test.csv", "sample_submission.csv",
                        "Sentences_Oare_FirstWord_LinNum.csv"]:
        df = pd.read_csv(os.path.join(DATA_DIR, basename))
        print(f"\n{basename}: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(df.head(2).to_string())

print("\n✓ Data setup complete. Files saved to:", DATA_DIR)
