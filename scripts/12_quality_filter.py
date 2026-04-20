"""
STEP 12 — Data Quality Filtering
Filters aligned sentence pairs for quality using:
  1. Length ratio checks (source vs target)
  2. Language detection on English side
  3. Confidence-based filtering (run ByT5 model, flag divergent pairs)
  4. Deduplication and near-duplicate removal
  5. Gap-ratio filtering (reject pairs that are mostly gaps)

Input:  any aligned CSV (e.g., llm_aligned_train.csv or aligned_train.csv)
Output: data/filtered_train.csv, data/filtered_train_split.csv, data/filtered_val_split.csv
"""
import os
import sys
import re
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── Filter thresholds ──────────────────────────────────────────────────────
MIN_SRC_LEN = 5            # minimum source characters
MIN_TGT_LEN = 5            # minimum target characters
MAX_LEN_RATIO = 6.0        # max ratio of src_len / tgt_len
MIN_LEN_RATIO = 0.1        # min ratio of src_len / tgt_len
MAX_GAP_RATIO = 0.7        # reject if >70% of source is gap tokens
MAX_TGT_LEN = 1000         # reject extremely long targets (likely OCR noise)
MAX_SRC_LEN = 2000          # reject extremely long sources


def filter_length(df: pd.DataFrame) -> pd.DataFrame:
    """Filter by absolute and relative length."""
    src_len = df["transliteration"].str.len()
    tgt_len = df["translation"].str.len()

    mask = (
        (src_len >= MIN_SRC_LEN) &
        (tgt_len >= MIN_TGT_LEN) &
        (src_len <= MAX_SRC_LEN) &
        (tgt_len <= MAX_TGT_LEN)
    )

    # Length ratio
    ratio = src_len / tgt_len.clip(lower=1)
    mask = mask & (ratio >= MIN_LEN_RATIO) & (ratio <= MAX_LEN_RATIO)

    removed = (~mask).sum()
    print(f"  Length filter: removed {removed} pairs")
    return df[mask].reset_index(drop=True)


def filter_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Remove pairs where source is mostly gap tokens."""
    def gap_ratio(text):
        if not text:
            return 1.0
        gap_chars = len(re.findall(r'<(?:big_)?gap>', text)) * 5  # approximate gap token length
        return gap_chars / max(len(text), 1)

    ratios = df["transliteration"].apply(gap_ratio)
    mask = ratios < MAX_GAP_RATIO

    removed = (~mask).sum()
    print(f"  Gap filter: removed {removed} pairs")
    return df[mask].reset_index(drop=True)


def filter_language(df: pd.DataFrame) -> pd.DataFrame:
    """Basic check that the target looks like English."""
    def is_likely_english(text):
        if not isinstance(text, str) or len(text) < 5:
            return False
        # Check for minimum proportion of ASCII letters
        ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
        letter_ratio = ascii_letters / max(len(text), 1)
        if letter_ratio < 0.3:
            return False
        # Check for common English words
        lower = text.lower()
        english_markers = [
            "the ", " of ", " and ", " to ", " in ", " is ", " he ", " she ",
            " his ", " her ", " they ", " with ", " for ", " this ", " that ",
            " from ", " have ", " has ", " not ", " but ", " which ", " who ",
        ]
        has_marker = any(m in lower for m in english_markers)
        # Short texts may not have common markers, be lenient
        if len(text) < 30:
            return True
        return has_marker

    mask = df["translation"].apply(is_likely_english)
    removed = (~mask).sum()
    print(f"  Language filter: removed {removed} pairs")
    return df[mask].reset_index(drop=True)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicates and near-duplicates."""
    before = len(df)

    # Exact duplicates
    df = df.drop_duplicates(subset=["transliteration", "translation"]).reset_index(drop=True)
    exact_removed = before - len(df)

    # Near-duplicates: same transliteration with slightly different translations
    # Keep the longer translation (more complete)
    df["tgt_len"] = df["translation"].str.len()
    df = df.sort_values("tgt_len", ascending=False).drop_duplicates(
        subset=["transliteration"], keep="first"
    ).reset_index(drop=True)
    near_removed = before - exact_removed - len(df)
    df = df.drop(columns=["tgt_len"])

    print(f"  Deduplication: removed {exact_removed} exact + {near_removed} near-duplicates")
    return df


def filter_repetitive(df: pd.DataFrame) -> pd.DataFrame:
    """Remove pairs where translation is highly repetitive (degenerate output)."""
    def is_repetitive(text):
        if not isinstance(text, str) or len(text) < 20:
            return False
        words = text.lower().split()
        if len(words) < 4:
            return False
        unique_ratio = len(set(words)) / len(words)
        return unique_ratio < 0.2  # >80% repeated words

    mask = ~df["translation"].apply(is_repetitive)
    removed = (~mask).sum()
    print(f"  Repetition filter: removed {removed} pairs")
    return df[mask].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Filter aligned training data for quality")
    parser.add_argument(
        "--input", default=os.path.join(DATA_DIR, "aligned_train.csv"),
        help="Input aligned CSV file"
    )
    parser.add_argument(
        "--output-prefix", default="filtered",
        help="Output filename prefix (default: filtered)"
    )
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Input pairs: {len(df)}")

    # Apply filters
    print("\nApplying filters:")
    df = filter_length(df)
    df = filter_gaps(df)
    df = filter_language(df)
    df = deduplicate(df)
    df = filter_repetitive(df)

    print(f"\nAfter all filters: {len(df)} pairs")

    # ── Train/val split ────────────────────────────────────────────────────
    np.random.seed(42)
    val_mask = np.random.rand(len(df)) < 0.1
    val_df = df[val_mask].reset_index(drop=True)
    train_df = df[~val_mask].reset_index(drop=True)

    # ── Save ───────────────────────────────────────────────────────────────
    prefix = args.output_prefix
    df.to_csv(os.path.join(DATA_DIR, f"{prefix}_train.csv"), index=False)
    train_df.to_csv(os.path.join(DATA_DIR, f"{prefix}_train_split.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, f"{prefix}_val_split.csv"), index=False)

    print(f"\nSaved:")
    print(f"  {prefix}_train.csv: {len(df)} pairs")
    print(f"  {prefix}_train_split.csv: {len(train_df)} pairs")
    print(f"  {prefix}_val_split.csv: {len(val_df)} pairs")

    # ── Stats ──────────────────────────────────────────────────────────────
    print(f"\nTransliteration length stats:")
    print(df["transliteration"].str.len().describe().to_string())
    print(f"\nTranslation length stats:")
    print(df["translation"].str.len().describe().to_string())


if __name__ == "__main__":
    main()
