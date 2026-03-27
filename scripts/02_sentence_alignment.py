"""
STEP 3 — Sentence Alignment
Split document-level training data into sentence-level pairs.

Data schema (from script 1 output):
  train.csv:      ['oare_id', 'transliteration', 'translation']  — document-level
  Sentences CSV:  ['display_name', 'text_uuid', 'sentence_uuid', 'sentence_obj_in_text',
                   'translation', 'first_word_transcription', 'first_word_spelling',
                   'first_word_number', 'first_word_obj_in_text', 'line_number', 'side', 'column']

Key join: train.oare_id == sentences.text_uuid
The sentences CSV already has per-sentence English translations and line numbers.
"""
import os
import re
import sys
import pandas as pd
import numpy as np

# Add parent dir to path so we can import preprocess.py from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from preprocess import clean_transliteration, clean_translation

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def split_transliteration_by_spaces(text: str) -> list[str]:
    """Split a document transliteration into whitespace-separated words."""
    if not isinstance(text, str) or not text.strip():
        return []
    return text.split()


def split_english_sentences(text: str) -> list[str]:
    """Split English translation into sentences at period + capital letter boundaries."""
    if not isinstance(text, str) or not text.strip():
        return []
    sentences = re.split(r'(?<=\.)\s+(?=[A-Z])', text)
    result = []
    for s in sentences:
        parts = s.split('\n')
        result.extend(p.strip() for p in parts if p.strip())
    return result


def align_with_sentences_csv(oare_id, doc_translit, doc_translation, doc_sentences):
    """
    Align using the Sentences_Oare_FirstWord_LinNum.csv data.

    Each row in doc_sentences has:
      - translation: the English sentence
      - first_word_number: the 1-based word index where this sentence starts
                           in the document transliteration
      - line_number: the tablet line number where this sentence starts
      - sentence_obj_in_text: ordering of sentences within the document

    Strategy: use first_word_number to slice the document transliteration
    into segments corresponding to each sentence.
    """
    # Sort sentences by their order in the text
    doc_sentences = doc_sentences.sort_values("sentence_obj_in_text").reset_index(drop=True)

    # Filter to rows that have both a translation and word position
    valid = doc_sentences.dropna(subset=["translation"])
    if len(valid) == 0:
        return []

    all_words = split_transliteration_by_spaces(doc_translit)
    n_words = len(all_words)

    pairs = []

    # Get word start positions (1-based from CSV → 0-based)
    word_starts = []
    for _, row in valid.iterrows():
        eng = str(row["translation"]).strip()
        word_num = row.get("first_word_number")
        if pd.notna(word_num):
            word_starts.append((int(word_num) - 1, eng))  # convert to 0-based
        else:
            word_starts.append((None, eng))

    # Build pairs using word boundaries
    for i, (start_idx, eng_sentence) in enumerate(word_starts):
        if not eng_sentence:
            continue

        if start_idx is not None and n_words > 0:
            # Determine end: next sentence's start, or end of document
            end_idx = n_words
            for j in range(i + 1, len(word_starts)):
                if word_starts[j][0] is not None:
                    end_idx = word_starts[j][0]
                    break

            start_idx = max(0, min(start_idx, n_words - 1))
            end_idx = max(start_idx + 1, min(end_idx, n_words))
            akk_segment = " ".join(all_words[start_idx:end_idx])
        else:
            # No word position — can't extract Akkadian segment
            akk_segment = ""

        if akk_segment.strip() and eng_sentence.strip():
            pairs.append({
                "oare_id": oare_id,
                "transliteration": akk_segment,
                "translation": eng_sentence,
            })

    return pairs


def proportional_align(oare_id, doc_translit, doc_translation):
    """Fallback: split English at sentence boundaries, distribute Akkadian proportionally."""
    eng_sentences = split_english_sentences(doc_translation)
    all_words = split_transliteration_by_spaces(doc_translit)
    n_words = len(all_words)
    n_eng = len(eng_sentences)

    if n_eng == 0 or n_words == 0:
        if doc_translit.strip() and doc_translation.strip():
            return [{"oare_id": oare_id,
                     "transliteration": doc_translit.replace('\n', ' '),
                     "translation": doc_translation.replace('\n', ' ')}]
        return []

    if n_eng == 1:
        return [{"oare_id": oare_id,
                 "transliteration": " ".join(all_words),
                 "translation": eng_sentences[0]}]

    # Distribute words proportionally by English sentence length
    eng_lens = [max(len(s), 1) for s in eng_sentences]
    total_len = sum(eng_lens)

    allocs = [max(1, round(n_words * l / total_len)) for l in eng_lens]

    # Adjust to match total word count
    while sum(allocs) > n_words and max(allocs) > 1:
        allocs[allocs.index(max(allocs))] -= 1
    while sum(allocs) < n_words:
        allocs[allocs.index(min(allocs))] += 1

    pairs = []
    word_idx = 0
    for i, n_w in enumerate(allocs):
        end_idx = min(word_idx + n_w, n_words)
        akk_segment = " ".join(all_words[word_idx:end_idx])
        if akk_segment.strip() and eng_sentences[i].strip():
            pairs.append({
                "oare_id": oare_id,
                "transliteration": akk_segment,
                "translation": eng_sentences[i],
            })
        word_idx = end_idx

    return pairs


def main():
    print("Loading data...")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    sentences_df = pd.read_csv(os.path.join(DATA_DIR, "Sentences_Oare_FirstWord_LinNum.csv"))

    print(f"Train documents: {len(train)}")
    print(f"Sentences CSV: {sentences_df.shape}")
    print(f"Sentences CSV columns: {list(sentences_df.columns)}")

    # Build lookup: text_uuid → sentence rows
    # Join key: train.oare_id == sentences.text_uuid
    sentences_by_doc = sentences_df.groupby("text_uuid")
    matched_uuids = set(sentences_df["text_uuid"].unique()) & set(train["oare_id"].unique())
    print(f"Documents with sentence data: {len(matched_uuids)} / {len(train)}")

    # ── Align all documents ────────────────────────────────────────────────
    all_pairs = []
    aligned_with_csv = 0
    aligned_proportional = 0
    skipped = 0

    for idx, row in train.iterrows():
        oare_id = str(row["oare_id"])
        translit = str(row["transliteration"]) if pd.notna(row["transliteration"]) else ""
        translation = str(row["translation"]) if pd.notna(row["translation"]) else ""

        if not translit.strip() or not translation.strip():
            skipped += 1
            continue

        # Try sentence CSV alignment first
        if oare_id in matched_uuids:
            doc_sents = sentences_by_doc.get_group(oare_id)
            pairs = align_with_sentences_csv(oare_id, translit, translation, doc_sents)
            if pairs:
                aligned_with_csv += 1
                all_pairs.extend(pairs)
                continue

        # Fallback: proportional alignment
        pairs = proportional_align(oare_id, translit, translation)
        if pairs:
            aligned_proportional += 1
            all_pairs.extend(pairs)

        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1}/{len(train)} documents, "
                  f"{len(all_pairs)} sentence pairs so far")

    print(f"\nAlignment summary:")
    print(f"  Aligned with sentence CSV: {aligned_with_csv}")
    print(f"  Aligned with proportional fallback: {aligned_proportional}")
    print(f"  Skipped (empty): {skipped}")
    print(f"  Total sentence pairs: {len(all_pairs)}")

    # ── Create DataFrame and clean ─────────────────────────────────────────
    aligned = pd.DataFrame(all_pairs)

    # Apply text normalization
    print("Applying text normalization...")
    aligned["transliteration"] = aligned["transliteration"].apply(clean_transliteration)
    aligned["translation"] = aligned["translation"].apply(clean_translation)

    # Remove empty pairs after cleaning
    aligned = aligned[
        (aligned["transliteration"].str.len() > 0) &
        (aligned["translation"].str.len() > 0)
    ].reset_index(drop=True)

    print(f"After cleaning: {len(aligned)} pairs")
    print(f"\nTransliteration length stats:\n{aligned['transliteration'].str.len().describe()}")
    print(f"\nTranslation length stats:\n{aligned['translation'].str.len().describe()}")

    # ── Train/val split (90/10) ────────────────────────────────────────────
    np.random.seed(42)
    val_mask = np.random.rand(len(aligned)) < 0.1
    val_df = aligned[val_mask].reset_index(drop=True)
    train_df = aligned[~val_mask].reset_index(drop=True)

    print(f"\nTrain set: {len(train_df)} pairs")
    print(f"Val set:   {len(val_df)} pairs")

    # ── Save ───────────────────────────────────────────────────────────────
    aligned.to_csv(os.path.join(DATA_DIR, "aligned_train.csv"), index=False)
    train_df.to_csv(os.path.join(DATA_DIR, "aligned_train_split.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, "aligned_val_split.csv"), index=False)

    print(f"\nSaved:")
    print(f"  {os.path.join(DATA_DIR, 'aligned_train.csv')} ({len(aligned)} pairs)")
    print(f"  {os.path.join(DATA_DIR, 'aligned_train_split.csv')} ({len(train_df)} pairs)")
    print(f"  {os.path.join(DATA_DIR, 'aligned_val_split.csv')} ({len(val_df)} pairs)")

    # ── Sample output ──────────────────────────────────────────────────────
    print("\n=== Sample aligned pairs ===")
    for i in range(min(5, len(aligned))):
        print(f"\n--- Pair {i+1} (doc: {aligned.iloc[i]['oare_id']}) ---")
        print(f"  AKK: {aligned.iloc[i]['transliteration'][:120]}...")
        print(f"  ENG: {aligned.iloc[i]['translation'][:120]}...")

    print("\n✓ Sentence alignment complete.")


if __name__ == "__main__":
    main()
