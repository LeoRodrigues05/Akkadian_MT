"""
STEP 9 — Mine published_texts.csv for additional monolingual data
Extracts transliterations NOT already in train.csv for use in:
  - Continued Pre-Training (CPT) of ByT5 on monolingual Akkadian
  - Pseudo-labeling (translate with best model, then add to training)

Also extracts any 'note' field content that may contain partial translations.

Output:
  data/monolingual_akkadian.csv  — cleaned transliterations for CPT
  data/published_with_notes.csv  — texts that have note content (potential translations)
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from preprocess import clean_transliteration

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    print("Loading data...")
    pt = pd.read_csv(os.path.join(DATA_DIR, "published_texts.csv"))
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

    train_ids = set(train["oare_id"].unique())
    print(f"published_texts: {len(pt)} rows")
    print(f"train.csv: {len(train)} rows ({len(train_ids)} unique IDs)")

    # ── Separate already-in-train vs new ───────────────────────────────────
    pt["in_train"] = pt["oare_id"].isin(train_ids)
    new_texts = pt[~pt["in_train"]].copy()
    print(f"New texts (not in train): {len(new_texts)}")

    # ── Extract monolingual transliterations ───────────────────────────────
    # Use the 'transliteration' column (already cleaned by competition organizers)
    mono = new_texts[new_texts["transliteration"].notna()].copy()
    mono["clean_translit"] = mono["transliteration"].apply(clean_transliteration)
    mono = mono[mono["clean_translit"].str.len() > 10]  # skip very short fragments

    mono_out = mono[["oare_id", "clean_translit", "label", "genre_label"]].rename(
        columns={"clean_translit": "transliteration"}
    )
    mono_path = os.path.join(DATA_DIR, "monolingual_akkadian.csv")
    mono_out.to_csv(mono_path, index=False)
    print(f"\nMonolingual Akkadian texts: {len(mono_out)}")
    print(f"  Saved to: {mono_path}")

    # ── Extract texts with notes (potential partial translations) ──────────
    with_notes = pt[pt["note"].notna()].copy()
    # Notes are mostly bibliographic references, but some contain translations
    # Filter for notes that look like they might have translation content
    # (contain English words, longer than 50 chars, etc.)
    with_notes["note_len"] = with_notes["note"].str.len()
    promising_notes = with_notes[with_notes["note_len"] > 50].copy()

    if len(promising_notes) > 0:
        promising_notes["clean_translit"] = promising_notes["transliteration"].apply(
            clean_transliteration
        )
        notes_out = promising_notes[
            ["oare_id", "clean_translit", "note", "label", "genre_label"]
        ].rename(columns={"clean_translit": "transliteration"})
        notes_path = os.path.join(DATA_DIR, "published_with_notes.csv")
        notes_out.to_csv(notes_path, index=False)
        print(f"\nTexts with substantial notes: {len(notes_out)}")
        print(f"  Saved to: {notes_path}")
    else:
        print("\nNo texts with substantial notes found.")

    # ── Summary statistics ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Total published_texts: {len(pt)}")
    print(f"  Already in train.csv: {pt['in_train'].sum()}")
    print(f"  New monolingual texts: {len(mono_out)}")
    print(f"  Texts with notes: {len(with_notes)}")
    print(f"  Genre distribution (new texts):")
    if "genre_label" in new_texts.columns:
        genre_counts = new_texts["genre_label"].value_counts().head(10)
        for genre, count in genre_counts.items():
            print(f"    {genre}: {count}")

    # ── Transliteration length stats ───────────────────────────────────────
    print(f"\nMonolingual text length (chars):")
    print(mono_out["transliteration"].str.len().describe().to_string())


if __name__ == "__main__":
    main()
