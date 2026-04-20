"""
STEP 10 — Lexicon Integration & Dictionary Drill Generation
Uses OA_Lexicon_eBL.csv and eBL_Dictionary.csv to:
  1. Build word-level lookups (transliteration form → English meaning)
  2. Generate synthetic dictionary drill pairs for data augmentation
  3. Build proper noun lookup table for post-processing

Output:
  data/lexicon_drills.csv       — synthetic transliteration→English word pairs
  data/proper_noun_lookup.json  — PN/GN form → normalized name mapping
"""
import os
import sys
import json
import re
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from preprocess import clean_transliteration

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def parse_ebl_definition(definition: str) -> str:
    """Extract the core English meaning from an eBL dictionary entry.
    Definitions often contain grammatical notes in parentheses, references, etc.
    E.g.: '"father" (OB freq.)' → 'father'
    """
    if not isinstance(definition, str) or not definition.strip():
        return ""

    text = definition.strip()

    # Extract text in quotes first (most reliable)
    quoted = re.findall(r'"([^"]+)"', text)
    if quoted:
        # Take all quoted meanings, join with '; '
        meanings = [q.strip() for q in quoted if q.strip()]
        if meanings:
            return "; ".join(meanings[:3])  # limit to 3 meanings

    # Fallback: take text before first parenthesis or semicolon
    text = re.split(r'[;(]', text)[0].strip()
    # Remove leading numbers/roman numerals
    text = re.sub(r'^\d+\.?\s*', '', text)
    text = re.sub(r'^[IVX]+\.?\s*', '', text)

    return text.strip('" ').strip()


def build_word_lookup(oa_lexicon: pd.DataFrame, ebl_dict: pd.DataFrame) -> dict:
    """Build form → English meaning mapping by joining lexicon with dictionary."""
    # Build dictionary lookup: word → definition
    dict_lookup = {}
    for _, row in ebl_dict.iterrows():
        word = str(row["word"]).strip()
        defn = parse_ebl_definition(str(row.get("definition", "")))
        if word and defn:
            # Strip Roman numeral suffixes from word (e.g., "abum I" → "abum")
            base_word = re.sub(r'\s+[IVX]+$', '', word).strip()
            if base_word not in dict_lookup:
                dict_lookup[base_word] = defn

    print(f"Dictionary entries with parseable definitions: {len(dict_lookup)}")

    # Join lexicon forms to dictionary
    word_lookup = {}
    matched = 0
    for _, row in oa_lexicon.iterrows():
        form = str(row["form"]).strip()
        lexeme = str(row.get("lexeme", "")).strip()
        norm = str(row.get("norm", "")).strip()

        # Try to find English meaning via lexeme
        meaning = dict_lookup.get(lexeme) or dict_lookup.get(norm) or ""
        if form and meaning:
            word_lookup[form] = meaning
            matched += 1

    print(f"OA Lexicon forms matched to English: {matched}/{len(oa_lexicon)}")
    return word_lookup


def generate_drills(word_lookup: dict, oa_lexicon: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic transliteration → English drill pairs."""
    drills = []

    for form, meaning in word_lookup.items():
        if not meaning or len(meaning) < 2:
            continue

        # Clean the transliteration form
        clean_form = clean_transliteration(form)
        if not clean_form or len(clean_form) < 2:
            continue

        drills.append({
            "transliteration": clean_form,
            "translation": meaning,
        })

    df = pd.DataFrame(drills)
    # Deduplicate
    df = df.drop_duplicates(subset=["transliteration", "translation"]).reset_index(drop=True)
    return df


def build_proper_noun_lookup(oa_lexicon: pd.DataFrame) -> dict:
    """Build lookup from transliteration form → normalized proper noun."""
    pn_types = {"PN", "GN"}
    pn_rows = oa_lexicon[oa_lexicon["type"].isin(pn_types)]

    lookup = {}
    for _, row in pn_rows.iterrows():
        form = str(row["form"]).strip()
        norm = str(row.get("norm", "")).strip()
        lexeme = str(row.get("lexeme", "")).strip()
        pn_type = str(row["type"])

        if form and (norm or lexeme):
            clean_form = clean_transliteration(form)
            if clean_form:
                lookup[clean_form] = {
                    "normalized": norm if norm else lexeme,
                    "type": pn_type,
                }

    return lookup


def main():
    print("Loading lexical resources...")
    oa_lexicon = pd.read_csv(os.path.join(DATA_DIR, "OA_Lexicon_eBL.csv"))
    ebl_dict = pd.read_csv(os.path.join(DATA_DIR, "eBL_Dictionary.csv"))

    print(f"OA Lexicon: {len(oa_lexicon)} entries")
    print(f"  Types: {dict(oa_lexicon['type'].value_counts())}")
    print(f"eBL Dictionary: {len(ebl_dict)} entries")

    # ── Build word lookup ──────────────────────────────────────────────────
    print("\nBuilding word lookup...")
    word_lookup = build_word_lookup(oa_lexicon, ebl_dict)

    # ── Generate dictionary drills ─────────────────────────────────────────
    print("\nGenerating dictionary drills...")
    # Only use common words (not proper nouns) for drills
    common_words = oa_lexicon[oa_lexicon["type"] == "word"]
    common_lookup = {
        form: meaning
        for form, meaning in word_lookup.items()
        if form in set(common_words["form"])
    }
    drills = generate_drills(common_lookup, oa_lexicon)

    drills_path = os.path.join(DATA_DIR, "lexicon_drills.csv")
    drills.to_csv(drills_path, index=False)
    print(f"\nDictionary drills: {len(drills)} pairs")
    print(f"  Saved to: {drills_path}")
    if len(drills) > 0:
        print(f"\n  Sample drills:")
        for _, row in drills.head(10).iterrows():
            print(f"    {row['transliteration']} → {row['translation']}")

    # ── Build proper noun lookup ───────────────────────────────────────────
    print("\nBuilding proper noun lookup...")
    pn_lookup = build_proper_noun_lookup(oa_lexicon)

    pn_path = os.path.join(DATA_DIR, "proper_noun_lookup.json")
    with open(pn_path, "w", encoding="utf-8") as f:
        json.dump(pn_lookup, f, indent=2, ensure_ascii=False)
    print(f"Proper noun lookup: {len(pn_lookup)} entries")
    print(f"  Saved to: {pn_path}")

    # ── Summary ────────────────────────────────────────────────────────────
    pn_count = sum(1 for v in pn_lookup.values() if v["type"] == "PN")
    gn_count = sum(1 for v in pn_lookup.values() if v["type"] == "GN")
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Word lookup entries: {len(word_lookup)}")
    print(f"  Dictionary drills: {len(drills)}")
    print(f"  Proper noun lookup: {pn_count} personal names, {gn_count} geographic names")


if __name__ == "__main__":
    main()
