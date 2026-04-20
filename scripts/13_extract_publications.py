"""
STEP 13 — Extract Parallel Data from publications.csv OCR Text
Parses ~880 scholarly PDFs (OCR output) to find Akkadian transliterations and
their English translations, then aligns them with transliterations from
published_texts.csv using document identifiers.

Strategy:
  1. Load publications.csv (OCR text per page) and published_texts.csv (metadata)
  2. Build identifier cross-reference (publication_catalog, aliases → oare_id)
  3. For each publication page, detect:
     a. Akkadian transliteration blocks (hyphenated syllabic text)
     b. English translation blocks (following "Translation:" headers or similar)
  4. Match detected translations to published_texts.csv entries
  5. Output aligned pairs for training

Output:
  data/ocr_extracted_pairs.csv  — extracted transliteration-translation pairs
"""
import os
import sys
import re
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from preprocess import clean_transliteration, clean_translation

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── Patterns for detecting text types ──────────────────────────────────────
# Akkadian transliteration: contains hyphenated syllables with typical Akkadian chars
AKKADIAN_PATTERN = re.compile(
    r'(?:[a-zA-ZšŠṣṢṭṬḫḪ][a-zA-ZšŠṣṢṭṬḫḪ0-9]*-){2,}[a-zA-ZšŠṣṢṭṬḫḪ0-9]+',
    re.UNICODE
)

# Translation section headers
TRANSLATION_HEADERS = re.compile(
    r'(?:^|\n)\s*(?:Translation|Translat(?:ion)?|English|Übersetzung|Traduction)\s*[:.]?\s*\n',
    re.IGNORECASE | re.MULTILINE
)

# Line number pattern (common in transliteration blocks)
LINE_NUM_PATTERN = re.compile(r"^\s*(\d+['']*\.?\s)", re.MULTILINE)


def build_identifier_index(published_texts: pd.DataFrame) -> dict:
    """Build index: various identifiers → oare_id."""
    index = {}

    for _, row in published_texts.iterrows():
        oare_id = str(row["oare_id"])

        # Index by label
        label = str(row.get("label", ""))
        if label and label != "nan":
            index[label.strip().upper()] = oare_id

        # Index by aliases (pipe-separated)
        aliases = str(row.get("aliases", ""))
        if aliases and aliases != "nan":
            for alias in aliases.split("|"):
                alias = alias.strip().upper()
                if alias:
                    index[alias] = oare_id

        # Index by publication_catalog (pipe-separated)
        pub_cat = str(row.get("publication_catalog", ""))
        if pub_cat and pub_cat != "nan":
            for cat in pub_cat.split("|"):
                cat = cat.strip().upper()
                if cat:
                    index[cat] = oare_id

    return index


def extract_text_references(page_text: str) -> list[str]:
    """Extract publication/tablet references from a page of OCR text.
    Looks for patterns like 'ICK 1 146', 'BIN 4 45', 'CCT 1 14a', 'Kt a/k 123'.
    """
    refs = set()

    # Standard publication patterns: ABBREV NUM NUM
    pub_pattern = re.compile(
        r'\b([A-Z]{2,6})\s+(\d{1,3})\s*[,.]?\s*(\d{1,4}[a-z]?)\b'
    )
    for m in pub_pattern.finditer(page_text):
        ref = f"{m.group(1)} {m.group(2)} {m.group(3)}".upper()
        refs.add(ref)

    # Kt pattern: Kt LETTER/LETTER NUMBER
    kt_pattern = re.compile(r'\b(Kt\s+[a-z]/[a-z]\s+\d+[a-z]?)\b', re.IGNORECASE)
    for m in kt_pattern.finditer(page_text):
        refs.add(m.group(1).strip().upper())

    return list(refs)


def detect_translation_block(page_text: str) -> str:
    """Detect and extract English translation text from an OCR page."""
    # Look for explicit translation headers
    header_match = TRANSLATION_HEADERS.search(page_text)
    if header_match:
        # Take text after the header until next section or end
        start = header_match.end()
        # End at next header-like pattern or end of text
        end_pattern = re.compile(
            r'\n\s*(?:Commentary|Notes?|Transliteration|Bibliography|References|'
            r'Discussion|Analysis|Tablet|Text|Obverse|Reverse)\s*[:.]?\s*\n',
            re.IGNORECASE
        )
        end_match = end_pattern.search(page_text, start)
        end = end_match.start() if end_match else len(page_text)
        translation = page_text[start:end].strip()

        # Basic validation: should contain English words
        if len(translation) > 20 and re.search(r'[a-zA-Z]{3,}', translation):
            return translation

    return ""


def has_akkadian_content(text: str) -> bool:
    """Check if text contains Akkadian transliteration content."""
    matches = AKKADIAN_PATTERN.findall(text)
    return len(matches) >= 3  # at least 3 hyphenated Akkadian words


def extract_pairs_from_page(
    page_text: str,
    pdf_name: str,
    id_index: dict,
    published_texts: pd.DataFrame,
) -> list[dict]:
    """Extract transliteration-translation pairs from a single OCR page."""
    pairs = []

    # Find tablet references on this page
    refs = extract_text_references(page_text)
    if not refs:
        return []

    # Check if page has a translation block
    translation = detect_translation_block(page_text)
    if not translation:
        return []

    # Try to match references to published_texts
    for ref in refs:
        oare_id = id_index.get(ref)
        if not oare_id:
            continue

        # Get the transliteration from published_texts
        pt_row = published_texts[published_texts["oare_id"] == oare_id]
        if len(pt_row) == 0:
            continue

        translit = str(pt_row.iloc[0].get("transliteration", ""))
        if not translit or translit == "nan" or len(translit) < 10:
            continue

        clean_t = clean_transliteration(translit)
        clean_e = clean_translation(translation)

        if clean_t and clean_e and len(clean_t) > 10 and len(clean_e) > 10:
            pairs.append({
                "oare_id": oare_id,
                "transliteration": clean_t,
                "translation": clean_e,
                "source_pdf": pdf_name,
                "reference": ref,
            })

    return pairs


def main():
    print("Extracting parallel data from publications.csv OCR text")
    print(f"{'='*60}")

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading publications.csv...")
    publications = pd.read_csv(os.path.join(DATA_DIR, "publications.csv"))
    print(f"  Pages: {len(publications)}")
    print(f"  PDFs: {publications['pdf_name'].nunique()}")

    print("Loading published_texts.csv...")
    published_texts = pd.read_csv(os.path.join(DATA_DIR, "published_texts.csv"))
    print(f"  Texts: {len(published_texts)}")

    # Exclude texts already in train.csv
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    train_ids = set(train["oare_id"].unique())
    print(f"  Excluding {len(train_ids)} texts already in train.csv")

    # ── Build identifier index ─────────────────────────────────────────────
    print("\nBuilding identifier index...")
    id_index = build_identifier_index(published_texts)
    print(f"  Index entries: {len(id_index)}")

    # ── Filter pages with Akkadian content ─────────────────────────────────
    has_akk = publications[publications["has_akkadian"] == True]
    print(f"\nPages with Akkadian content: {len(has_akk)}")

    # ── Process pages ──────────────────────────────────────────────────────
    all_pairs = []
    pages_with_translations = 0

    for idx, row in has_akk.iterrows():
        page_text = str(row.get("page_text", ""))
        pdf_name = str(row.get("pdf_name", ""))

        if not page_text or len(page_text) < 50:
            continue

        pairs = extract_pairs_from_page(page_text, pdf_name, id_index, published_texts)
        if pairs:
            # Filter out pairs for texts already in train.csv
            pairs = [p for p in pairs if p["oare_id"] not in train_ids]
            if pairs:
                all_pairs.extend(pairs)
                pages_with_translations += 1

        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx+1}/{len(has_akk)} pages, "
                  f"{len(all_pairs)} pairs found")

    print(f"\nPages with extractable translations: {pages_with_translations}")
    print(f"Total extracted pairs: {len(all_pairs)}")

    if not all_pairs:
        print("No pairs extracted. This is expected — OCR extraction is noisy.")
        print("Consider using LLM-assisted alignment (script 11) instead.")
        # Save empty file for pipeline consistency
        pd.DataFrame(columns=["oare_id", "transliteration", "translation"]).to_csv(
            os.path.join(DATA_DIR, "ocr_extracted_pairs.csv"), index=False
        )
        return

    # ── Deduplicate ────────────────────────────────────────────────────────
    pairs_df = pd.DataFrame(all_pairs)
    pairs_df = pairs_df.drop_duplicates(
        subset=["oare_id", "transliteration"]
    ).reset_index(drop=True)

    # ── Save ───────────────────────────────────────────────────────────────
    output_path = os.path.join(DATA_DIR, "ocr_extracted_pairs.csv")
    pairs_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path} ({len(pairs_df)} pairs)")

    # ── Sample output ──────────────────────────────────────────────────────
    print(f"\nSample extracted pairs:")
    for _, row in pairs_df.head(5).iterrows():
        print(f"  [{row['reference']}] {row['transliteration'][:60]}...")
        print(f"    → {row['translation'][:80]}...")
        print()

    print(f"\nSource PDF distribution:")
    pdf_counts = pairs_df["source_pdf"].value_counts().head(10)
    for pdf, count in pdf_counts.items():
        print(f"  {pdf}: {count} pairs")


if __name__ == "__main__":
    main()
