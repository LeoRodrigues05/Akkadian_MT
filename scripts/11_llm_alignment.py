"""
STEP 11 — LLM-Assisted Sentence Alignment
Uses an LLM API (Google Gemini free tier or OpenAI) to improve sentence alignment
of document-level train.csv pairs into sentence-level pairs.

Current alignment:
  - 253 docs have explicit alignment via Sentences_Oare_FirstWord_LinNum.csv
  - ~1,300 docs rely on noisy proportional fallback
  - This is the single biggest source of training noise

Strategy:
  1. For each document-level pair (transliteration, translation):
     - Send both to LLM with structured prompt
     - Ask LLM to split into aligned sentence pairs
     - Parse JSON response
  2. Quality-check the LLM output (length ratios, completeness)
  3. Merge with existing CSV-aligned data
  4. Re-split into train/val

Requires:
  - GEMINI_API_KEY or OPENAI_API_KEY environment variable
  - pip install google-genai  (for Gemini) or openai (for OpenAI)

Output:
  data/llm_aligned_train.csv     — all LLM-aligned sentence pairs
  data/llm_aligned_train_split.csv
  data/llm_aligned_val_split.csv
"""
import os
import sys
import json
import time
import re
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from preprocess import clean_transliteration, clean_translation

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── LLM Configuration ─────────────────────────────────────────────────────
# Set one of these environment variables:
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Rate limiting
REQUESTS_PER_MINUTE = 14  # Gemini free tier: 15 RPM, leave buffer
DELAY_BETWEEN_REQUESTS = 60.0 / REQUESTS_PER_MINUTE

ALIGNMENT_PROMPT = """You are an expert in Akkadian (Old Assyrian) transliterations and their English translations.

I will give you a document-level transliteration and its English translation. Your task is to split them into aligned sentence pairs.

TRANSLITERATION:
{transliteration}

ENGLISH TRANSLATION:
{translation}

Instructions:
1. Split the transliteration into logical sentence segments (using line breaks, sentence boundaries, or natural Akkadian phrase boundaries like "um-ma ... -ma" for speech openings).
2. Match each transliteration segment to its corresponding English sentence.
3. Ensure every part of the transliteration and translation is accounted for.
4. If a segment cannot be aligned, skip it.

Return ONLY a JSON array of objects, each with "akkadian" and "english" keys:
[
  {{"akkadian": "...", "english": "..."}},
  {{"akkadian": "...", "english": "..."}}
]

Return ONLY the JSON array, no other text."""


def call_gemini(prompt: str) -> str:
    """Call Google Gemini API."""
    try:
        from google import genai
    except ImportError:
        raise ImportError("Install google-genai: pip install google-genai")

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return response.text


def call_openai(prompt: str) -> str:
    """Call OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install openai: pip install openai")

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=4096,
    )
    return response.choices[0].message.content


def call_llm(prompt: str) -> str:
    """Route to available LLM API."""
    if GEMINI_API_KEY:
        return call_gemini(prompt)
    elif OPENAI_API_KEY:
        return call_openai(prompt)
    else:
        raise ValueError(
            "Set GEMINI_API_KEY or OPENAI_API_KEY environment variable. "
            "Gemini free tier: https://aistudio.google.com/app/apikey"
        )


def parse_llm_response(response: str) -> list[dict]:
    """Parse LLM JSON response into list of aligned pairs."""
    if not response:
        return []

    # Try to extract JSON from response (may have markdown code fences)
    text = response.strip()
    # Remove markdown code fences
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)

    try:
        pairs = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON array in response
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                pairs = json.loads(match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(pairs, list):
        return []

    result = []
    for pair in pairs:
        if isinstance(pair, dict):
            akk = str(pair.get("akkadian", "")).strip()
            eng = str(pair.get("english", "")).strip()
            if akk and eng and len(akk) > 3 and len(eng) > 3:
                result.append({"transliteration": akk, "translation": eng})

    return result


def quality_check_pair(akk: str, eng: str) -> bool:
    """Basic quality check for an aligned pair."""
    if not akk or not eng:
        return False

    akk_len = len(akk)
    eng_len = len(eng)

    # Length ratio check: reject extremely mismatched pairs
    if akk_len > 0 and eng_len > 0:
        ratio = akk_len / eng_len
        if ratio > 8.0 or ratio < 0.1:
            return False

    # Reject if either side is too short
    if akk_len < 5 or eng_len < 5:
        return False

    return True


def align_document_with_llm(oare_id: str, translit: str, translation: str) -> list[dict]:
    """Align a single document using the LLM."""
    prompt = ALIGNMENT_PROMPT.format(
        transliteration=translit[:3000],  # truncate to stay within context
        translation=translation[:3000],
    )

    try:
        response = call_llm(prompt)
        pairs = parse_llm_response(response)
    except Exception as e:
        print(f"  Error for {oare_id}: {e}")
        return []

    # Quality check each pair
    good_pairs = []
    for pair in pairs:
        akk = clean_transliteration(pair["transliteration"])
        eng = clean_translation(pair["translation"])
        if quality_check_pair(akk, eng):
            good_pairs.append({
                "oare_id": oare_id,
                "transliteration": akk,
                "translation": eng,
            })

    return good_pairs


def main():
    print("LLM-Assisted Sentence Alignment")
    print(f"{'='*60}")

    if not GEMINI_API_KEY and not OPENAI_API_KEY:
        print("ERROR: Set GEMINI_API_KEY or OPENAI_API_KEY environment variable.")
        print("  Gemini free tier: https://aistudio.google.com/app/apikey")
        print("  OpenAI: https://platform.openai.com/api-keys")
        return

    api_name = "Gemini" if GEMINI_API_KEY else "OpenAI"
    print(f"Using API: {api_name}")

    # ── Load data ──────────────────────────────────────────────────────────
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    sentences_df = pd.read_csv(os.path.join(DATA_DIR, "Sentences_Oare_FirstWord_LinNum.csv"))

    # Documents that already have CSV alignment
    csv_aligned_ids = set(sentences_df["text_uuid"].unique()) & set(train["oare_id"].unique())
    print(f"\nTotal documents: {len(train)}")
    print(f"Already CSV-aligned: {len(csv_aligned_ids)}")
    print(f"Need LLM alignment: {len(train) - len(csv_aligned_ids)}")

    # ── Load existing CSV-aligned pairs ────────────────────────────────────
    # Keep the existing good alignments from 02_sentence_alignment.py
    existing_aligned = pd.read_csv(os.path.join(DATA_DIR, "aligned_train.csv"))
    csv_pairs = existing_aligned[existing_aligned["oare_id"].isin(csv_aligned_ids)]
    print(f"Existing CSV-aligned pairs: {len(csv_pairs)}")

    # ── LLM-align the remaining documents ──────────────────────────────────
    docs_to_align = train[~train["oare_id"].isin(csv_aligned_ids)].copy()
    docs_to_align = docs_to_align[
        docs_to_align["transliteration"].notna() &
        docs_to_align["translation"].notna()
    ]
    print(f"\nDocuments to LLM-align: {len(docs_to_align)}")

    # Check for checkpoint (resume support)
    checkpoint_path = os.path.join(DATA_DIR, "llm_alignment_checkpoint.json")
    completed_ids = set()
    llm_pairs = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
            completed_ids = set(checkpoint.get("completed_ids", []))
            llm_pairs = checkpoint.get("pairs", [])
        print(f"Resuming from checkpoint: {len(completed_ids)} docs already done")

    # ── Process documents ──────────────────────────────────────────────────
    remaining = docs_to_align[~docs_to_align["oare_id"].isin(completed_ids)]
    total = len(remaining)
    print(f"Remaining to process: {total}")

    errors = 0
    for idx, (_, row) in enumerate(remaining.iterrows()):
        oare_id = str(row["oare_id"])
        translit = str(row["transliteration"])
        translation = str(row["translation"])

        if not translit.strip() or not translation.strip():
            completed_ids.add(oare_id)
            continue

        pairs = align_document_with_llm(oare_id, translit, translation)
        llm_pairs.extend(pairs)
        completed_ids.add(oare_id)

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{total}] {len(llm_pairs)} pairs so far "
                  f"({len(pairs)} from this doc)")

        # Save checkpoint every 50 documents
        if (idx + 1) % 50 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump({
                    "completed_ids": list(completed_ids),
                    "pairs": llm_pairs,
                }, f)
            print(f"  Checkpoint saved ({len(completed_ids)} docs)")

        # Rate limiting
        time.sleep(DELAY_BETWEEN_REQUESTS)

        if errors > 20:
            print("Too many errors, stopping.")
            break

    # ── Combine CSV-aligned + LLM-aligned ──────────────────────────────────
    print(f"\nLLM-aligned pairs: {len(llm_pairs)}")

    llm_df = pd.DataFrame(llm_pairs)
    combined = pd.concat([csv_pairs, llm_df], ignore_index=True)

    # Deduplicate
    combined = combined.drop_duplicates(
        subset=["transliteration", "translation"]
    ).reset_index(drop=True)

    # Remove empty
    combined = combined[
        (combined["transliteration"].str.len() > 0) &
        (combined["translation"].str.len() > 0)
    ].reset_index(drop=True)

    print(f"Combined aligned pairs: {len(combined)}")

    # ── Train/val split (90/10, same seed as original) ─────────────────────
    np.random.seed(42)
    val_mask = np.random.rand(len(combined)) < 0.1
    val_df = combined[val_mask].reset_index(drop=True)
    train_df = combined[~val_mask].reset_index(drop=True)

    # ── Save ───────────────────────────────────────────────────────────────
    combined.to_csv(os.path.join(DATA_DIR, "llm_aligned_train.csv"), index=False)
    train_df.to_csv(os.path.join(DATA_DIR, "llm_aligned_train_split.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, "llm_aligned_val_split.csv"), index=False)

    print(f"\nSaved:")
    print(f"  llm_aligned_train.csv: {len(combined)} pairs")
    print(f"  llm_aligned_train_split.csv: {len(train_df)} pairs")
    print(f"  llm_aligned_val_split.csv: {len(val_df)} pairs")

    # Clean up checkpoint on successful completion
    if os.path.exists(checkpoint_path) and len(remaining) == 0:
        os.remove(checkpoint_path)
        print("  Checkpoint removed (all done)")


if __name__ == "__main__":
    main()
