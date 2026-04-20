"""
Sentence-level alignment of 1,308 Old Assyrian documents.

For each document NOT in Sentences_Oare_FirstWord_LinNum.csv we:
  1. Detect structural anchors in both Akkadian transliteration and
     English translation.
  2. Align anchors greedily left-to-right by label.
  3. Pair text spans between consecutive aligned anchor positions.
  4. Fall back to proportional splitting when no anchors match.
  5. Apply quality filters.

Outputs
-------
scripts/data/llm_aligned_pairs.csv         – new pairs from the 1,308 docs
scripts/data/llm_aligned_train.csv         – combined with aligned_train.csv
scripts/data/llm_aligned_train_split.csv   – 90% train
scripts/data/llm_aligned_val_split.csv     – 10% val
"""

import re
import random
import pathlib
from typing import List, Tuple

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE = pathlib.Path(__file__).parent
DATA = BASE / "data"

TRAIN_CSV          = DATA / "train.csv"
SENTENCES_CSV      = DATA / "Sentences_Oare_FirstWord_LinNum.csv"
ALIGNED_TRAIN_CSV  = DATA / "aligned_train.csv"

OUT_NEW_PAIRS   = DATA / "llm_aligned_pairs.csv"
OUT_COMBINED    = DATA / "llm_aligned_train.csv"
OUT_TRAIN_SPLIT = DATA / "llm_aligned_train_split.csv"
OUT_VAL_SPLIT   = DATA / "llm_aligned_val_split.csv"

RANDOM_SEED = 42

# ──────────────────────────────────────────────────────────────────────────────
# Anchor patterns  (label → (akkadian_regex, english_regex))
# ──────────────────────────────────────────────────────────────────────────────
ANCHOR_DEFS = [
    # Seals
    ("KISIB",
     re.compile(r'\bKIŠIB\b', re.IGNORECASE),
     re.compile(r'\bSeal\s+of\b', re.IGNORECASE)),
    # Letter opening: um-ma … qí-bi-ma → From … to …:
    ("UMMA_QIBI",
     re.compile(r'\bum-ma\b.{1,120}?\bqí-bi[₄4]?-ma\b', re.IGNORECASE | re.DOTALL),
     re.compile(r'\bFrom\b.{1,120}?:', re.IGNORECASE | re.DOTALL)),
    # Reported speech: um-ma X-ma → Thus X: / says:
    ("UMMA",
     re.compile(r'\bum-ma\b', re.IGNORECASE),
     re.compile(r'\b(?:Thus|says?|said)\b', re.IGNORECASE)),
    # Address formula: a-na X qí-bi-ma → To X:
    ("ANA_QIBI",
     re.compile(r'\ba-na\b.{1,80}?\bqí-bi[₄4]?-ma\b', re.IGNORECASE | re.DOTALL),
     re.compile(r'\bTo\b.{1,80}?:', re.IGNORECASE | re.DOTALL)),
    # Witness list
    ("IGI",
     re.compile(r'\bIGI\b', re.IGNORECASE),
     re.compile(r'\b(?:Witnessed\s+by|Witnesses?:?|Before)\b', re.IGNORECASE)),
    # Eponym / date formula
    ("LIMUM",
     re.compile(r'\bli-mu(?:-um)?\b', re.IGNORECASE),
     re.compile(r'\beponymy?\b|\beponymate\b', re.IGNORECASE)),
]


def _find_anchors(text: str, pats) -> List[Tuple[int, str, int, int]]:
    """Return (start, label, match_start, match_end) sorted by start, deduplicated."""
    found = []
    for label, pat in pats:
        for m in pat.finditer(text):
            found.append((m.start(), label, m.start(), m.end()))
    found.sort(key=lambda x: x[0])
    # Remove overlapping same-label duplicates
    result, last_end_by_label = [], {}
    for start, label, s, e in found:
        if start < last_end_by_label.get(label, -1):
            continue
        last_end_by_label[label] = e
        result.append((start, label, s, e))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# English sentence splitter
# ──────────────────────────────────────────────────────────────────────────────
_EN_SPLIT_RE = re.compile(
    r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÀÈÌÒÙÄËÏÖÜÑ\u0100-\u024f])'
    r'|(?<=:)\s+(?=[A-ZÁÉÍÓÚÀÈÌÒÙÄËÏÖÜÑ\u0100-\u024f])',
    re.UNICODE,
)

def split_english(text: str) -> List[str]:
    text = re.sub(r'\s+', ' ', text.strip().strip('"'))
    parts = _EN_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# Akkadian structural splitter
# ──────────────────────────────────────────────────────────────────────────────
_AK_STRUCT_RE = re.compile(
    r'(?=\bKIŠIB\b|\bIGI\b|\bli-mu(?:-um)?\b|\bum-ma\b)',
    re.IGNORECASE,
)

def split_akkadian(text: str) -> List[str]:
    text = re.sub(r'\s+', ' ', text.strip())
    parts = _AK_STRUCT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# Anchor-based alignment
# ──────────────────────────────────────────────────────────────────────────────
def _anchored_align(ak: str, en: str) -> List[Tuple[str, str]]:
    ak_anchors = _find_anchors(ak, [(l, p) for l, p, _ in ANCHOR_DEFS])
    en_anchors = _find_anchors(en, [(l, p) for l, _, p in ANCHOR_DEFS])

    if not ak_anchors or not en_anchors:
        return []

    # Greedy label-matching
    paired = []
    ei = 0
    for ak_a in ak_anchors:
        for j in range(ei, len(en_anchors)):
            if en_anchors[j][1] == ak_a[1]:
                paired.append((ak_a, en_anchors[j]))
                ei = j + 1
                break

    if not paired:
        return []

    ak_used = [a for a, _ in paired]
    en_used = [a for _, a in paired]

    results = []

    # Pre-anchor prefix
    ak_pre = ak[:ak_used[0][2]].strip()
    en_pre = en[:en_used[0][2]].strip()
    if ak_pre and en_pre:
        results.append((ak_pre, en_pre))

    # Each anchor span
    for i, (ak_a, en_a) in enumerate(paired):
        ak_start = ak_a[2]
        en_start = en_a[2]
        ak_end   = ak_used[i + 1][2] if i + 1 < len(ak_used) else len(ak)
        en_end   = en_used[i + 1][2] if i + 1 < len(en_used) else len(en)
        ak_seg = ak[ak_start:ak_end].strip()
        en_seg = en[en_start:en_end].strip()
        if ak_seg and en_seg:
            results.append((ak_seg, en_seg))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Proportional fallback
# ──────────────────────────────────────────────────────────────────────────────
def _proportional_align(ak: str, en: str) -> List[Tuple[str, str]]:
    en_sents = split_english(en)
    ak_words = ak.split()
    if not en_sents or not ak_words:
        return [(ak, en)]

    en_lengths = [len(s.split()) for s in en_sents]
    total_en = sum(en_lengths)
    if total_en == 0:
        return [(ak, en)]

    results, ak_idx = [], 0
    for i, en_sent in enumerate(en_sents):
        if i == len(en_sents) - 1:
            ak_seg = ' '.join(ak_words[ak_idx:])
        else:
            n = max(1, round(en_lengths[i] / total_en * len(ak_words)))
            ak_seg = ' '.join(ak_words[ak_idx:ak_idx + n])
            ak_idx += n
        if ak_seg.strip() and en_sent.strip():
            results.append((ak_seg.strip(), en_sent.strip()))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main alignment dispatcher
# ──────────────────────────────────────────────────────────────────────────────
def align_document(ak: str, en: str) -> List[Tuple[str, str]]:
    ak = str(ak).strip() if pd.notna(ak) else ''
    en = str(en).strip() if pd.notna(en) else ''
    if not ak or not en:
        return []

    pairs = _anchored_align(ak, en)
    if pairs:
        return pairs

    en_sents = split_english(en)
    if len(en_sents) <= 1:
        return [(ak, en)]

    ak_segs = split_akkadian(ak)
    if 1 < len(ak_segs) == len(en_sents):
        return list(zip(ak_segs, en_sents))

    return _proportional_align(ak, en)


# ──────────────────────────────────────────────────────────────────────────────
# Quality filter
# ──────────────────────────────────────────────────────────────────────────────
_GAP_RE = re.compile(r'<gap>', re.IGNORECASE)

def passes_quality(ak: str, en: str) -> bool:
    if len(ak) < 5 or len(en) < 5:
        return False
    ak_w = len(ak.split())
    en_w = len(en.split())
    if en_w == 0:
        return False
    ratio = ak_w / en_w
    if ratio > 8.0 or ratio < 0.1:
        return False
    gaps = len(_GAP_RE.findall(ak))
    if ak_w > 0 and gaps / ak_w > 0.70:
        return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    random.seed(RANDOM_SEED)

    print("Loading data …")
    train_df     = pd.read_csv(TRAIN_CSV)
    sentences_df = pd.read_csv(SENTENCES_CSV)
    aligned_df   = pd.read_csv(ALIGNED_TRAIN_CSV)

    sentence_ids = set(sentences_df['text_uuid'].dropna())
    needs_align  = train_df[~train_df['oare_id'].isin(sentence_ids)].copy()

    print(f"  train docs total:     {len(train_df):>6}")
    print(f"  already aligned docs: {len(train_df) - len(needs_align):>6}")
    print(f"  docs to align:        {len(needs_align):>6}")

    # ──────────────────────────────────────────────────────────────────────
    # Align the 1,308 documents
    # ──────────────────────────────────────────────────────────────────────
    new_pairs: List[Tuple[str, str]] = []
    seen: set = set()
    docs_with_pairs = 0

    for _, row in needs_align.iterrows():
        ak = str(row['transliteration']) if pd.notna(row['transliteration']) else ''
        en = str(row['translation'])     if pd.notna(row['translation'])     else ''
        if not ak or not en:
            continue

        pairs = align_document(ak, en)
        doc_contributed = False
        for ak_seg, en_seg in pairs:
            ak_seg, en_seg = ak_seg.strip(), en_seg.strip()
            key = (ak_seg, en_seg)
            if key in seen or not passes_quality(ak_seg, en_seg):
                continue
            seen.add(key)
            new_pairs.append(key)
            doc_contributed = True
        if doc_contributed:
            docs_with_pairs += 1

    print(f"  new pairs generated:  {len(new_pairs):>6}")
    print(f"  docs contributing ≥1 pair: {docs_with_pairs} / {len(needs_align)}")

    # ──────────────────────────────────────────────────────────────────────
    # Write llm_aligned_pairs.csv
    # ──────────────────────────────────────────────────────────────────────
    new_df = pd.DataFrame(new_pairs, columns=['transliteration', 'translation'])
    new_df.to_csv(OUT_NEW_PAIRS, index=False)
    print(f"\n  Written: {OUT_NEW_PAIRS.name}  ({len(new_df)} rows)")

    # ──────────────────────────────────────────────────────────────────────
    # Combine with existing aligned_train.csv
    # ──────────────────────────────────────────────────────────────────────
    existing = aligned_df[['transliteration', 'translation']].copy()
    # Remove duplicates already captured in new_pairs
    new_df_set = set(zip(new_df['transliteration'], new_df['translation']))
    existing = existing[
        ~existing.apply(
            lambda r: (r['transliteration'], r['translation']) in new_df_set, axis=1
        )
    ]

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.dropna(subset=['transliteration', 'translation'])
    combined = combined[combined['transliteration'].str.strip() != '']
    combined = combined[combined['translation'].str.strip() != '']
    combined = combined.drop_duplicates(subset=['transliteration', 'translation'])
    combined = combined.reset_index(drop=True)
    combined.to_csv(OUT_COMBINED, index=False)
    print(f"  Written: {OUT_COMBINED.name}  ({len(combined)} rows)")

    # ──────────────────────────────────────────────────────────────────────
    # 90 / 10 train / val split
    # ──────────────────────────────────────────────────────────────────────
    idx = list(range(len(combined)))
    random.shuffle(idx)
    cut = int(0.9 * len(idx))
    train_idx = sorted(idx[:cut])
    val_idx   = sorted(idx[cut:])

    train_split = combined.iloc[train_idx].reset_index(drop=True)
    val_split   = combined.iloc[val_idx].reset_index(drop=True)

    train_split.to_csv(OUT_TRAIN_SPLIT, index=False)
    val_split.to_csv(OUT_VAL_SPLIT, index=False)
    print(f"  Written: {OUT_TRAIN_SPLIT.name}  ({len(train_split)} rows)")
    print(f"  Written: {OUT_VAL_SPLIT.name}   ({len(val_split)} rows)")

    print("\nDone.")


if __name__ == '__main__':
    main()
