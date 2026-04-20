"""
STEP 16 — Minimum Bayes Risk (MBR) Decoding
Generates N translation candidates per input, scores each candidate against
all others using the competition metric (geometric mean of BLEU and chrF++),
and selects the highest-scoring candidate.

Consistently improves translation quality by 1-3 points over beam search.

Usage:
  # Default: MBR with 20 samples
  python 16_mbr_decode.py

  # Custom settings
  python 16_mbr_decode.py --n-samples 30 --model checkpoints/byt5-large-sft/best_model

Output: submission_mbr.csv
"""
import os
import sys
import math
import argparse
import pandas as pd
import numpy as np
import torch
import sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import unicodedata

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# ── Inline preprocessing ──────────────────────────────────────────────────
_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₓ", "0123456789x")
_GLOTTAL_CHARS = "\u02BE\u02BF\u02BC\u02BB\u0027\u2018\u2019"
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
    text = re.sub(r'\[damaged\]', '<gap>', text, flags=re.IGNORECASE)
    text = re.sub(r'[˹⸢˺⸣]', '', text)
    text = re.sub(r'<<[^>]*>>', '', text)
    text = re.sub(r'\[([^\]]*)\]', r'\1', text)
    text = re.sub(r'<(?!gap>|big_gap>)([^>]*)>', r'\1', text)
    text = re.sub(r'[!?*#]', '', text)
    text = text.replace("sz", "š").replace("SZ", "Š")
    text = re.sub(r's,', 'ṣ', text)
    text = re.sub(r'S,', 'Ṣ', text)
    text = re.sub(r't,', 'ṭ', text)
    text = re.sub(r'T,', 'Ṭ', text)
    for pattern, replacement in _VOWEL_ACCENT_PATTERNS:
        text = pattern.sub(replacement, text)
    text = re.sub(r'(<big_gap>\s*){2,}', '<big_gap> ', text)
    text = re.sub(r'(<gap>\s*){2,}', '<gap> ', text)
    text = re.sub(r'(<(?:big_)?gap>\s*){2,}', '<big_gap> ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_sentence_score(hypothesis: str, references: list[str]) -> float:
    """Score a single hypothesis against multiple references using geo_mean(BLEU, chrF++)."""
    if not hypothesis.strip() or not references:
        return 0.0

    # Use sentence-level BLEU and chrF++ against each reference, then average
    scores = []
    for ref in references:
        if not ref.strip():
            continue
        try:
            bleu = sacrebleu.sentence_bleu(hypothesis, [ref]).score
            chrf = sacrebleu.sentence_chrf(hypothesis, [ref], word_order=2).score
            if bleu > 0 and chrf > 0:
                geo = math.sqrt(bleu * chrf)
            else:
                geo = 0.0
            scores.append(geo)
        except Exception:
            scores.append(0.0)

    return np.mean(scores) if scores else 0.0


def mbr_select(candidates: list[str]) -> str:
    """Select best candidate using Minimum Bayes Risk criterion.
    For each candidate, compute average score against all other candidates.
    Return the candidate with highest average score.
    """
    if len(candidates) == 0:
        return ""
    if len(candidates) == 1:
        return candidates[0]

    # Deduplicate while preserving order
    unique = list(dict.fromkeys(candidates))
    if len(unique) == 1:
        return unique[0]

    best_score = -1
    best_candidate = unique[0]

    for i, hyp in enumerate(unique):
        # Score this hypothesis against all others
        others = [c for j, c in enumerate(unique) if j != i]
        score = compute_sentence_score(hyp, others)
        if score > best_score:
            best_score = score
            best_candidate = hyp

    return best_candidate


def generate_candidates(
    model,
    tokenizer,
    texts: list[str],
    n_samples: int,
    device: torch.device,
    max_length: int = 512,
) -> list[list[str]]:
    """Generate N candidate translations per input using sampling."""
    all_candidates = [[] for _ in range(len(texts))]

    batch_texts = [t if t else "<gap>" for t in texts]
    inputs = tokenizer(
        batch_texts, max_length=max_length, truncation=True,
        padding=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        # Generate via sampling
        for _ in range(n_samples):
            outputs = model.generate(
                **inputs,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                max_length=max_length,
                num_return_sequences=1,
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for i, d in enumerate(decoded):
                all_candidates[i].append(d.strip())

        # Also add beam search result for stability
        beam_outputs = model.generate(
            **inputs,
            num_beams=8,
            max_length=max_length,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
        )
        beam_decoded = tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
        for i, d in enumerate(beam_decoded):
            all_candidates[i].append(d.strip())

    return all_candidates


def main():
    parser = argparse.ArgumentParser(description="MBR Decoding for Akkadian MT")
    parser.add_argument(
        "--model",
        default=os.path.join(PROJECT_DIR, "checkpoints", "byt5-base", "best_model"),
        help="Model checkpoint path"
    )
    parser.add_argument(
        "--test-csv",
        default=os.path.join(DATA_DIR, "test.csv"),
        help="Test CSV"
    )
    parser.add_argument("--n-samples", type=int, default=20, help="Number of candidates per input")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for generation")
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_DIR, "submission_mbr.csv"),
        help="Output submission CSV"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"N samples: {args.n_samples}")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    model.eval()

    # Load test data
    test_df = pd.read_csv(args.test_csv)
    translit_col = next((c for c in test_df.columns if "translit" in c.lower()), test_df.columns[-1])
    id_col = "id" if "id" in test_df.columns else test_df.columns[0]

    test_df["clean"] = test_df[translit_col].apply(clean_transliteration)
    texts = test_df["clean"].tolist()
    print(f"Test samples: {len(texts)}")

    # Generate and select via MBR
    all_predictions = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i:i + args.batch_size]

        # Generate N candidates per input
        candidates = generate_candidates(
            model, tokenizer, batch, args.n_samples, device
        )

        # MBR selection
        for j, cands in enumerate(candidates):
            best = mbr_select(cands)
            all_predictions.append(best)

        done = min(i + args.batch_size, len(texts))
        if (i // args.batch_size + 1) % 5 == 0:
            print(f"  {done}/{len(texts)} samples processed")

    # Save submission
    submission = pd.DataFrame({
        "id": test_df[id_col],
        "translation": all_predictions,
    })
    submission["translation"] = submission["translation"].fillna("")
    submission.to_csv(args.output, index=False)
    print(f"\nMBR submission saved: {args.output} ({len(submission)} rows)")

    # Show samples
    print("\nSample predictions:")
    for _, row in submission.head(5).iterrows():
        print(f"  [{row['id']}]: {row['translation'][:100]}...")


if __name__ == "__main__":
    main()
