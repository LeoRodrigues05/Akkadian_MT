"""
STEP 14 — Data Augmentation
Combines multiple data sources and augmentation strategies:
  1. Merge filtered aligned pairs (from quality filtering)
  2. Add dictionary drill pairs (from lexicon integration)
  3. Pseudo-label monolingual Akkadian texts using best ByT5 model
  4. Template-based augmentation for formulaic patterns
  5. Output final expanded training dataset

Output:
  data/augmented_train.csv        — all augmented pairs
  data/augmented_train_split.csv  — 90% train
  data/augmented_val_split.csv    — 10% val (kept clean — no synthetic data)
"""
import os
import sys
import re
import json
import argparse
import pandas as pd
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from preprocess import clean_transliteration, clean_translation

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")


# ── Template-based augmentation ────────────────────────────────────────────
# Common Old Assyrian formulaic patterns
TEMPLATES = [
    # Debt notes
    (
        "{amount} ma-na {quality} KÙ.BABBAR i-ṣé-er {debtor} {creditor} i-šu",
        "{amount} minas of {quality_eng} silver — {debtor_eng} owes {creditor_eng}.",
    ),
    (
        "{amount} GÍN KÙ.BABBAR i-ṣé-er {debtor} {creditor} i-šu",
        "{amount} shekels of silver — {debtor_eng} owes {creditor_eng}.",
    ),
    # Letter openings
    (
        "um-ma {sender}-ma a-na {recipient} qí-bi-ma",
        'Thus {sender_eng}: Say to {recipient_eng}:',
    ),
    (
        "a-na {recipient} qí-bi-ma um-ma {sender}-ma",
        'Say to {recipient_eng}: Thus {sender_eng}:',
    ),
    # Seal inscriptions
    (
        "KIŠIB {person} DUMU {father}",
        "Seal of {person_eng} son of {father_eng}.",
    ),
    # Witness formulas
    (
        "IGI {witness1} IGI {witness2}",
        "Witness: {witness1_eng}. Witness: {witness2_eng}.",
    ),
    # Payment instructions
    (
        "{amount} ma-na KÙ.BABBAR a-na {recipient} dí-in",
        "Give {amount} minas of silver to {recipient_eng}.",
    ),
]


def load_proper_nouns(data_dir: str) -> list[tuple[str, str]]:
    """Load proper noun pairs from the lexicon lookup."""
    pn_path = os.path.join(data_dir, "proper_noun_lookup.json")
    if not os.path.exists(pn_path):
        print("  WARNING: proper_noun_lookup.json not found. Run 10_lexicon_integration.py first.")
        return []

    with open(pn_path, "r", encoding="utf-8") as f:
        pn_lookup = json.load(f)

    # Extract (akkadian_form, english_name) pairs for PN type
    names = []
    for form, info in pn_lookup.items():
        if info.get("type") == "PN" and info.get("normalized"):
            names.append((form, info["normalized"]))

    return names


def generate_template_pairs(names: list[tuple[str, str]], n_per_template: int = 100) -> list[dict]:
    """Generate synthetic pairs by filling templates with random names."""
    if not names:
        return []

    rng = np.random.RandomState(42)
    pairs = []
    amounts = ["1", "2", "3", "5", "10", "15", "20", "30"]
    qualities = [("SIG5", "refined"), ("ṣa-ru-pá-am", "refined")]

    for akk_template, eng_template in TEMPLATES:
        for _ in range(n_per_template):
            # Pick random names
            selected = rng.choice(len(names), size=min(4, len(names)), replace=False)
            name_pairs = [names[i] for i in selected]

            try:
                akk = akk_template
                eng = eng_template
                amount = rng.choice(amounts)
                quality_akk, quality_eng = qualities[rng.randint(len(qualities))]

                # Fill slots
                replacements = {
                    "amount": amount,
                    "quality": quality_akk,
                    "quality_eng": quality_eng,
                }

                # Assign names to person slots
                person_slots = ["sender", "recipient", "debtor", "creditor",
                                "person", "father", "witness1", "witness2"]
                for i, slot in enumerate(person_slots):
                    if f"{{{slot}}}" in akk:
                        if i < len(name_pairs):
                            replacements[slot] = name_pairs[i][0]
                            replacements[f"{slot}_eng"] = name_pairs[i][1]
                        else:
                            # Reuse a name if we run out
                            idx = i % len(name_pairs)
                            replacements[slot] = name_pairs[idx][0]
                            replacements[f"{slot}_eng"] = name_pairs[idx][1]

                for key, val in replacements.items():
                    akk = akk.replace(f"{{{key}}}", str(val))
                    eng = eng.replace(f"{{{key}}}", str(val))

                # Only add if all placeholders were filled
                if "{" not in akk and "{" not in eng:
                    pairs.append({
                        "transliteration": clean_transliteration(akk),
                        "translation": clean_translation(eng),
                        "source": "template",
                    })
            except (IndexError, KeyError):
                continue

    return pairs


def pseudo_label_monolingual(
    model_path: str,
    mono_csv: str,
    batch_size: int = 16,
    max_samples: int = 5000,
    confidence_threshold: float = 0.5,
) -> list[dict]:
    """Use the best ByT5 model to translate monolingual Akkadian texts."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    if not os.path.exists(model_path):
        print(f"  Model not found: {model_path}")
        return []
    if not os.path.exists(mono_csv):
        print(f"  Monolingual data not found: {mono_csv}")
        return []

    print(f"  Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    mono_df = pd.read_csv(mono_csv)
    mono_df = mono_df[mono_df["transliteration"].notna()].head(max_samples)
    texts = mono_df["transliteration"].tolist()
    print(f"  Translating {len(texts)} monolingual texts...")

    pairs = []
    for i in range(0, len(texts), batch_size):
        batch = [t if t else "<gap>" for t in texts[i:i + batch_size]]
        inputs = tokenizer(
            batch, max_length=512, truncation=True, padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_beams=4,
                max_length=512,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

        decoded = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        for j, translation in enumerate(decoded):
            src_idx = i + j
            if translation and len(translation) > 5:
                pairs.append({
                    "transliteration": texts[src_idx],
                    "translation": clean_translation(translation),
                    "source": "pseudo_label",
                })

        if (i // batch_size + 1) % 20 == 0:
            print(f"    Processed {min(i + batch_size, len(texts))}/{len(texts)}")

    print(f"  Generated {len(pairs)} pseudo-labels")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Augment training data")
    parser.add_argument(
        "--base-data", default=os.path.join(DATA_DIR, "aligned_train.csv"),
        help="Base aligned data CSV"
    )
    parser.add_argument(
        "--no-pseudo-label", action="store_true",
        help="Skip pseudo-labeling (slow, requires GPU)"
    )
    parser.add_argument(
        "--no-templates", action="store_true",
        help="Skip template augmentation"
    )
    parser.add_argument(
        "--no-drills", action="store_true",
        help="Skip dictionary drills"
    )
    parser.add_argument(
        "--templates-per-pattern", type=int, default=100,
        help="Number of template instances per pattern"
    )
    parser.add_argument(
        "--pseudo-label-max", type=int, default=3000,
        help="Maximum monolingual texts to pseudo-label"
    )
    parser.add_argument(
        "--max-drills", type=int, default=0,
        help="Cap dictionary drills to this count (0 = auto-cap at 30%% of total)"
    )
    args = parser.parse_args()

    print("Data Augmentation Pipeline")
    print(f"{'='*60}")

    # ── Load base data ─────────────────────────────────────────────────────
    print(f"\nLoading base data: {args.base_data}")
    base_df = pd.read_csv(args.base_data)
    print(f"  Base pairs: {len(base_df)}")

    all_sources = [("base", base_df[["transliteration", "translation"]].copy())]

    # ── Add OCR-extracted pairs if available ────────────────────────────────
    ocr_path = os.path.join(DATA_DIR, "ocr_extracted_pairs.csv")
    if os.path.exists(ocr_path):
        ocr_df = pd.read_csv(ocr_path)
        if len(ocr_df) > 0:
            print(f"  OCR-extracted pairs: {len(ocr_df)}")
            all_sources.append(("ocr", ocr_df[["transliteration", "translation"]].copy()))

    # ── Add dictionary drills ──────────────────────────────────────────────
    if not args.no_drills:
        drills_path = os.path.join(DATA_DIR, "lexicon_drills.csv")
        if os.path.exists(drills_path):
            drills_df = pd.read_csv(drills_path)
            # Auto-cap: keep synthetic (drills+templates) ≤ 30% of total
            n_base = len(base_df)
            est_templates = args.templates_per_pattern * 7  # ~7 patterns
            if args.max_drills > 0:
                drill_cap = args.max_drills
            else:
                # synthetic_budget = total * 0.30, total = n_base / 0.70
                synthetic_budget = int(n_base / 0.70 * 0.30)
                drill_cap = max(0, synthetic_budget - est_templates)
            if len(drills_df) > drill_cap:
                drills_df = drills_df.sample(n=drill_cap, random_state=42)
                print(f"  Dictionary drills: {len(drills_df)} (capped from {drill_cap} to keep synthetic ≤30%)")
            else:
                print(f"  Dictionary drills: {len(drills_df)}")
            all_sources.append(("drills", drills_df[["transliteration", "translation"]].copy()))
        else:
            print("  Dictionary drills not found. Run 10_lexicon_integration.py first.")

    # ── Template augmentation ──────────────────────────────────────────────
    if not args.no_templates:
        print("\nGenerating template-based augmentation...")
        names = load_proper_nouns(DATA_DIR)
        print(f"  Available proper nouns: {len(names)}")
        if names:
            template_pairs = generate_template_pairs(names, args.templates_per_pattern)
            if template_pairs:
                template_df = pd.DataFrame(template_pairs)[["transliteration", "translation"]]
                print(f"  Template pairs generated: {len(template_df)}")
                all_sources.append(("template", template_df))

    # ── Pseudo-labeling ────────────────────────────────────────────────────
    if not args.no_pseudo_label:
        print("\nRunning pseudo-labeling...")
        model_path = os.path.join(PROJECT_DIR, "checkpoints", "byt5-base", "best_model")
        mono_csv = os.path.join(DATA_DIR, "monolingual_akkadian.csv")
        pseudo_pairs = pseudo_label_monolingual(
            model_path, mono_csv,
            max_samples=args.pseudo_label_max,
        )
        if pseudo_pairs:
            pseudo_df = pd.DataFrame(pseudo_pairs)[["transliteration", "translation"]]
            print(f"  Pseudo-label pairs: {len(pseudo_df)}")
            all_sources.append(("pseudo", pseudo_df))

    # ── Combine all sources ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Combining data sources:")
    all_dfs = []
    for name, df in all_sources:
        df = df.copy()
        df["source"] = name
        all_dfs.append(df)
        print(f"  {name}: {len(df)} pairs")

    combined = pd.concat(all_dfs, ignore_index=True)

    # Deduplicate (keep first occurrence, which is base data)
    combined = combined.drop_duplicates(
        subset=["transliteration", "translation"], keep="first"
    ).reset_index(drop=True)

    # Remove empty
    combined = combined[
        (combined["transliteration"].str.len() > 0) &
        (combined["translation"].str.len() > 0)
    ].reset_index(drop=True)

    print(f"\nTotal after dedup: {len(combined)} pairs")
    print(f"Source breakdown:")
    print(combined["source"].value_counts().to_string())

    # ── Split: keep val set CLEAN (only base data) ─────────────────────────
    # Val set should only contain real human-translated pairs for fair evaluation
    base_only = combined[combined["source"] == "base"].reset_index(drop=True)
    augmented = combined[combined["source"] != "base"].reset_index(drop=True)

    np.random.seed(42)
    val_mask = np.random.rand(len(base_only)) < 0.1
    val_df = base_only[val_mask].reset_index(drop=True)
    train_base = base_only[~val_mask].reset_index(drop=True)

    # Training = base train + all augmented data
    train_df = pd.concat([train_base, augmented], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    # ── Save ───────────────────────────────────────────────────────────────
    output_cols = ["transliteration", "translation"]
    combined[output_cols + ["source"]].to_csv(
        os.path.join(DATA_DIR, "augmented_train.csv"), index=False
    )
    train_df[output_cols].to_csv(
        os.path.join(DATA_DIR, "augmented_train_split.csv"), index=False
    )
    val_df[output_cols].to_csv(
        os.path.join(DATA_DIR, "augmented_val_split.csv"), index=False
    )

    print(f"\nSaved:")
    print(f"  augmented_train.csv: {len(combined)} pairs (all sources)")
    print(f"  augmented_train_split.csv: {len(train_df)} pairs (train)")
    print(f"  augmented_val_split.csv: {len(val_df)} pairs (val, clean only)")


if __name__ == "__main__":
    main()
