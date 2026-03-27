"""
STEP 8b — ByT5 Evaluation: BLEU / chrF++ on validation set
Loads the best ByT5 checkpoint and runs beam-search generation to compute
BLEU, chrF++, and geometric mean on aligned_val_split.csv.
"""
import os
import json
import math
import pandas as pd
import torch
import sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

MODEL_PATH = os.path.join(PROJECT_DIR, "checkpoints", "byt5-base", "best_model")
VAL_CSV = os.path.join(DATA_DIR, "aligned_val_split.csv")

BATCH_SIZE = 16
NUM_BEAMS = 4
MAX_LENGTH = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Val CSV: {VAL_CSV}")

    # ── Load model ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # ── Load validation data ───────────────────────────────────────────────
    val_df = pd.read_csv(VAL_CSV).dropna(subset=["transliteration", "translation"])
    sources = val_df["transliteration"].tolist()
    references = val_df["translation"].tolist()
    print(f"Validation samples: {len(sources)}")

    # ── Generate predictions ───────────────────────────────────────────────
    predictions = []
    for i in range(0, len(sources), BATCH_SIZE):
        batch = sources[i : i + BATCH_SIZE]
        batch = [t if t else "<gap>" for t in batch]

        inputs = tokenizer(
            batch,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_beams=NUM_BEAMS,
                max_length=MAX_LENGTH,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)

        done = min(i + BATCH_SIZE, len(sources))
        if done % 64 == 0 or done == len(sources):
            print(f"  Generated {done}/{len(sources)}")

    # ── Compute metrics ────────────────────────────────────────────────────
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=2)

    bleu_score = bleu.score
    chrf_score = chrf.score
    geo_mean = math.sqrt(bleu_score * chrf_score) if bleu_score > 0 and chrf_score > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"ByT5-base Validation Results")
    print(f"  BLEU:     {bleu_score:.2f}")
    print(f"  chrF++:   {chrf_score:.2f}")
    print(f"  Geo Mean: {geo_mean:.2f}")
    print(f"  Params:   {total_params:,}")
    print(f"{'='*60}")

    # ── Sample predictions ─────────────────────────────────────────────────
    print("\nSample predictions:")
    for j in range(min(5, len(predictions))):
        print(f"  SRC:  {sources[j][:80]}")
        print(f"  PRED: {predictions[j][:80]}")
        print(f"  REF:  {references[j][:80]}")
        print()

    # ── Save results ───────────────────────────────────────────────────────
    results = {
        "model": "ByT5-base",
        "bleu": bleu_score,
        "chrf++": chrf_score,
        "geo_mean": geo_mean,
        "total_params": total_params,
        "num_beams": NUM_BEAMS,
        "val_samples": len(sources),
    }

    results_path = os.path.join(PROJECT_DIR, "checkpoints", "byt5-base", "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
