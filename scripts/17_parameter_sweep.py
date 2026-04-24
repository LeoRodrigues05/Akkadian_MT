"""
STEP 17 — Inference-side parameter sweep on the validation set.

Sweeps:
  (a) Beam width  k in {1, 2, 4, 8, 16}                         -- beam search
  (b) MBR candidate count K in {4, 8, 16, 32, 64}               -- sampling + MBR

For each configuration we score the generated translations against the
validation references with sacreBLEU (BLEU + chrF++ + geo_mean), and
write all results to a single JSON file for use in the supplemental
material of the final report.

Default model: checkpoints/byt5-base-sft/best_model
Default val :  scripts/data/augmented_val_split.csv
Default out :  checkpoints/byt5-base-sft/parameter_sweep.json

Usage (single config, quick smoke test):
    python 17_parameter_sweep.py --beams 4 --mbr-ks ""

Usage (full sweep):
    python 17_parameter_sweep.py
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

DEFAULT_MODEL_PATH = os.path.join(
    PROJECT_DIR, "checkpoints", "byt5-base-sft", "best_model"
)
DEFAULT_VAL_CSV = os.path.join(DATA_DIR, "augmented_val_split.csv")
DEFAULT_OUT = os.path.join(
    PROJECT_DIR, "checkpoints", "byt5-base-sft", "parameter_sweep.json"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Metric helpers ─────────────────────────────────────────────────────────
def corpus_metrics(predictions, references):
    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=2).score
    geo = math.sqrt(bleu * chrf) if bleu > 0 and chrf > 0 else 0.0
    return {"bleu": bleu, "chrf++": chrf, "geo_mean": geo}


def sentence_geo(hyp, ref):
    if not hyp.strip() or not ref.strip():
        return 0.0
    bleu = sacrebleu.sentence_bleu(hyp, [ref]).score
    chrf = sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score
    return math.sqrt(bleu * chrf) if bleu > 0 and chrf > 0 else 0.0


def mbr_select(candidates):
    """Select the candidate with the highest mean sentence-level geo_mean
    against the other candidates (utility = identity-free MBR)."""
    unique = list(dict.fromkeys(c.strip() for c in candidates if c is not None))
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    best, best_score = unique[0], -1.0
    for i, hyp in enumerate(unique):
        others = [c for j, c in enumerate(unique) if j != i]
        score = float(np.mean([sentence_geo(hyp, o) for o in others]))
        if score > best_score:
            best, best_score = hyp, score
    return best


# ── Generation helpers ─────────────────────────────────────────────────────
def beam_generate(model, tokenizer, sources, num_beams, batch_size, max_length):
    preds = []
    for i in range(0, len(sources), batch_size):
        batch = [t if t else "<gap>" for t in sources[i : i + batch_size]]
        inputs = tokenizer(
            batch, max_length=max_length, truncation=True,
            padding=True, return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                num_beams=num_beams,
                max_length=max_length,
                early_stopping=(num_beams > 1),
            )
        preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    return preds


def sample_candidates(model, tokenizer, sources, n_samples, batch_size, max_length):
    """For each source, return a list of n_samples sampled translations."""
    cands = [[] for _ in sources]
    for i in range(0, len(sources), batch_size):
        batch_idx = list(range(i, min(i + batch_size, len(sources))))
        batch = [sources[j] if sources[j] else "<gap>" for j in batch_idx]
        inputs = tokenizer(
            batch, max_length=max_length, truncation=True,
            padding=True, return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            for _ in range(n_samples):
                out = model.generate(
                    **inputs,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    max_length=max_length,
                    num_return_sequences=1,
                )
                decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
                for k, d in enumerate(decoded):
                    cands[batch_idx[k]].append(d.strip())
    return cands


# ── Main ───────────────────────────────────────────────────────────────────
def parse_int_list(s):
    s = (s or "").strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(description="Beam + MBR parameter sweep")
    p.add_argument("--model", default=DEFAULT_MODEL_PATH)
    p.add_argument("--val-csv", default=DEFAULT_VAL_CSV)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--beams", default="1,2,4,8,16",
                   help="Comma-separated beam widths (empty = skip beam sweep)")
    p.add_argument("--mbr-ks", default="4,8,16,32,64",
                   help="Comma-separated MBR candidate counts (empty = skip MBR sweep)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, evaluate only the first N validation rows (smoke test)")
    args = p.parse_args()

    print(f"Device:    {DEVICE}")
    print(f"Model:     {args.model}")
    print(f"Val CSV:   {args.val_csv}")
    print(f"Beam set:  {args.beams or '(skipped)'}")
    print(f"MBR K set: {args.mbr_ks or '(skipped)'}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(DEVICE)
    model.eval()

    val_df = pd.read_csv(args.val_csv).dropna(
        subset=["transliteration", "translation"]
    )
    if args.limit > 0:
        val_df = val_df.head(args.limit)
    sources = val_df["transliteration"].tolist()
    references = val_df["translation"].tolist()
    print(f"Validation samples: {len(sources)}")

    results = {
        "model_path": args.model,
        "val_csv": args.val_csv,
        "val_samples": len(sources),
        "beam_sweep": [],
        "mbr_sweep": [],
    }

    # ── Beam sweep ─────────────────────────────────────────────────────────
    for k in parse_int_list(args.beams):
        print(f"\n[beam] k={k}")
        t0 = time.time()
        preds = beam_generate(
            model, tokenizer, sources,
            num_beams=k, batch_size=args.batch_size, max_length=args.max_length,
        )
        dt = time.time() - t0
        m = corpus_metrics(preds, references)
        m.update({"num_beams": k, "seconds": dt})
        print(f"  BLEU={m['bleu']:.2f}  chrF++={m['chrf++']:.2f}  "
              f"geo={m['geo_mean']:.2f}  ({dt:.1f}s)")
        results["beam_sweep"].append(m)
        # save incrementally
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # ── MBR sweep ──────────────────────────────────────────────────────────
    mbr_ks = parse_int_list(args.mbr_ks)
    if mbr_ks:
        max_K = max(mbr_ks)
        print(f"\n[mbr] sampling {max_K} candidates per source (reused for all K)")
        t0 = time.time()
        all_cands = sample_candidates(
            model, tokenizer, sources,
            n_samples=max_K,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        sample_dt = time.time() - t0
        print(f"  sampling done in {sample_dt:.1f}s")

        for K in mbr_ks:
            t1 = time.time()
            preds = [mbr_select(c[:K]) for c in all_cands]
            dt = time.time() - t1
            m = corpus_metrics(preds, references)
            m.update({"mbr_K": K, "select_seconds": dt})
            print(f"  K={K:>3}  BLEU={m['bleu']:.2f}  "
                  f"chrF++={m['chrf++']:.2f}  geo={m['geo_mean']:.2f}  "
                  f"(select {dt:.1f}s)")
            results["mbr_sweep"].append(m)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)

        results["mbr_sample_seconds"] = sample_dt

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to: {args.out}")


if __name__ == "__main__":
    main()
