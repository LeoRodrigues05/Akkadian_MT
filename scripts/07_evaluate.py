"""
STEP 8 — Evaluation & Comparison
Loads results from all baselines and produces a comparison table.
"""
import os
import json
import math

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
CHECKPOINTS_DIR = os.path.join(PROJECT_DIR, "checkpoints")

BASELINES = [
    ("ByT5-base", os.path.join(CHECKPOINTS_DIR, "byt5-base", "eval_results.json")),
    ("BiLSTM Seq2Seq", os.path.join(CHECKPOINTS_DIR, "bilstm", "eval_results.json")),
    ("Vanilla Transformer", os.path.join(CHECKPOINTS_DIR, "transformer", "eval_results.json")),
]


def load_results(filepath):
    """Load eval results JSON, returning None if not found."""
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("AKKADIAN → ENGLISH MT — BASELINE COMPARISON")
    print("=" * 70)
    print(f"{'Model':<25} {'BLEU':>8} {'chrF++':>8} {'Geo Mean':>10} {'Params':>12}")
    print("-" * 70)

    results_found = 0
    all_results = {}

    for name, path in BASELINES:
        data = load_results(path)
        if data is None:
            print(f"{name:<25} {'—':>8} {'—':>8} {'—':>10} {'(not run)':>12}")
            continue

        results_found += 1
        bleu = data.get("bleu", data.get("eval_bleu", 0))
        chrf = data.get("chrf++", data.get("eval_chrf++", 0))

        # Handle ByT5 results format (HF Trainer saves with eval_ prefix)
        if "eval_bleu" in data:
            bleu = data["eval_bleu"]
        if "eval_chrf++" in data:
            chrf = data["eval_chrf++"]

        geo_mean = data.get("geo_mean", data.get("eval_geo_mean", 0))
        if geo_mean == 0 and bleu > 0 and chrf > 0:
            geo_mean = math.sqrt(bleu * chrf)

        params = data.get("total_params", "—")
        if isinstance(params, int):
            if params >= 1e9:
                params_str = f"{params / 1e9:.1f}B"
            elif params >= 1e6:
                params_str = f"{params / 1e6:.1f}M"
            else:
                params_str = f"{params / 1e3:.0f}K"
        else:
            params_str = str(params)

        print(f"{name:<25} {bleu:>8.2f} {chrf:>8.2f} {geo_mean:>10.2f} {params_str:>12}")

        all_results[name] = {
            "bleu": bleu, "chrf++": chrf, "geo_mean": geo_mean,
            "params": params
        }

    print("-" * 70)

    if results_found == 0:
        print("\nNo results found. Run the training scripts first:")
        print("  python 03_train_byt5.py")
        print("  python 04_train_bilstm.py")
        print("  python 05_train_transformer.py")
        return

    # ── Best model ─────────────────────────────────────────────────────────
    if all_results:
        best = max(all_results.items(), key=lambda x: x[1]["geo_mean"])
        print(f"\nBest model: {best[0]} (Geo Mean: {best[1]['geo_mean']:.2f})")

    # ── Save combined results ──────────────────────────────────────────────
    output_path = os.path.join(PROJECT_DIR, "results_comparison.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n✓ Evaluation complete.")


if __name__ == "__main__":
    main()
