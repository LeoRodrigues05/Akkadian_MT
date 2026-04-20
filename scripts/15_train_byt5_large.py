"""
STEP 15 — ByT5-large Two-Stage Training (CPT → SFT)
Stage 1: Continued Pre-Training (CPT) on ALL Akkadian monolingual text
Stage 2: Supervised Fine-Tuning (SFT) on expanded parallel data

Key differences from 03_train_byt5.py:
  - Supports byt5-base, byt5-large, and byt5-xl
  - Gradient checkpointing for memory efficiency
  - Two-stage training pipeline
  - Label smoothing
  - Cosine LR schedule with restarts
  - Reads from augmented data by default

Usage:
  # Stage 1: CPT (monolingual Akkadian)
  python 15_train_byt5_large.py --stage cpt --model google/byt5-large

  # Stage 2: SFT (parallel data, starting from CPT checkpoint)
  python 15_train_byt5_large.py --stage sft --model checkpoints/byt5-large-cpt/best_model

  # Single-stage SFT only (skip CPT):
  python 15_train_byt5_large.py --stage sft --model google/byt5-large
"""
import os
import sys
import json
import math
import argparse
import numpy as np
import pandas as pd
import torch
import sacrebleu
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 512


def load_parallel_data(train_csv: str, val_csv: str):
    """Load parallel training data."""
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # Keep only transliteration and translation columns
    train_df = train_df[["transliteration", "translation"]].dropna().reset_index(drop=True)
    val_df = val_df[["transliteration", "translation"]].dropna().reset_index(drop=True)

    print(f"Parallel data — Train: {len(train_df)}, Val: {len(val_df)}")
    return train_df, val_df


def load_monolingual_data():
    """Load monolingual Akkadian texts for CPT."""
    sources = []

    # Monolingual from published_texts (mined in script 09)
    mono_path = os.path.join(DATA_DIR, "monolingual_akkadian.csv")
    if os.path.exists(mono_path):
        mono = pd.read_csv(mono_path)
        if "transliteration" in mono.columns:
            sources.append(mono["transliteration"].dropna())
            print(f"  Monolingual (published_texts): {len(sources[-1])}")

    # Also use source side of training data
    for csv_name in ["aligned_train.csv", "augmented_train.csv"]:
        path = os.path.join(DATA_DIR, csv_name)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "transliteration" in df.columns:
                sources.append(df["transliteration"].dropna())
                print(f"  Transliterations from {csv_name}: {len(sources[-1])}")
            break  # only use one

    if not sources:
        print("  WARNING: No monolingual data found for CPT.")
        return pd.DataFrame({"text": []})

    all_texts = pd.concat(sources, ignore_index=True)
    all_texts = all_texts.drop_duplicates().reset_index(drop=True)
    mono_df = pd.DataFrame({"text": all_texts})
    print(f"  Total monolingual texts: {len(mono_df)}")
    return mono_df


def preprocess_sft(examples, tokenizer):
    """Tokenize for supervised fine-tuning."""
    inputs = examples["transliteration"]
    targets = examples["translation"]

    model_inputs = tokenizer(
        inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding=False,
    )
    labels = tokenizer(
        targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_cpt(examples, tokenizer):
    """Tokenize for continued pre-training (denoising objective).
    Uses span corruption similar to T5 pre-training:
    - Randomly mask 15% of tokens with sentinel tokens
    - Target is the masked spans
    """
    texts = examples["text"]
    model_inputs = {"input_ids": [], "labels": []}

    for text in texts:
        if not isinstance(text, str) or len(text) < 10:
            continue

        # Tokenize the text
        tokens = tokenizer.encode(text, max_length=MAX_INPUT_LENGTH, truncation=True)

        if len(tokens) < 5:
            continue

        # Simple span corruption: mask ~15% of tokens
        n_tokens = len(tokens)
        n_mask = max(1, int(n_tokens * 0.15))

        # Pick random positions to mask
        rng = np.random.RandomState()
        mask_positions = sorted(rng.choice(n_tokens, size=n_mask, replace=False))

        # Build corrupted input and target
        sentinel_start = 258  # ByT5 sentinel tokens start at 258
        input_tokens = []
        target_tokens = []
        sentinel_idx = 0
        prev_was_masked = False

        for i, token in enumerate(tokens):
            if i in mask_positions:
                if not prev_was_masked:
                    input_tokens.append(sentinel_start - sentinel_idx)
                    target_tokens.append(sentinel_start - sentinel_idx)
                    sentinel_idx = min(sentinel_idx + 1, 99)
                target_tokens.append(token)
                prev_was_masked = True
            else:
                input_tokens.append(token)
                prev_was_masked = False

        # Add EOS
        input_tokens.append(tokenizer.eos_token_id)
        target_tokens.append(tokenizer.eos_token_id)

        model_inputs["input_ids"].append(input_tokens)
        model_inputs["labels"].append(target_tokens)

    return model_inputs


def compute_metrics(eval_preds, tokenizer):
    """Compute BLEU, chrF++, and geometric mean."""
    preds, labels = eval_preds

    preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
    preds = np.where(preds >= tokenizer.vocab_size, tokenizer.pad_token_id, preds)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = [p.strip() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]
    decoded_labels = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]

    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
    chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels], word_order=2)

    bleu_score = bleu.score
    chrf_score = chrf.score
    geo_mean = math.sqrt(bleu_score * chrf_score) if bleu_score > 0 and chrf_score > 0 else 0.0

    return {"bleu": bleu_score, "chrf++": chrf_score, "geo_mean": geo_mean}


def run_cpt(model_name: str, output_dir: str, epochs: int, lr: float, batch_size: int):
    """Stage 1: Continued Pre-Training on monolingual Akkadian."""
    print(f"\n{'='*60}")
    print(f"Stage 1: Continued Pre-Training (CPT)")
    print(f"  Model: {model_name}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    mono_df = load_monolingual_data()
    if len(mono_df) == 0:
        print("No monolingual data available. Skipping CPT.")
        return model_name

    dataset = Dataset.from_pandas(mono_df)
    tokenized = dataset.map(
        lambda x: preprocess_cpt(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing CPT data",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100,
    )

    # Determine gradient accumulation for effective batch size of 64
    effective_batch = 64
    grad_accum = max(1, effective_batch // batch_size)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        gradient_checkpointing=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("Starting CPT...")
    trainer.train()

    best_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"CPT model saved to: {best_dir}")

    return best_dir


def run_sft(
    model_name: str,
    output_dir: str,
    train_csv: str,
    val_csv: str,
    epochs: int,
    lr: float,
    batch_size: int,
    label_smoothing: float,
):
    """Stage 2: Supervised Fine-Tuning on parallel data."""
    print(f"\n{'='*60}")
    print(f"Stage 2: Supervised Fine-Tuning (SFT)")
    print(f"  Model: {model_name}")
    print(f"  Output: {output_dir}")
    print(f"  Train: {train_csv}")
    print(f"  Val: {val_csv}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    train_df, val_df = load_parallel_data(train_csv, val_csv)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    print("Tokenizing datasets...")
    train_tokenized = train_dataset.map(
        lambda x: preprocess_sft(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )
    val_tokenized = val_dataset.map(
        lambda x: preprocess_sft(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing val",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100,
    )

    effective_batch = 64
    grad_accum = max(1, effective_batch // batch_size)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        predict_with_generate=False,
        generation_max_length=MAX_TARGET_LENGTH,
        bf16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_torch",
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        label_smoothing_factor=label_smoothing,
        gradient_checkpointing=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("Starting SFT...")
    trainer.train()

    best_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"SFT model saved to: {best_dir}")

    # Final evaluation
    print("\nRunning final evaluation...")
    eval_results = trainer.evaluate()
    for k, v in eval_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    results_file = os.path.join(output_dir, "eval_results.json")
    with open(results_file, "w") as f:
        json.dump(eval_results, f, indent=2)

    return best_dir


def main():
    parser = argparse.ArgumentParser(description="ByT5 Two-Stage Training (CPT → SFT)")
    parser.add_argument(
        "--stage", choices=["cpt", "sft", "both"], default="both",
        help="Training stage: cpt, sft, or both"
    )
    parser.add_argument(
        "--model", default="google/byt5-large",
        help="Base model name or path to CPT checkpoint"
    )
    parser.add_argument(
        "--train-csv",
        default=os.path.join(DATA_DIR, "augmented_train_split.csv"),
        help="Training data CSV (for SFT)"
    )
    parser.add_argument(
        "--val-csv",
        default=os.path.join(DATA_DIR, "augmented_val_split.csv"),
        help="Validation data CSV (for SFT)"
    )
    parser.add_argument("--cpt-epochs", type=int, default=5, help="CPT epochs")
    parser.add_argument("--sft-epochs", type=int, default=15, help="SFT epochs")
    parser.add_argument("--cpt-lr", type=float, default=1e-4, help="CPT learning rate")
    parser.add_argument("--sft-lr", type=float, default=3e-4, help="SFT learning rate")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing")
    args = parser.parse_args()

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Determine model size suffix
    model_base = args.model.split("/")[-1] if "/" in args.model else "byt5"
    cpt_dir = os.path.join(PROJECT_DIR, "checkpoints", f"{model_base}-cpt")
    sft_dir = os.path.join(PROJECT_DIR, "checkpoints", f"{model_base}-sft")

    model_path = args.model

    if args.stage in ("cpt", "both"):
        model_path = run_cpt(
            model_name=args.model,
            output_dir=cpt_dir,
            epochs=args.cpt_epochs,
            lr=args.cpt_lr,
            batch_size=args.batch_size,
        )

    if args.stage in ("sft", "both"):
        # Fall back to aligned data if augmented doesn't exist yet
        train_csv = args.train_csv
        val_csv = args.val_csv
        if not os.path.exists(train_csv):
            train_csv = os.path.join(DATA_DIR, "aligned_train_split.csv")
            val_csv = os.path.join(DATA_DIR, "aligned_val_split.csv")
            print(f"Augmented data not found, falling back to: {train_csv}")

        run_sft(
            model_name=model_path,
            output_dir=sft_dir,
            train_csv=train_csv,
            val_csv=val_csv,
            epochs=args.sft_epochs,
            lr=args.sft_lr,
            batch_size=args.batch_size,
            label_smoothing=args.label_smoothing,
        )

    print("\n✓ Training complete.")


if __name__ == "__main__":
    main()
