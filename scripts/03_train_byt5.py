"""
STEP 4 — Baseline 1: ByT5-base fine-tuning
Primary baseline using google/byt5-base for Akkadian → English MT.
"""
import os
import json
import math
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
)

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "checkpoints", "byt5-base")

MODEL_NAME = "google/byt5-base"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 512


def load_data():
    """Load aligned train and validation splits."""
    train_df = pd.read_csv(os.path.join(DATA_DIR, "aligned_train_split.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "aligned_val_split.csv"))

    # Drop rows with NaN
    train_df = train_df.dropna(subset=["transliteration", "translation"]).reset_index(drop=True)
    val_df = val_df.dropna(subset=["transliteration", "translation"]).reset_index(drop=True)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    return train_df, val_df


def preprocess_function(examples, tokenizer):
    """Tokenize inputs and targets for ByT5."""
    inputs = examples["transliteration"]
    targets = examples["translation"]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False,
    )

    labels = tokenizer(
        targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_preds, tokenizer):
    """Compute BLEU and chrF++ metrics."""
    preds, labels = eval_preds

    # Replace -100 and out-of-range IDs in preds and labels
    preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
    preds = np.where(preds >= tokenizer.vocab_size, tokenizer.pad_token_id, preds)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Strip whitespace
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    # Compute BLEU
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])

    # Compute chrF++
    chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels], word_order=2)

    # Geometric mean
    bleu_score = bleu.score
    chrf_score = chrf.score
    if bleu_score > 0 and chrf_score > 0:
        geo_mean = math.sqrt(bleu_score * chrf_score)
    else:
        geo_mean = 0.0

    return {
        "bleu": bleu_score,
        "chrf++": chrf_score,
        "geo_mean": geo_mean,
    }


def main():
    print(f"Loading model: {MODEL_NAME}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load data ──────────────────────────────────────────────────────────
    train_df, val_df = load_data()
    train_dataset = Dataset.from_pandas(train_df[["transliteration", "translation"]])
    val_dataset = Dataset.from_pandas(val_df[["transliteration", "translation"]])

    # ── Load tokenizer and model ───────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # ── Tokenize datasets ──────────────────────────────────────────────────
    print("Tokenizing datasets...")
    train_tokenized = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )
    val_tokenized = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing val",
    )

    # ── Data collator ──────────────────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    # ── Training arguments ─────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=15,
        learning_rate=3e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=16,
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
    )

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    print("Starting training...")
    train_result = trainer.train()

    # ── Save best model ────────────────────────────────────────────────────
    best_dir = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"Best model saved to: {best_dir}")

    # ── Final evaluation ───────────────────────────────────────────────────
    print("\nRunning final evaluation on validation set...")
    eval_results = trainer.evaluate()
    print(f"\nValidation Results:")
    for k, v in eval_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ── Save results ───────────────────────────────────────────────────────
    results_file = os.path.join(OUTPUT_DIR, "eval_results.json")
    with open(results_file, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Results saved to: {results_file}")

    # ── Training stats ─────────────────────────────────────────────────────
    metrics = train_result.metrics
    print(f"\nTraining metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n✓ ByT5-base training complete.")


if __name__ == "__main__":
    main()
