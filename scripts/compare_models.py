"""
Compare all 4 model checkpoints on 10 test samples,
and compute local BLEU/chrF++ for byt5-large-sft.
"""
import gc
import json
import math
import os

import pandas as pd
import sacrebleu
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from preprocess import clean_transliteration

PROJECT_DIR = "/home/leo.rodrigues/Akkadian_MT"
DATA_DIR    = os.path.join(PROJECT_DIR, "scripts", "data")
CKPT_DIR    = os.path.join(PROJECT_DIR, "checkpoints")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")


# ── Helpers ────────────────────────────────────────────────────────────────
def free_gpu(model=None):
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_byt5(model_path, texts, num_beams=4, batch_size=8, max_length=512):
    print(f"  Loading ByT5 from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True).to(DEVICE)
    model.eval()
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = [t if t else "<gap>" for t in texts[i:i + batch_size]]
        inputs = tokenizer(batch, max_length=max_length, truncation=True,
                           padding=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(**inputs, num_beams=num_beams,
                                     max_length=max_length, early_stopping=True)
        preds.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        print(f"    {min(i+batch_size, len(texts))}/{len(texts)}")
    free_gpu(model)
    del tokenizer
    free_gpu()
    return preds


# ── Load BiLSTM and Transformer classes from training scripts ──────────────
import runpy
bilstm_mod = runpy.run_path(os.path.join(PROJECT_DIR, "scripts", "04_train_bilstm.py"))
transformer_mod = runpy.run_path(os.path.join(PROJECT_DIR, "scripts", "05_train_transformer.py"))

PAD_IDX           = bilstm_mod["PAD_IDX"]
CharVocab         = bilstm_mod["CharVocab"]
Encoder           = bilstm_mod["Encoder"]
Decoder           = bilstm_mod["Decoder"]
Seq2Seq           = bilstm_mod["Seq2Seq"]
TFCharVocab       = transformer_mod["CharVocab"]
TransformerMT     = transformer_mod["TransformerMT"]


def _vocab_from_dict(VocabClass, char2idx):
    v = VocabClass()
    v.char2idx = char2idx
    v.idx2char = {i: c for c, i in char2idx.items()}
    v.size = len(char2idx)
    return v


def generate_bilstm(texts):
    print("  Loading BiLSTM...")
    ckpt = torch.load(os.path.join(CKPT_DIR, "bilstm", "best_model.pt"),
                      map_location=DEVICE, weights_only=False)
    sv = _vocab_from_dict(CharVocab, ckpt["src_vocab"])
    tv = _vocab_from_dict(CharVocab, ckpt["tgt_vocab"])
    enc = Encoder(sv.size, 256, 512, 2, 0.3)
    dec = Decoder(tv.size, 256, 512, 512*2, 2, 0.3)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    preds = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i+32]
        encoded = [torch.tensor(sv.encode(t if t else "<gap>"), dtype=torch.long) for t in batch]
        lens = torch.tensor([len(e) for e in encoded])
        padded = pad_sequence(encoded, batch_first=True, padding_value=PAD_IDX).to(DEVICE)
        with torch.no_grad():
            out = model.translate(padded, lens)
        for row in out.cpu().numpy():
            preds.append(tv.decode(row))
    free_gpu(model)
    return preds


def generate_transformer(texts):
    print("  Loading Transformer...")
    ckpt = torch.load(os.path.join(CKPT_DIR, "transformer", "best_model.pt"),
                      map_location=DEVICE, weights_only=False)
    sv = _vocab_from_dict(TFCharVocab, ckpt["src_vocab"])
    tv = _vocab_from_dict(TFCharVocab, ckpt["tgt_vocab"])
    model = TransformerMT(sv.size, tv.size,
                          d_model=256, n_heads=4, n_layers=4,
                          d_ff=512, dropout=0.1, max_len=300).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    preds = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i+32]
        encoded = [torch.tensor(sv.encode(t if t else "<gap>"), dtype=torch.long) for t in batch]
        padded = pad_sequence(encoded, batch_first=True, padding_value=PAD_IDX).to(DEVICE)
        with torch.no_grad():
            out = model.translate(padded)
        for row in out.cpu().numpy():
            preds.append(tv.decode(row))
    free_gpu(model)
    return preds


# ── Task 1: 10-sentence comparison ────────────────────────────────────────
print("\n" + "="*60)
print("TASK 1: 10-sentence model comparison")
print("="*60)

sample_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv")).head(10).copy()
sample_df["clean_transliteration"] = sample_df["transliteration"].apply(clean_transliteration)
texts = sample_df["clean_transliteration"].tolist()

print("\n[1/4] ByT5-base...")
sample_df["byt5_base"] = generate_byt5(
    os.path.join(CKPT_DIR, "byt5-base", "best_model"), texts)

print("\n[2/4] ByT5-base-sft...")
sample_df["byt5_sft"] = generate_byt5(
    os.path.join(CKPT_DIR, "byt5-base-sft", "best_model"), texts)

print("\n[3/4] BiLSTM...")
sample_df["bilstm"] = generate_bilstm(texts)

print("\n[4/4] Transformer...")
sample_df["transformer"] = generate_transformer(texts)

out_cols = ["id", "line_start", "line_end", "transliteration",
            "clean_transliteration", "byt5_base", "byt5_sft", "bilstm", "transformer"]
comparison_path = os.path.join(PROJECT_DIR, "sample_model_translations_10.csv")
sample_df[out_cols].to_csv(comparison_path, index=False)
print(f"\nSaved: {comparison_path}")


# ── Task 2: ByT5-large-sft BLEU evaluation ────────────────────────────────
print("\n" + "="*60)
print("TASK 2: ByT5-large-sft evaluation on augmented val split")
print("="*60)

val_df = pd.read_csv(os.path.join(DATA_DIR, "augmented_val_split.csv")).dropna(
    subset=["transliteration", "translation"])
val_sources = val_df["transliteration"].tolist()
val_refs    = val_df["translation"].tolist()
print(f"Val samples: {len(val_sources)}")

val_preds = generate_byt5(
    os.path.join(CKPT_DIR, "byt5-large-sft", "best_model"),
    val_sources, num_beams=4, batch_size=16, max_length=512)

bleu = sacrebleu.corpus_bleu(val_preds, [val_refs]).score
chrf = sacrebleu.corpus_chrf(val_preds, [val_refs], word_order=2).score
geo  = math.sqrt(bleu * chrf) if bleu > 0 and chrf > 0 else 0.0

results = {
    "model": "byt5-large-sft",
    "bleu": bleu,
    "chrf++": chrf,
    "geo_mean": geo,
    "val_samples": len(val_sources),
    "model_path": os.path.join(CKPT_DIR, "byt5-large-sft", "best_model"),
    "val_csv": os.path.join(DATA_DIR, "augmented_val_split.csv"),
    "sample_predictions": [
        {"src": val_sources[i], "pred": val_preds[i], "ref": val_refs[i]}
        for i in range(min(5, len(val_preds)))
    ],
}

results_path = os.path.join(CKPT_DIR, "byt5-large-sft", "eval_results-local.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"ByT5-large-sft Local Results")
print(f"  BLEU:     {bleu:.2f}")
print(f"  chrF++:   {chrf:.2f}")
print(f"  Geo Mean: {geo:.2f}")
print(f"  Saved to: {results_path}")
print(f"{'='*60}")
print("\nDone.")
