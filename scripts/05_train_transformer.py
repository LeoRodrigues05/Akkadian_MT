"""
STEP 6 — Baseline 3: Vanilla Transformer from scratch
Academic comparison baseline — character-level small Transformer for Akkadian → English MT.
"""
import os
import json
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import sacrebleu

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "checkpoints", "transformer")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────
D_MODEL = 256
D_FF = 512
N_HEADS = 4
N_LAYERS = 4
DROPOUT = 0.1
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-4
WARMUP_STEPS = 2000
MAX_LEN = 300
SEED = 42

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Vocabulary (same class as BiLSTM baseline) ─────────────────────────────
class CharVocab:
    def __init__(self):
        self.char2idx = {"<pad>": PAD_IDX, "<sos>": SOS_IDX,
                         "<eos>": EOS_IDX, "<unk>": UNK_IDX}
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.size = 4

    def build(self, texts):
        chars = set()
        for text in texts:
            chars.update(text)
        for c in sorted(chars):
            if c not in self.char2idx:
                self.char2idx[c] = self.size
                self.idx2char[self.size] = c
                self.size += 1

    def encode(self, text, max_len=MAX_LEN):
        ids = [SOS_IDX]
        for c in text[:max_len - 2]:
            ids.append(self.char2idx.get(c, UNK_IDX))
        ids.append(EOS_IDX)
        return ids

    def decode(self, ids):
        chars = []
        for idx in ids:
            if idx == EOS_IDX:
                break
            if idx not in (PAD_IDX, SOS_IDX):
                chars.append(self.idx2char.get(idx, "?"))
        return "".join(chars)


# ── Dataset ────────────────────────────────────────────────────────────────
class MTDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab):
        self.src = [torch.tensor(src_vocab.encode(t), dtype=torch.long) for t in src_texts]
        self.tgt = [torch.tensor(tgt_vocab.encode(t), dtype=torch.long) for t in tgt_texts]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]


def collate_fn(batch):
    srcs, tgts = zip(*batch)
    srcs_padded = pad_sequence(srcs, batch_first=True, padding_value=PAD_IDX)
    tgts_padded = pad_sequence(tgts, batch_first=True, padding_value=PAD_IDX)
    return srcs_padded, tgts_padded


# ── Positional Encoding ───────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ── Transformer Model ─────────────────────────────────────────────────────
class TransformerMT(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads,
                 n_layers, d_ff, dropout, max_len):
        super().__init__()
        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt):
        # Create masks
        src_pad_mask = (src == PAD_IDX)  # (batch, src_len)
        tgt_pad_mask = (tgt == PAD_IDX)  # (batch, tgt_len)
        tgt_len = tgt.size(1)
        tgt_causal_mask = self.transformer.generate_square_subsequent_mask(
            tgt_len, device=src.device
        )

        # Embeddings + positional encoding
        src_emb = self.pos_enc(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        output = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_causal_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )

        return self.fc_out(output)

    def translate(self, src, max_len=MAX_LEN):
        """Greedy autoregressive decoding."""
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            src_pad_mask = (src == PAD_IDX)
            src_emb = self.pos_enc(self.src_embedding(src) * math.sqrt(self.d_model))
            memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_pad_mask)

            # Start with <sos>
            ys = torch.full((batch_size, 1), SOS_IDX, dtype=torch.long, device=src.device)

            for _ in range(max_len):
                tgt_emb = self.pos_enc(self.tgt_embedding(ys) * math.sqrt(self.d_model))
                tgt_mask = self.transformer.generate_square_subsequent_mask(
                    ys.size(1), device=src.device
                )
                output = self.transformer.decoder(
                    tgt_emb, memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_pad_mask,
                )
                logits = self.fc_out(output[:, -1:])
                next_token = logits.argmax(-1)
                ys = torch.cat([ys, next_token], dim=1)

                if (next_token == EOS_IDX).all():
                    break

            return ys[:, 1:]  # remove <sos>


# ── Warmup + Cosine LR Scheduler ──────────────────────────────────────────
class WarmupScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * self.warmup_steps ** -1.5
        )
        for p in self.optimizer.param_groups:
            p["lr"] = lr


def evaluate_model(model, val_loader, tgt_vocab):
    """Generate translations and compute metrics."""
    model.eval()
    all_preds = []
    all_refs = []

    with torch.no_grad():
        for src, tgt in val_loader:
            src = src.to(DEVICE)
            output_tokens = model.translate(src)
            output_tokens = output_tokens.cpu().numpy()

            for i in range(len(output_tokens)):
                pred = tgt_vocab.decode(output_tokens[i])
                ref = tgt_vocab.decode(tgt[i].numpy())
                all_preds.append(pred)
                all_refs.append(ref)

    bleu = sacrebleu.corpus_bleu(all_preds, [all_refs])
    chrf = sacrebleu.corpus_chrf(all_preds, [all_refs], word_order=2)

    bleu_score = bleu.score
    chrf_score = chrf.score
    geo_mean = math.sqrt(bleu_score * chrf_score) if bleu_score > 0 and chrf_score > 0 else 0.0

    return {
        "bleu": bleu_score,
        "chrf++": chrf_score,
        "geo_mean": geo_mean,
        "sample_preds": all_preds[:5],
        "sample_refs": all_refs[:5],
    }


def main():
    print(f"Device: {DEVICE}")
    print(f"Output dir: {OUTPUT_DIR}")

    # ── Load data ──────────────────────────────────────────────────────────
    train_df = pd.read_csv(os.path.join(DATA_DIR, "aligned_train_split.csv")).dropna()
    val_df = pd.read_csv(os.path.join(DATA_DIR, "aligned_val_split.csv")).dropna()
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    src_texts = train_df["transliteration"].tolist()
    tgt_texts = train_df["translation"].tolist()
    val_src = val_df["transliteration"].tolist()
    val_tgt = val_df["translation"].tolist()

    # ── Build vocabularies ─────────────────────────────────────────────────
    src_vocab = CharVocab()
    tgt_vocab = CharVocab()
    src_vocab.build(src_texts + val_src)
    tgt_vocab.build(tgt_texts + val_tgt)
    print(f"Source vocab size: {src_vocab.size}")
    print(f"Target vocab size: {tgt_vocab.size}")

    # ── Create datasets ────────────────────────────────────────────────────
    train_dataset = MTDataset(src_texts, tgt_texts, src_vocab, tgt_vocab)
    val_dataset = MTDataset(val_src, val_tgt, src_vocab, tgt_vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)

    # ── Build model ────────────────────────────────────────────────────────
    model = TransformerMT(
        src_vocab_size=src_vocab.size,
        tgt_vocab_size=tgt_vocab.size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        max_len=MAX_LEN,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ── Training setup ─────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupScheduler(optimizer, D_MODEL, WARMUP_STEPS)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_val_loss = float("inf")
    best_geo_mean = 0.0
    history = []

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        n_batches = 0

        for src, tgt in train_loader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:, :-1]  # all but last
            tgt_output = tgt[:, 1:]  # all but first

            optimizer.zero_grad()
            logits = model(src, tgt_input)

            loss = criterion(logits.contiguous().view(-1, logits.size(-1)),
                             tgt_output.contiguous().view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Validation loss
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(DEVICE)
                tgt = tgt.to(DEVICE)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                logits = model(src, tgt_input)
                loss = criterion(logits.contiguous().view(-1, logits.size(-1)),
                                 tgt_output.contiguous().view(-1))
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {avg_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

        history.append({"epoch": epoch, "train_loss": avg_loss,
                        "val_loss": avg_val_loss, "lr": current_lr})

        # Evaluate every 5 epochs and save best model by geo_mean
        if epoch % 5 == 0 or epoch == EPOCHS:
            metrics = evaluate_model(model, val_loader, tgt_vocab)
            print(f"  BLEU: {metrics['bleu']:.2f} | chrF++: {metrics['chrf++']:.2f} | "
                  f"Geo Mean: {metrics['geo_mean']:.2f}")
            print(f"  Sample predictions:")
            for p, r in zip(metrics["sample_preds"][:3], metrics["sample_refs"][:3]):
                print(f"    PRED: {p[:80]}")
                print(f"    REF:  {r[:80]}")
                print()

            if metrics["geo_mean"] > best_geo_mean:
                best_geo_mean = metrics["geo_mean"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "src_vocab": src_vocab.char2idx,
                    "tgt_vocab": tgt_vocab.char2idx,
                    "epoch": epoch,
                    "geo_mean": best_geo_mean,
                }, os.path.join(OUTPUT_DIR, "best_model.pt"))
                print(f"  → Saved best model (geo_mean={best_geo_mean:.2f})")

    # ── Final evaluation ───────────────────────────────────────────────────
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, "best_model.pt"), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_metrics = evaluate_model(model, val_loader, tgt_vocab)
    print(f"\n{'='*60}")
    print(f"Final Results (Vanilla Transformer):")
    print(f"  BLEU:     {final_metrics['bleu']:.2f}")
    print(f"  chrF++:   {final_metrics['chrf++']:.2f}")
    print(f"  Geo Mean: {final_metrics['geo_mean']:.2f}")

    # Save results
    results = {
        "model": "Vanilla Transformer",
        "bleu": final_metrics["bleu"],
        "chrf++": final_metrics["chrf++"],
        "geo_mean": final_metrics["geo_mean"],
        "total_params": total_params,
        "best_epoch": checkpoint["epoch"],
        "history": history,
    }
    with open(os.path.join(OUTPUT_DIR, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n✓ Transformer training complete.")


if __name__ == "__main__":
    main()
