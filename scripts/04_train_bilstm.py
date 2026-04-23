"""
STEP 5 — Baseline 2: BiLSTM Seq2Seq with Bahdanau Attention
Academic comparison baseline — character-level Akkadian → English MT.
"""
import argparse
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
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import sacrebleu

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "checkpoints", "bilstm")

# ── Hyperparameters ────────────────────────────────────────────────────────
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
MAX_LEN = 300  # character-level max length
TEACHER_FORCING_RATIO = 0.5
SEED = 42

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Vocabulary ─────────────────────────────────────────────────────────────
class CharVocab:
    """Character-level vocabulary."""

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
    src_lens = torch.tensor([len(s) for s in srcs])
    tgt_lens = torch.tensor([len(t) for t in tgts])
    srcs_padded = pad_sequence(srcs, batch_first=True, padding_value=PAD_IDX)
    tgts_padded = pad_sequence(tgts, batch_first=True, padding_value=PAD_IDX)
    return srcs_padded, tgts_padded, src_lens, tgt_lens


# ── Encoder ────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lens):
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(
            embedded, src_lens.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        outputs, (hidden, cell) = self.rnn(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        # Combine bidirectional hidden states
        # hidden: (num_layers*2, batch, hidden) → (num_layers, batch, hidden)
        hidden = torch.cat([hidden[0::2], hidden[1::2]], dim=2)
        cell = torch.cat([cell[0::2], cell[1::2]], dim=2)
        hidden = torch.tanh(self.fc_h(hidden))
        cell = torch.tanh(self.fc_c(cell))

        return outputs, hidden, cell


# ── Bahdanau Attention ─────────────────────────────────────────────────────
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim, enc_dim):
        super().__init__()
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(enc_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask):
        # decoder_hidden: (batch, hidden)
        # encoder_outputs: (batch, src_len, enc_dim)
        query = self.W_q(decoder_hidden).unsqueeze(1)  # (batch, 1, hidden)
        keys = self.W_k(encoder_outputs)  # (batch, src_len, hidden)
        energy = self.v(torch.tanh(query + keys)).squeeze(2)  # (batch, src_len)

        energy = energy.masked_fill(mask == 0, -1e10)
        attn_weights = torch.softmax(energy, dim=1)

        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch, 1, enc_dim)
        return context.squeeze(1), attn_weights


# ── Decoder ────────────────────────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, enc_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.attention = BahdanauAttention(hidden_dim, enc_dim)
        self.rnn = nn.LSTM(
            embed_dim + enc_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_dim + enc_dim + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs, mask):
        # input_token: (batch,) single token
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))  # (batch, 1, embed)

        context, attn_weights = self.attention(hidden[-1], encoder_outputs, mask)

        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)  # (batch, 1, embed+enc)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        prediction = self.fc_out(
            torch.cat([output.squeeze(1), context, embedded.squeeze(1)], dim=1)
        )
        return prediction, hidden, cell, attn_weights


# ── Seq2Seq Model ──────────────────────────────────────────────────────────
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, src_lens, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size, device=self.device)
        encoder_outputs, hidden, cell = self.encoder(src, src_lens)

        # Create mask for attention
        mask = (src != PAD_IDX).float()

        input_token = tgt[:, 0]  # <sos>

        for t in range(1, tgt_len):
            prediction, hidden, cell, _ = self.decoder(
                input_token, hidden, cell, encoder_outputs, mask
            )
            outputs[:, t] = prediction
            if random.random() < teacher_forcing_ratio:
                input_token = tgt[:, t]
            else:
                input_token = prediction.argmax(1)

        return outputs

    def translate(self, src, src_lens, max_len=MAX_LEN):
        """Greedy decoding for inference."""
        self.eval()
        with torch.no_grad():
            encoder_outputs, hidden, cell = self.encoder(src, src_lens)
            mask = (src != PAD_IDX).float()
            batch_size = src.size(0)

            input_token = torch.full((batch_size,), SOS_IDX, dtype=torch.long, device=self.device)
            all_tokens = []

            for _ in range(max_len):
                prediction, hidden, cell, _ = self.decoder(
                    input_token, hidden, cell, encoder_outputs, mask
                )
                input_token = prediction.argmax(1)
                all_tokens.append(input_token.unsqueeze(1))
                if (input_token == EOS_IDX).all():
                    break

            return torch.cat(all_tokens, dim=1)


def evaluate_model(model, val_loader, tgt_vocab):
    """Generate translations and compute metrics."""
    model.eval()
    all_preds = []
    all_refs = []

    with torch.no_grad():
        for src, tgt, src_lens, tgt_lens in val_loader:
            src = src.to(DEVICE)
            src_lens = src_lens.to(DEVICE)

            output_tokens = model.translate(src, src_lens)
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
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Train BiLSTM Seq2Seq")
    parser.add_argument("--train-csv", default=os.path.join(DATA_DIR, "aligned_train_split.csv"),
                        help="Path to training CSV (default: aligned_train_split.csv)")
    parser.add_argument("--val-csv", default=os.path.join(DATA_DIR, "aligned_val_split.csv"),
                        help="Path to validation CSV (default: aligned_val_split.csv)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Directory for best_model.pt and eval_results.json "
                             "(default: checkpoints/bilstm)")
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Output dir: {OUTPUT_DIR}")

    # ── Load data ──────────────────────────────────────────────────────────
    train_df = pd.read_csv(args.train_csv).dropna()
    val_df = pd.read_csv(args.val_csv).dropna()
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
    enc_dim = HIDDEN_DIM * 2  # bidirectional
    encoder = Encoder(src_vocab.size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    decoder = Decoder(tgt_vocab.size, EMBED_DIM, HIDDEN_DIM, enc_dim, NUM_LAYERS, DROPOUT)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ── Training setup ─────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_val_loss = float("inf")
    best_geo_mean = 0.0
    history = []

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        n_batches = 0

        for src, tgt, src_lens, tgt_lens in train_loader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            src_lens = src_lens.to(DEVICE)

            optimizer.zero_grad()
            output = model(src, tgt, src_lens, TEACHER_FORCING_RATIO)

            # output: (batch, tgt_len, vocab), tgt: (batch, tgt_len)
            output = output[:, 1:].contiguous().view(-1, output.size(-1))
            tgt_flat = tgt[:, 1:].contiguous().view(-1)

            loss = criterion(output, tgt_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Validation loss
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for src, tgt, src_lens, tgt_lens in val_loader:
                src = src.to(DEVICE)
                tgt = tgt.to(DEVICE)
                src_lens = src_lens.to(DEVICE)

                output = model(src, tgt, src_lens, 0)  # no teacher forcing
                output = output[:, 1:].contiguous().view(-1, output.size(-1))
                tgt_flat = tgt[:, 1:].contiguous().view(-1)
                loss = criterion(output, tgt_flat)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        print(f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        history.append({"epoch": epoch, "train_loss": avg_loss, "val_loss": avg_val_loss})

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
    # Load best model
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, "best_model.pt"), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_metrics = evaluate_model(model, val_loader, tgt_vocab)
    print(f"\n{'='*60}")
    print(f"Final Results (BiLSTM Seq2Seq):")
    print(f"  BLEU:     {final_metrics['bleu']:.2f}")
    print(f"  chrF++:   {final_metrics['chrf++']:.2f}")
    print(f"  Geo Mean: {final_metrics['geo_mean']:.2f}")

    # Save results
    results = {
        "model": "BiLSTM Seq2Seq",
        "bleu": final_metrics["bleu"],
        "chrf++": final_metrics["chrf++"],
        "geo_mean": final_metrics["geo_mean"],
        "total_params": total_params,
        "best_epoch": checkpoint["epoch"],
        "history": history,
    }
    with open(os.path.join(OUTPUT_DIR, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n✓ BiLSTM training complete.")


if __name__ == "__main__":
    main()
