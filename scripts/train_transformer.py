"""Train the compact transformer next-event predictor.

Mirrors scripts/train_lstm.py exactly in training-loop structure (same data
prep, same mixed-precision/early-stopping hooks, same train-on-normal-only
convention) — only the model class differs. Keeping the loops identical
means the LSTM-vs-transformer comparison in later phases isn't confounded
by incidental training-procedure differences.

--smoke-test caps data/epochs for a fast end-to-end pipeline check rather
than producing a usable model.
"""
import argparse
import csv
import functools
import json
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from lstm_model import build_windows  # noqa: E402
from transformer_model import CompactTransformer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_SEQ = ROOT / "data" / "sequences"
DATA_SPLITS = ROOT / "data" / "splits"
DATA_MASKED = ROOT / "data" / "masked"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def load_normal_sequences(dataset: str, split: str) -> list[list[int]]:
    seqs = {}
    with (DATA_SEQ / f"{dataset}.csv").open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seqs[row["seq_id"]] = [int(e) for e in row["event_ids"].split()]

    normal_ids = []
    with (DATA_SPLITS / f"{dataset}_{split}.csv").open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["label"] == "0":
                normal_ids.append(row["seq_id"])

    return [seqs[sid] for sid in normal_ids if sid in seqs]


def max_event_id(dataset: str) -> int:
    max_id = 0
    with (DATA_MASKED / dataset / "templates.csv").open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            max_id = max(max_id, int(row["event_id"]))
    return max_id


def run(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and not args.no_mixed_precision

    print(f"[{args.dataset}] seed={args.seed} device={device} mixed_precision={use_amp} "
          f"window_size={args.window_size} smoke_test={args.smoke_test}")

    train_seqs = load_normal_sequences(args.dataset, "train")
    val_seqs = load_normal_sequences(args.dataset, "val")
    vocab_size = max_event_id(args.dataset)

    x_train, y_train = build_windows(train_seqs, args.window_size)
    x_val, y_val = build_windows(val_seqs, args.window_size)

    if args.smoke_test:
        x_train, y_train = x_train[:3000], y_train[:3000]
        x_val, y_val = x_val[:1000], y_val[:1000]
        args.epochs = min(args.epochs, 3)
        args.patience = min(args.patience, 1)

    print(f"[{args.dataset}] train_windows={len(x_train):,} val_windows={len(x_val):,} vocab_size={vocab_size}")

    if len(x_train) == 0 or len(x_val) == 0:
        raise RuntimeError(
            f"[{args.dataset}] no training/val windows produced (window_size={args.window_size} "
            f"may be larger than most sequences)."
        )

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=(device.type == "cuda"))

    model = CompactTransformer(
        vocab_size=vocab_size,
        window_size=args.window_size,
        embed_dim=args.embed_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item() * xb.size(0)
            train_n += xb.size(0)
        train_loss = train_loss_sum / train_n

        model.eval()
        val_loss_sum, val_n, val_correct = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                val_loss_sum += loss.item() * xb.size(0)
                val_n += xb.size(0)
                val_correct += (logits.argmax(dim=1) == yb).sum().item()
        val_loss = val_loss_sum / val_n
        val_acc = val_correct / val_n

        elapsed = time.time() - t0
        print(f"[{args.dataset}] epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({elapsed:.1f}s elapsed)")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})

        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[{args.dataset}] early stopping at epoch {epoch} "
                      f"(no val_loss improvement for {args.patience} epochs)")
                break

    MODELS.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke_test else f"_seed{args.seed}"
    ckpt_path = MODELS / f"{args.dataset}_transformer{suffix}.pt"
    torch.save(
        {
            "model_state": best_state if best_state is not None else model.state_dict(),
            "vocab_size": vocab_size,
            "window_size": args.window_size,
            "embed_dim": args.embed_dim,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_feedforward,
        },
        ckpt_path,
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    hist_path = RESULTS / f"{args.dataset}_transformer{suffix}_history.json"
    hist_path.write_text(json.dumps({
        "dataset": args.dataset, "seed": args.seed, "device": str(device), "mixed_precision": use_amp,
        "smoke_test": args.smoke_test, "best_val_loss": best_val_loss, "history": history,
    }, indent=2), encoding="utf-8")

    print(f"[{args.dataset}] done. best_val_loss={best_val_loss:.4f} -> {ckpt_path}, {hist_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hdfs", "bgl", "thunderbird"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window-size", type=int, default=5)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dim-feedforward", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--min-delta", type=float, default=5e-4)
    ap.add_argument("--no-mixed-precision", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()
    run(args)
