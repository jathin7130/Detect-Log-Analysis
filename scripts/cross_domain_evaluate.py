"""Cross-domain transfer evaluation: zero-shot ("before adaptation") and
after a short fine-tune on a small target sample ("limited target
adaptation"), per the proposal's generalization-test design.

Takes the SOURCE-trained LSTM (one seed, scoped per the earlier ETA
discussion) and:
  1. Scores it zero-shot on the cross-domain-remapped target test split
     (built by cross_domain_build.py — target content re-parsed through
     source's frozen template vocabulary).
  2. Fine-tunes it briefly (few epochs, low LR) on a small slice of the
     target's own normal sequences in that same remapped id space, then
     scores it again.
Both use the SAME held-out cross-domain test split, so the before/after
numbers are directly comparable.
"""
import argparse
import csv
import functools
import json
import random
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from lstm_model import DeepLogLSTM, build_windows  # noqa: E402
from evaluate import evaluate, pick_threshold  # noqa: E402
from evaluate_sequence_model import score_items  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_CROSS = ROOT / "data" / "cross_domain"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

SEED = 2026
ADAPT_FRACTION = 0.15  # fraction of cross-domain sequences used for limited adaptation


def load_cross_domain(source: str, target: str):
    path = DATA_CROSS / f"{source}_to_{target}.csv"
    items = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append((row["seq_id"], [int(e) for e in row["event_ids"].split()], int(row["label"])))
    return items


def split_adapt_val_test(items):
    rng = random.Random(SEED)
    by_label = {}
    for it in items:
        by_label.setdefault(it[2], []).append(it)
    adapt, val, test = [], [], []
    for label, group in by_label.items():
        group = group[:]
        rng.shuffle(group)
        n = len(group)
        n_adapt = int(n * ADAPT_FRACTION)
        n_val = int(n * 0.15)
        adapt += group[:n_adapt]
        val += group[n_adapt:n_adapt + n_val]
        test += group[n_adapt + n_val:]
    return adapt, val, test


def run(source: str, target: str, seed: int = 42, adapt_epochs: int = 3, adapt_lr: float = 1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = MODELS / f"{source}_lstm_seed{seed}.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    window_size = ckpt["window_size"]

    model = DeepLogLSTM(vocab_size=ckpt["vocab_size"], embed_dim=ckpt["embed_dim"], hidden_dim=ckpt["hidden_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    items = load_cross_domain(source, target)
    adapt_items, val_items, test_items = split_adapt_val_test(items)
    print(f"[{source}->{target}] n_adapt={len(adapt_items)} n_val={len(val_items)} n_test={len(test_items)}")

    y_val = np.array([lbl for _, _, lbl in val_items])
    y_test = np.array([lbl for _, _, lbl in test_items])

    # --- before adaptation (zero-shot) ---
    model.eval()
    scores_val_before, _ = score_items(model, device, val_items, window_size, topk=3)
    scores_test_before, _ = score_items(model, device, test_items, window_size, topk=3)
    threshold_before = pick_threshold(y_val, scores_val_before)
    metrics_before = evaluate(y_test, scores_test_before, threshold_before)
    metrics_before["mean_raw_score"] = float(scores_test_before.mean())

    # --- limited target adaptation: fine-tune on normal adapt sequences ---
    adapt_normal = [(seq, lbl) for _, seq, lbl in adapt_items if lbl == 0]
    adapt_seqs = [seq for seq, _ in adapt_normal]
    x_adapt, y_adapt = build_windows(adapt_seqs, window_size)

    if len(x_adapt) > 0:
        ds = TensorDataset(torch.from_numpy(x_adapt), torch.from_numpy(y_adapt))
        loader = DataLoader(ds, batch_size=512, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=adapt_lr)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for epoch in range(adapt_epochs):
            loss_sum, n = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * xb.size(0)
                n += xb.size(0)
            print(f"[{source}->{target}] adapt epoch {epoch+1}/{adapt_epochs} loss={loss_sum/n:.4f}")
    else:
        print(f"[{source}->{target}] no normal adapt windows available, skipping fine-tune")

    model.eval()
    scores_val_after, _ = score_items(model, device, val_items, window_size, topk=3)
    scores_test_after, _ = score_items(model, device, test_items, window_size, topk=3)
    threshold_after = pick_threshold(y_val, scores_val_after)
    metrics_after = evaluate(y_test, scores_test_after, threshold_after)
    metrics_after["mean_raw_score"] = float(scores_test_after.mean())

    result = {
        "source": source, "target": target, "seed": seed,
        "n_adapt": len(adapt_items), "n_val": len(val_items), "n_test": len(test_items),
        "note": (
            "With 100% vocabulary collapse (see cross_domain_build manifests), every window "
            "is an identical constant input, so the model's prediction is constant across all "
            "sequences. The precision/recall/F1 threshold-selection can then only express "
            "'flag everyone' regardless of whether the model's constant prediction is "
            "'miss' (before) or 'hit' (after adaptation) -- so thresholded P/R/F1 can look "
            "identical even though the model genuinely changed. mean_raw_score (not "
            "threshold-dependent) is the reliable signal: it should swing from ~1.0 (always "
            "'miss'/anomalous) before adaptation toward ~0.0 (always 'hit'/normal) after, "
            "since the model was fine-tuned exclusively on normal target sequences."
        ),
        "before_adaptation": metrics_before,
        "after_adaptation": metrics_after,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"cross_domain_{source}_to_{target}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"[{source}->{target}] before: F1={metrics_before['f1']:.3f} PR-AUC={metrics_before['pr_auc']:.3f} | "
          f"after: F1={metrics_after['f1']:.3f} PR-AUC={metrics_after['pr_auc']:.3f} -> {out_path}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["hdfs", "bgl", "thunderbird"])
    ap.add_argument("--target", required=True, choices=["hdfs", "bgl", "thunderbird"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.source, args.target, seed=args.seed)
