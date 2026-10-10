"""Measure how much benign drift (scripts/inject_drift.py) degrades a
trained detector's false-positive rate.

Uses the SAME val-selected threshold as the main evaluation (not re-tuned
on drifted data) — the question is "does the operating point we already
committed to start misfiring on drift", not "could we find a threshold
that tolerates this drift".
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evaluate import pick_threshold  # noqa: E402
from evaluate_sequence_model import load_model, load_split_all, score_items  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DRIFT = ROOT / "data" / "drift"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def load_drifted(dataset: str, vocab_size: int):
    """inject_drift.py deliberately assigns ids above vocab_max for the
    'new template' / renamed-token drift mechanisms -- that's the point,
    they're meant to be genuinely novel. But the model's embedding table
    is sized to the ORIGINAL vocab, so any id outside [0, vocab_size] has
    to be clipped to the 'unseen' index (0) before scoring, or indexing
    into the embedding table crashes with an out-of-bounds CUDA assert.
    """
    path = DATA_DRIFT / f"{dataset}_drifted.csv"
    items = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events = [int(e) if int(e) <= vocab_size else 0 for e in row["event_ids"].split()]
            items.append((row["seq_id"], events, int(row["label"])))
    return items


def run(model_type: str, dataset: str, seed: int, topk: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = MODELS / f"{dataset}_{model_type}_seed{seed}.pt"
    model, window_size, vocab_size = load_model(model_type, ckpt_path, device)

    val_items = load_split_all(dataset, "val")
    y_val = np.array([label for _, _, label in val_items])
    scores_val, _ = score_items(model, device, val_items, window_size, topk)
    threshold = pick_threshold(y_val, scores_val)

    test_items = load_split_all(dataset, "test")
    clean_normal = [it for it in test_items if it[2] == 0]
    scores_clean, n_short_clean = score_items(model, device, clean_normal, window_size, topk)
    fpr_clean = float((scores_clean >= threshold).mean()) if len(scores_clean) else float("nan")

    drifted_items = load_drifted(dataset, vocab_size)
    scores_drift, n_short_drift = score_items(model, device, drifted_items, window_size, topk)
    fpr_drift = float((scores_drift >= threshold).mean()) if len(scores_drift) else float("nan")

    result = {
        "dataset": dataset, "model": model_type, "seed": seed, "topk": topk, "threshold": threshold,
        "n_clean_normal": len(clean_normal), "n_drifted": len(drifted_items),
        "n_too_short_clean": n_short_clean, "n_too_short_drift": n_short_drift,
        "false_positive_rate_clean": fpr_clean,
        "false_positive_rate_drifted": fpr_drift,
        "fpr_increase": fpr_drift - fpr_clean,
        "fpr_increase_relative": (fpr_drift / fpr_clean - 1) if fpr_clean > 0 else float("inf"),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"{dataset}_{model_type}_seed{seed}_drift_eval.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"[{dataset}] {model_type} seed={seed}: FPR clean={fpr_clean:.4f} drifted={fpr_drift:.4f} "
          f"(+{result['fpr_increase']:.4f}) -> {out_path}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lstm", "transformer"])
    ap.add_argument("--dataset", required=True, choices=["hdfs", "bgl", "thunderbird"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()
    run(args.model, args.dataset, args.seed, args.topk)
