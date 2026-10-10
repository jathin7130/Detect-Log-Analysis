"""Score a trained LSTM/transformer checkpoint as an anomaly detector.

DeepLog-style scoring: for each window in a sequence, check whether the
TRUE next event is among the model's top-k predicted candidates. A
sequence's anomaly score is its "miss rate" — the fraction of its windows
where the true next event fell outside top-k. Sequences too short to form
even one window get score 0.0 (treated as normal by default; counted and
reported separately so they don't silently bias the metrics).

Threshold is picked on the val split (all labels, maximizing F1) and
applied unchanged to test — same convention as the classical baselines in
Phase 2, so results are directly comparable in results/baselines_locked.*.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from lstm_model import DeepLogLSTM  # noqa: E402
from lstm_combined_model import CombinedLSTM  # noqa: E402
from transformer_model import CompactTransformer  # noqa: E402
from evaluate import evaluate, pick_threshold  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_SEQ = ROOT / "data" / "sequences"
DATA_SPLITS = ROOT / "data" / "splits"
DATA_FEATURES = ROOT / "data" / "features"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def load_split_all(dataset: str, split: str, seq_path: Path = None):
    seq_path = seq_path or (DATA_SEQ / f"{dataset}.csv")
    seqs = {}
    with seq_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seqs[row["seq_id"]] = [int(e) for e in row["event_ids"].split()]

    items = []  # (seq_id, event_ids, label)
    with (DATA_SPLITS / f"{dataset}_{split}.csv").open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["seq_id"]
            if sid in seqs:
                items.append((sid, seqs[sid], int(row["label"])))
    return items


def load_model(model_type: str, ckpt_path: Path, device: torch.device, dataset: str = None):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if model_type == "lstm":
        model = DeepLogLSTM(
            vocab_size=ckpt["vocab_size"], embed_dim=ckpt["embed_dim"], hidden_dim=ckpt["hidden_dim"]
        )
    elif model_type == "lstm_combined":
        semantic_embeddings = np.load(DATA_FEATURES / dataset / "template_embeddings.npy")
        model = CombinedLSTM(
            vocab_size=ckpt["vocab_size"], semantic_embeddings=semantic_embeddings,
            embed_dim=ckpt["embed_dim"], hidden_dim=ckpt["hidden_dim"],
        )
    else:
        model = CompactTransformer(
            vocab_size=ckpt["vocab_size"], window_size=ckpt["window_size"], embed_dim=ckpt["embed_dim"],
            nhead=ckpt["nhead"], num_layers=ckpt["num_layers"], dim_feedforward=ckpt["dim_feedforward"],
        )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["window_size"], ckpt["vocab_size"]


@torch.no_grad()
def score_items(model, device, items, window_size, topk, batch_size=4096):
    """Returns (scores: np.ndarray aligned with items, n_too_short: int)."""
    # Flatten all (window, next, item_idx) triples across all sequences so
    # we can batch the forward pass, then aggregate back per sequence.
    xs, ys, owners = [], [], []
    n_too_short = 0
    for idx, (sid, seq, label) in enumerate(items):
        if len(seq) <= window_size:
            n_too_short += 1
            continue
        arr = np.asarray(seq, dtype=np.int64)
        n_windows = len(seq) - window_size
        windows = np.lib.stride_tricks.sliding_window_view(arr, window_size)[:n_windows]
        xs.append(windows)
        ys.append(arr[window_size:window_size + n_windows])
        owners.extend([idx] * n_windows)

    scores = np.zeros(len(items), dtype=np.float64)
    if not xs:
        return scores, n_too_short

    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    owners = np.asarray(owners, dtype=np.int64)

    miss_counts = np.zeros(len(items), dtype=np.int64)
    window_counts = np.zeros(len(items), dtype=np.int64)
    np.add.at(window_counts, owners, 1)

    for start in range(0, len(x_all), batch_size):
        xb = torch.from_numpy(x_all[start:start + batch_size]).to(device)
        yb = y_all[start:start + batch_size]
        logits = model(xb)
        topk_ids = torch.topk(logits, k=min(topk, logits.size(1)), dim=1).indices.cpu().numpy()
        hit = (topk_ids == yb[:, None]).any(axis=1)
        miss = ~hit
        batch_owners = owners[start:start + batch_size]
        np.add.at(miss_counts, batch_owners[miss], 1)

    nonzero = window_counts > 0
    scores[nonzero] = miss_counts[nonzero] / window_counts[nonzero]
    return scores, n_too_short


def run(model_type: str, dataset: str, seed: int, topk: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suffix = f"_seed{seed}"
    ckpt_path = MODELS / f"{dataset}_{model_type}{suffix}.pt"
    model, window_size, vocab_size = load_model(model_type, ckpt_path, device, dataset=dataset)

    val_items = load_split_all(dataset, "val")
    test_items = load_split_all(dataset, "test")

    y_val = np.array([label for _, _, label in val_items])
    y_test = np.array([label for _, _, label in test_items])

    scores_val, n_short_val = score_items(model, device, val_items, window_size, topk)
    scores_test, n_short_test = score_items(model, device, test_items, window_size, topk)

    threshold = pick_threshold(y_val, scores_val)
    metrics = evaluate(y_test, scores_test, threshold)
    metrics.update({
        "dataset": dataset, "model": model_type, "seed": seed, "topk": topk,
        "window_size": window_size, "vocab_size": vocab_size,
        "n_too_short_val": n_short_val, "n_too_short_test": n_short_test,
    })

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"{dataset}_{model_type}{suffix}_eval.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[{dataset}] {model_type} seed={seed}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
          f"F1={metrics['f1']:.3f} PR-AUC={metrics['pr_auc']:.3f} "
          f"FA/10k={metrics['false_alarms_per_10k_events']:.1f} -> {out_path}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lstm", "lstm_combined", "transformer"])
    ap.add_argument("--dataset", required=True, choices=["hdfs", "bgl", "thunderbird"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()
    run(args.model, args.dataset, args.seed, args.topk)
