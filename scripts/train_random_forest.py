"""Train a Random Forest classical baseline on TF-IDF features.

Supervised: trained directly on train-split labels. class_weight="balanced"
compensates for the anomaly-rate imbalance (2-40% anomalous depending on
dataset) instead of silently biasing the forest toward the majority class.
Threshold on predicted probability is picked on the validation set
(maximizing F1), then applied as-is to test.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evaluate import evaluate, pick_threshold  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_FEATURES = ROOT / "data" / "features"
RESULTS = ROOT / "results"

SEED = 42


def load_split(dataset: str, split: str, feat: str):
    X = sp.load_npz(DATA_FEATURES / dataset / f"{feat}_{split}.npz")
    y = np.load(DATA_FEATURES / dataset / f"labels_{split}.npy")
    return X, y


def run(dataset: str, feat: str = "tfidf"):
    X_train, y_train = load_split(dataset, "train", feat)
    X_val, y_val = load_split(dataset, "val", feat)
    X_test, y_test = load_split(dataset, "test", feat)

    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    scores_val = model.predict_proba(X_val)[:, 1]
    scores_test = model.predict_proba(X_test)[:, 1]

    threshold = pick_threshold(y_val, scores_val)
    metrics = evaluate(y_test, scores_test, threshold)
    metrics["train_time_sec"] = train_time
    metrics["dataset"] = dataset
    metrics["model"] = "random_forest"
    metrics["features"] = feat

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"{dataset}_random_forest.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(
        f"[{dataset}] RandomForest: P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
        f"F1={metrics['f1']:.3f} PR-AUC={metrics['pr_auc']:.3f} "
        f"FA/10k={metrics['false_alarms_per_10k_events']:.1f} train_time={train_time:.1f}s -> {out_path}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hdfs", "bgl", "thunderbird"])
    ap.add_argument("--features", default="tfidf", choices=["tfidf", "counts"])
    args = ap.parse_args()
    run(args.dataset, args.features)
