"""Closes a gap between what Section 3.3 of the proposal promised ("paired
bootstrap intervals and non-parametric paired tests to compare models on
the same incident sets") and what Phase 3 actually delivered (independent
per-seed mean +/- 95% CI, which doesn't test whether two specific models
differ on the SAME test instances).

For each dataset, scores every model on the identical frozen test split
(same seq_id order), then runs a paired bootstrap: resample test indices
with replacement B times, recompute both models' F1 (at their already-
fixed, independently-chosen thresholds) on each resample, and report the
95% CI of the F1 difference plus a two-sided bootstrap p-value (fraction of
resamples where the sign of the difference flips, doubled). Classical
baselines (Isolation Forest, Random Forest) are re-fit in-memory with the
same seed used originally (deterministic -- these weren't persisted to
disk, only their metrics were) specifically to get per-instance scores.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.ensemble import IsolationForest, RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evaluate import evaluate, pick_threshold  # noqa: E402
from evaluate_sequence_model import load_model, load_split_all, score_items  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_FEATURES = ROOT / "data" / "features"
RESULTS = ROOT / "results"

SEED = 42
N_BOOTSTRAP = 2000


def classical_scores(dataset: str, model_type: str):
    """Re-fit the classical baseline (deterministic, same seed as the
    original Phase 2 run) and return (y_val, scores_val, y_test, scores_test)
    aligned with the frozen split order."""
    def load(split, feat="tfidf"):
        X = sp.load_npz(DATA_FEATURES / dataset / f"{feat}_{split}.npz")
        y = np.load(DATA_FEATURES / dataset / f"labels_{split}.npy")
        return X, y

    X_train, y_train = load("train")
    X_val, y_val = load("val")
    X_test, y_test = load("test")

    if model_type == "isolation_forest":
        contamination = max(min(y_train.mean(), 0.5), 1e-4)
        model = IsolationForest(n_estimators=100, contamination=contamination, random_state=SEED, n_jobs=-1)
        model.fit(X_train.tocsc())
        scores_val = -model.score_samples(X_val.tocsc())
        scores_test = -model.score_samples(X_test.tocsc())
    else:
        model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1)
        model.fit(X_train, y_train)
        scores_val = model.predict_proba(X_val)[:, 1]
        scores_test = model.predict_proba(X_test)[:, 1]

    return y_val, scores_val, y_test, scores_test


def neural_scores(dataset: str, model_type: str, seed: int = SEED, topk: int = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ROOT / "models" / f"{dataset}_{model_type}_seed{seed}.pt"
    model, window_size, vocab_size = load_model(model_type, ckpt_path, device, dataset=dataset)

    val_items = load_split_all(dataset, "val")
    test_items = load_split_all(dataset, "test")
    y_val = np.array([lbl for _, _, lbl in val_items])
    y_test = np.array([lbl for _, _, lbl in test_items])
    scores_val, _ = score_items(model, device, val_items, window_size, topk)
    scores_test, _ = score_items(model, device, test_items, window_size, topk)

    # free GPU memory immediately -- we score one model at a time, not all at once
    del model
    torch.cuda.empty_cache()
    return y_val, scores_val, y_test, scores_test, [sid for sid, _, _ in test_items]


def paired_bootstrap_f1(y_test, scores_a, thr_a, scores_b, thr_b, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(y_test)
    diffs = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        f1_a = evaluate(y_test[idx], scores_a[idx], thr_a)["f1"]
        f1_b = evaluate(y_test[idx], scores_b[idx], thr_b)["f1"]
        diffs[i] = f1_a - f1_b

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    # two-sided bootstrap p-value: how often the resampled difference crosses zero
    p_value = 2 * min((diffs >= 0).mean(), (diffs <= 0).mean())
    p_value = min(p_value, 1.0)
    return {
        "mean_diff": float(diffs.mean()),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_value": float(p_value),
        "significant_at_0.05": bool(p_value < 0.05),
    }


MODEL_GETTERS = {
    "isolation_forest": lambda ds: classical_scores(ds, "isolation_forest"),
    "random_forest": lambda ds: classical_scores(ds, "random_forest"),
    "lstm": lambda ds: neural_scores(ds, "lstm")[:4],
    "transformer": lambda ds: neural_scores(ds, "transformer")[:4],
}


def run(dataset: str, model_a: str, model_b: str):
    y_val_a, scores_val_a, y_test_a, scores_test_a = MODEL_GETTERS[model_a](dataset)
    y_val_b, scores_val_b, y_test_b, scores_test_b = MODEL_GETTERS[model_b](dataset)

    # both scored the same frozen test split in the same order -- sanity check
    assert np.array_equal(y_test_a, y_test_b), f"test label order mismatch for {dataset}: {model_a} vs {model_b}"

    thr_a = pick_threshold(y_val_a, scores_val_a)
    thr_b = pick_threshold(y_val_b, scores_val_b)
    metrics_a = evaluate(y_test_a, scores_test_a, thr_a)
    metrics_b = evaluate(y_test_b, scores_test_b, thr_b)

    boot = paired_bootstrap_f1(y_test_a, scores_test_a, thr_a, scores_test_b, thr_b)

    result = {
        "dataset": dataset, "model_a": model_a, "model_b": model_b,
        "f1_a": metrics_a["f1"], "f1_b": metrics_b["f1"],
        "n_bootstrap": N_BOOTSTRAP,
        **boot,
    }
    print(f"[{dataset}] {model_a} (F1={metrics_a['f1']:.3f}) vs {model_b} (F1={metrics_b['f1']:.3f}): "
          f"diff={boot['mean_diff']:+.3f} 95%CI=[{boot['ci95_low']:+.3f}, {boot['ci95_high']:+.3f}] "
          f"p={boot['p_value']:.4f} {'SIGNIFICANT' if boot['significant_at_0.05'] else 'not significant'}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hdfs", "bgl", "thunderbird"])
    ap.add_argument("--model-a", required=True, choices=list(MODEL_GETTERS.keys()))
    ap.add_argument("--model-b", required=True, choices=list(MODEL_GETTERS.keys()))
    args = ap.parse_args()
    result = run(args.dataset, args.model_a, args.model_b)

    out_dir = RESULTS / "paired_tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_{args.model_a}_vs_{args.model_b}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
