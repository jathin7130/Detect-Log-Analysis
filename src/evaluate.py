"""Shared evaluation utilities for anomaly detectors.

Threshold is always selected on the validation set (maximizing F1) and then
applied, unchanged, to the test set — the test set never influences model
or threshold selection, per the frozen-split rule.
"""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def pick_threshold(y_val: np.ndarray, scores_val: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_val, scores_val)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    # precision_recall_curve returns one more point than thresholds
    best_idx = np.argmax(f1[:-1]) if len(thresholds) else 0
    return float(thresholds[best_idx]) if len(thresholds) else 0.5


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    n_total = len(y_true)
    false_alarms_per_10k = fp / n_total * 10_000 if n_total else 0.0

    pr_auc = average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else float("nan")

    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "false_positive_rate": fpr,
        "false_alarms_per_10k_events": false_alarms_per_10k,
        "n_total": n_total,
    }
