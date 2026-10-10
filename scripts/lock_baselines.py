"""Aggregate the per-run baseline JSONs into one locked comparison table.

"Locked" means: this file is the reference classical-baseline numbers that
Phase 3's LSTM/transformer runs get compared against. Refuses to overwrite
unless --force is passed.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

DATASETS = ["hdfs", "bgl", "thunderbird"]
MODELS = ["isolation_forest", "random_forest"]

COLS = ["precision", "recall", "f1", "pr_auc", "false_positive_rate", "false_alarms_per_10k_events", "train_time_sec"]


def main(force: bool = False):
    locked_path = RESULTS / "baselines_locked.json"
    if locked_path.exists() and not force:
        print(f"Baselines already locked at {locked_path}, use --force to redo.")
        return

    table = {}
    md_lines = ["# Phase 2 — Classical Baselines (locked)\n"]
    md_lines.append("| Dataset | Model | Precision | Recall | F1 | PR-AUC | FPR | FA/10k seq | Train (s) |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|")

    for ds in DATASETS:
        table[ds] = {}
        for model in MODELS:
            path = RESULTS / f"{ds}_{model}.json"
            if not path.exists():
                print(f"missing: {path}")
                continue
            metrics = json.loads(path.read_text(encoding="utf-8"))
            table[ds][model] = {k: metrics[k] for k in COLS}
            md_lines.append(
                f"| {ds} | {model} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | "
                f"{metrics['f1']:.3f} | {metrics['pr_auc']:.3f} | {metrics['false_positive_rate']:.3f} | "
                f"{metrics['false_alarms_per_10k_events']:.1f} | {metrics['train_time_sec']:.1f} |"
            )

    locked_path.write_text(json.dumps(table, indent=2), encoding="utf-8")
    (RESULTS / "baselines_locked.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Locked -> {locked_path}")
    print(f"Locked -> {RESULTS / 'baselines_locked.md'}")
    print("\n".join(md_lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    main(force=args.force)
