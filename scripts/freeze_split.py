"""Freeze a 70/15/15 train/val/test split over sequences, stratified by label.

Writes data/splits/<dataset>_{train,val,test}.csv (seq_id, label) and a
manifest.json recording the seed and counts. Refuses to overwrite an existing
split unless --force is passed, since the whole point of freezing it is that
later phases never see it change.
"""
import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_SEQ = ROOT / "data" / "sequences"
DATA_SPLITS = ROOT / "data" / "splits"

SEED = 1337
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def freeze(dataset: str, force: bool = False):
    DATA_SPLITS.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_SPLITS / f"{dataset}_manifest.json"
    if manifest_path.exists() and not force:
        print(f"[{dataset}] split already frozen ({manifest_path}), skipping. Use --force to redo.")
        return

    seq_path = DATA_SEQ / f"{dataset}.csv"
    rows = []
    with seq_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["seq_id"], int(row["label"])))

    by_label: dict[int, list] = {}
    for seq_id, label in rows:
        by_label.setdefault(label, []).append(seq_id)

    rng = random.Random(SEED)
    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    for label, ids in by_label.items():
        ids = ids[:]
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * RATIOS["train"])
        n_val = int(n * RATIOS["val"])
        splits["train"] += [(i, label) for i in ids[:n_train]]
        splits["val"] += [(i, label) for i in ids[n_train:n_train + n_val]]
        splits["test"] += [(i, label) for i in ids[n_train + n_val:]]

    for split_name, items in splits.items():
        out_path = DATA_SPLITS / f"{dataset}_{split_name}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["seq_id", "label"])
            writer.writerows(items)

    manifest = {
        "dataset": dataset,
        "seed": SEED,
        "ratios": RATIOS,
        "n_total": len(rows),
        "counts": {k: len(v) for k, v in splits.items()},
        "anomaly_counts": {
            k: sum(1 for _, label in v if label == 1) for k, v in splits.items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[{dataset}] frozen split: {manifest['counts']} -> {manifest_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hdfs", "bgl", "thunderbird"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    freeze(args.dataset, force=args.force)
