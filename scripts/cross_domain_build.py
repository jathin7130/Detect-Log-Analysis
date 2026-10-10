"""Re-map a target dataset's content into a source dataset's template-id
space, for cross-domain transfer evaluation.

Why this is needed: HDFS/BGL/Thunderbird event ids are each their own,
unrelated namespace (HDFS's event id 3 has nothing to do with BGL's event
id 3) — a model trained on HDFS's ids literally cannot be applied to BGL's
ids, they're not the same kind of thing. To make "train on source, test on
target" meaningful at all, both sides need to go through the SAME template
vocabulary. We don't persist Drain3 state during the main parse (see
scripts/mask_and_parse.py), so this rebuilds the source miner in memory by
replaying its already-masked content, then uses Drain3's match() (read-only,
creates no new clusters) to classify the target's content against it.
Anything in target that doesn't match any source template gets id 0 (the
same index already reserved for padding — an "unseen" signal the source
model was never trained to expect, which is the point: this is a lower
bound on cross-domain performance, not a best-case one).
"""
import argparse
import csv
import json
import time
from pathlib import Path

from drain3 import TemplateMiner

ROOT = Path(__file__).resolve().parent.parent
DATA_MASKED = ROOT / "data" / "masked"
DATA_SEQ = ROOT / "data" / "sequences"
DATA_CROSS = ROOT / "data" / "cross_domain"

GROUP_COL = {"hdfs": "block_id"}  # others use fixed 100-event windows
WINDOW_SIZE = 100


def rebuild_source_miner(source: str, limit: int = None) -> TemplateMiner:
    miner = TemplateMiner()
    path = DATA_MASKED / source / "parsed.csv"
    t0 = time.time()
    n = 0
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if limit and n >= limit:
                break
            miner.add_log_message(row["masked_content"])
            n += 1
    print(f"[{source}] rebuilt template miner from {n:,} lines "
          f"({len(miner.drain.clusters)} clusters) in {time.time()-t0:.1f}s")
    return miner


def build_target_windows(target: str, miner: TemplateMiner, limit: int = None):
    """Windowed (BGL/Thunderbird-style) cross-domain re-mapping."""
    path = DATA_MASKED / target / "parsed.csv"
    sequences = []
    current = []
    current_anom = False
    n_unseen, n_total = 0, 0

    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if limit and n_total >= limit:
                break
            n_total += 1
            cluster = miner.match(row["masked_content"])
            mapped_id = cluster.cluster_id if cluster is not None else 0
            if cluster is None:
                n_unseen += 1
            current.append(mapped_id)
            if row.get("label", "-") != "-":
                current_anom = True
            if len(current) == WINDOW_SIZE:
                sequences.append((current, int(current_anom)))
                current, current_anom = [], False
    if current:
        sequences.append((current, int(current_anom)))

    return sequences, n_unseen, n_total


def build_hdfs_target(miner: TemplateMiner, source_is_hdfs_label_available: bool, limit: int = None):
    """HDFS-as-target: group by block_id, label from anomaly_label.csv."""
    import csv as _csv
    path = DATA_MASKED / "hdfs" / "parsed.csv"
    label_path = ROOT / "data" / "raw" / "HDFS_v1" / "preprocessed" / "anomaly_label.csv"
    labels = {}
    with label_path.open(encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            labels[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0

    groups = {}
    n_unseen, n_total = 0, 0
    with path.open(encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            if limit and n_total >= limit:
                break
            n_total += 1
            bid = row.get("block_id", "")
            if not bid:
                continue
            cluster = miner.match(row["masked_content"])
            mapped_id = cluster.cluster_id if cluster is not None else 0
            if cluster is None:
                n_unseen += 1
            groups.setdefault(bid, []).append(mapped_id)

    sequences = [(events, labels.get(bid, 0)) for bid, events in groups.items() if bid in labels]
    return sequences, n_unseen, n_total


def run(source: str, target: str, source_limit: int = None, target_limit: int = None):
    if source == target:
        raise ValueError("source and target must differ")

    miner = rebuild_source_miner(source, limit=source_limit)

    if target == "hdfs":
        sequences, n_unseen, n_total = build_hdfs_target(miner, source_is_hdfs_label_available=True, limit=target_limit)
    else:
        sequences, n_unseen, n_total = build_target_windows(target, miner, limit=target_limit)

    DATA_CROSS.mkdir(parents=True, exist_ok=True)
    out_path = DATA_CROSS / f"{source}_to_{target}.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq_id", "label", "event_ids"])
        for i, (events, label) in enumerate(sequences):
            writer.writerow([f"{source}_to_{target}_{i}", label, " ".join(map(str, events))])

    manifest = {
        "source": source, "target": target,
        "n_sequences": len(sequences),
        "n_lines_total": n_total,
        "n_lines_unseen": n_unseen,
        "unseen_rate": n_unseen / n_total if n_total else 0,
        "source_vocab_size": len(miner.drain.clusters),
    }
    (DATA_CROSS / f"{source}_to_{target}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[{source}->{target}] {len(sequences)} sequences, unseen_rate={manifest['unseen_rate']:.1%} -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["hdfs", "bgl", "thunderbird"])
    ap.add_argument("--target", required=True, choices=["hdfs", "bgl", "thunderbird"])
    ap.add_argument("--source-limit", type=int, default=None)
    ap.add_argument("--target-limit", type=int, default=None)
    args = ap.parse_args()
    run(args.source, args.target, source_limit=args.source_limit, target_limit=args.target_limit)
