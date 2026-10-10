"""Turn parsed/templated log lines into labeled sequences.

HDFS: session-based — group by block_id (the natural session identifier),
      label from the official anomaly_label.csv.
BGL / Thunderbird: no natural session id, so use fixed-size non-overlapping
      sliding windows over the event stream (window = 100 events), labeled
      anomalous if any line in the window carries a non-"-" alert label.

Output: data/sequences/<dataset>.csv with columns
    seq_id, length, label (0=normal, 1=anomaly), event_ids (space-separated)
"""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_MASKED = ROOT / "data" / "masked"
DATA_SEQ = ROOT / "data" / "sequences"
DATA_RAW = ROOT / "data" / "raw"

WINDOW_SIZE = 100


def build_hdfs_sequences():
    parsed_path = DATA_MASKED / "hdfs" / "parsed.csv"
    label_path = DATA_RAW / "HDFS_v1" / "preprocessed" / "anomaly_label.csv"

    labels = {}
    with label_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0

    sequences = {}  # block_id -> list[event_id]
    with parsed_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bid = row["block_id"]
            if not bid:
                continue
            sequences.setdefault(bid, []).append(row["event_id"])

    out_path = DATA_SEQ / "hdfs.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_unlabeled = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq_id", "length", "label", "event_ids"])
        for bid, events in sequences.items():
            if bid not in labels:
                n_unlabeled += 1
                continue
            writer.writerow([bid, len(events), labels[bid], " ".join(events)])
            n_written += 1

    print(f"[hdfs] {n_written} block sequences written, {n_unlabeled} blocks had no label, -> {out_path}")


def build_windowed_sequences(dataset: str):
    parsed_path = DATA_MASKED / dataset / "parsed.csv"
    out_path = DATA_SEQ / f"{dataset}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with parsed_path.open(encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["seq_id", "length", "label", "event_ids"])

        window_events = []
        window_anomalous = False
        seq_idx = 0

        for row in reader:
            window_events.append(row["event_id"])
            if row.get("label", "-") != "-":
                window_anomalous = True
            if len(window_events) == WINDOW_SIZE:
                writer.writerow(
                    [f"{dataset}_{seq_idx}", len(window_events), int(window_anomalous), " ".join(window_events)]
                )
                n_written += 1
                seq_idx += 1
                window_events = []
                window_anomalous = False

        if window_events:  # trailing partial window
            writer.writerow(
                [f"{dataset}_{seq_idx}", len(window_events), int(window_anomalous), " ".join(window_events)]
            )
            n_written += 1

    print(f"[{dataset}] {n_written} windowed sequences (window={WINDOW_SIZE}) written -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hdfs", "bgl", "thunderbird"])
    args = ap.parse_args()
    if args.dataset == "hdfs":
        build_hdfs_sequences()
    else:
        build_windowed_sequences(args.dataset)
