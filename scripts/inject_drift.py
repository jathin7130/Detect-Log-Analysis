"""Inject controlled benign drift into a dataset's NORMAL test sequences.

Per the proposal's robustness-test design: a detector trained on today's
normal workflow shouldn't flag tomorrow's slightly-different-but-still-normal
workflow as anomalous. Three drift mechanisms, applied independently with
their own rates so their individual contribution is traceable:

  1. token substitution  — some existing benign event id is consistently
     renamed to a new id (simulates a template's wording changing after a
     software update: same meaning, different parsed template).
  2. new template insertion — a brand-new event id (never seen in training)
     is spliced into the sequence at random positions (simulates a new but
     benign log line type appearing, e.g. from a new library/feature).
  3. frequency tweak — some existing events are randomly duplicated or
     dropped, shifting event frequency without changing the vocabulary.

Only applied to sequences already labeled normal (label==0) in the test
split — a drifted ANOMALOUS sequence is a different experiment (it would
muddy "did drift cause a false alarm" with "did drift hide a real anomaly").

Output: data/drift/<dataset>_drifted.csv (seq_id, label, event_ids) and
        data/drift/<dataset>_drift_manifest.json (the rates used, for
        reproducibility).
"""
import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_SEQ = ROOT / "data" / "sequences"
DATA_SPLITS = ROOT / "data" / "splits"
DATA_MASKED = ROOT / "data" / "masked"
DATA_DRIFT = ROOT / "data" / "drift"

SEED = 2026
SUBSTITUTION_RATE = 0.10   # fraction of distinct benign event ids renamed
INSERTION_RATE = 0.05      # probability of inserting a new-template token per position
FREQ_TWEAK_RATE = 0.10     # probability of duplicating/dropping an event per position


def max_event_id(dataset: str) -> int:
    max_id = 0
    with (DATA_MASKED / dataset / "templates.csv").open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            max_id = max(max_id, int(row["event_id"]))
    return max_id


def load_test_normal(dataset: str):
    seqs = {}
    with (DATA_SEQ / f"{dataset}.csv").open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seqs[row["seq_id"]] = [int(e) for e in row["event_ids"].split()]

    test_ids = []
    with (DATA_SPLITS / f"{dataset}_test.csv").open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["label"] == "0":
                test_ids.append(row["seq_id"])

    return {sid: seqs[sid] for sid in test_ids if sid in seqs}


def inject(dataset: str):
    rng = random.Random(SEED)
    vocab_max = max_event_id(dataset)
    normal_seqs = load_test_normal(dataset)

    # Fixed rename map for token substitution — same original id always maps
    # to the same new id, consistent with "a template's wording changed".
    all_ids = sorted({eid for seq in normal_seqs.values() for eid in seq})
    n_substitute = max(1, int(len(all_ids) * SUBSTITUTION_RATE))
    substitute_ids = set(rng.sample(all_ids, min(n_substitute, len(all_ids))))
    rename_map = {eid: vocab_max + 1 + i for i, eid in enumerate(sorted(substitute_ids))}
    new_template_id = vocab_max + 1 + len(rename_map)  # one shared id for "a new benign template"

    drifted = {}
    for sid, seq in normal_seqs.items():
        out = []
        for eid in seq:
            # 1. token substitution
            eid = rename_map.get(eid, eid)
            out.append(eid)
            # 2. new template insertion (after this position)
            if rng.random() < INSERTION_RATE:
                out.append(new_template_id)
            # 3. frequency tweak: duplicate or drop with equal sub-probability
            r = rng.random()
            if r < FREQ_TWEAK_RATE / 2:
                out.append(eid)  # duplicate
            elif r < FREQ_TWEAK_RATE:
                out.pop()  # drop the event just appended
        if out:
            drifted[sid] = out

    DATA_DRIFT.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DRIFT / f"{dataset}_drifted.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq_id", "label", "event_ids"])
        for sid, seq in drifted.items():
            writer.writerow([sid, 0, " ".join(map(str, seq))])

    manifest = {
        "dataset": dataset,
        "seed": SEED,
        "substitution_rate": SUBSTITUTION_RATE,
        "insertion_rate": INSERTION_RATE,
        "freq_tweak_rate": FREQ_TWEAK_RATE,
        "n_ids_renamed": len(rename_map),
        "new_template_id": new_template_id,
        "n_sequences_drifted": len(drifted),
        "vocab_max_before_drift": vocab_max,
    }
    manifest_path = DATA_DRIFT / f"{dataset}_drift_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[{dataset}] drifted {len(drifted)} normal test sequences "
          f"({len(rename_map)} ids renamed, new_template_id={new_template_id}) -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hdfs", "bgl", "thunderbird"])
    args = ap.parse_args()
    inject(args.dataset)
