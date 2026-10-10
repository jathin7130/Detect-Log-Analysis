"""Build the RCA incident corpus for Phase 7: the 8 testbed fault-injection
incidents (rich, known ground truth: service + fault type) plus a sample of
real detected anomalies from HDFS/BGL/Thunderbird (true positives from the
trained LSTM seed42, so these are genuine correctly-flagged anomalies, not
just any random window) to round out the hand-check set toward ~20 cases.
"""
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from lstm_model import DeepLogLSTM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTBED = ROOT / "testbed"
DATA_SEQ = ROOT / "data" / "sequences"
DATA_SPLITS = ROOT / "data" / "splits"
DATA_MASKED = ROOT / "data" / "masked"
MODELS = ROOT / "models"
RCA = ROOT / "rca"

N_REAL_SAMPLES_PER_DATASET = 4


def build_testbed_incidents():
    manifest = json.loads((TESTBED / "incidents" / "ground_truth.json").read_text(encoding="utf-8"))
    log_rows = []
    for log_path in sorted((TESTBED / "logs").glob("*.log")):
        service = log_path.stem
        for line in log_path.read_text(encoding="utf-8").splitlines():
            # format: "YYYY-MM-DD HH:MM:SS LEVEL service: message"
            parts = line.split(" ", 3)
            if len(parts) < 4:
                continue
            ts = f"{parts[0]} {parts[1]}"
            level = parts[2]
            _, _, msg = parts[3].partition(": ")
            log_rows.append({"ts": ts, "service": service, "level": level.strip(), "msg": msg.strip()})
    log_rows.sort(key=lambda r: r["ts"])

    incidents = []
    for i, inc in enumerate(manifest["incidents"]):
        inc_time = datetime.fromisoformat(inc["injected_at"]).replace(tzinfo=None)
        window_start = inc_time - timedelta(seconds=2)
        window_end = inc_time + timedelta(seconds=inc["fault_duration_s"] + 5)

        window_lines = [
            r for r in log_rows
            if window_start <= datetime.strptime(r["ts"], "%Y-%m-%d %H:%M:%S") <= window_end
        ]

        incidents.append({
            "case_id": f"testbed_{i}",
            "source": "testbed",
            "ground_truth": {"service": inc["service"], "fault_type": inc["fault_type"]},
            "log_lines": window_lines,
            "query_text": " ".join(r["msg"] for r in window_lines if r["level"] in ("ERROR", "WARNING")),
        })
    return incidents


def max_event_id(dataset: str) -> int:
    max_id = 0
    with (DATA_MASKED / dataset / "templates.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            max_id = max(max_id, int(row["event_id"]))
    return max_id


def load_templates(dataset: str) -> dict:
    templates = {}
    with (DATA_MASKED / dataset / "templates.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            templates[int(row["event_id"])] = row["template"]
    return templates


def sample_real_anomalies(dataset: str, n: int, seed: int = 42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODELS / f"{dataset}_lstm_seed{seed}.pt", map_location=device, weights_only=False)
    model = DeepLogLSTM(vocab_size=ckpt["vocab_size"], embed_dim=ckpt["embed_dim"], hidden_dim=ckpt["hidden_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    window_size = ckpt["window_size"]

    templates = load_templates(dataset)

    seqs = {}
    with (DATA_SEQ / f"{dataset}.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seqs[row["seq_id"]] = [int(e) for e in row["event_ids"].split()]

    test_anomalous = []
    with (DATA_SPLITS / f"{dataset}_test.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] == "1" and row["seq_id"] in seqs:
                test_anomalous.append(row["seq_id"])

    rng = np.random.RandomState(seed)
    rng.shuffle(test_anomalous)

    incidents = []
    with torch.no_grad():
        for sid in test_anomalous:
            if len(incidents) >= n:
                break
            seq = seqs[sid]
            if len(seq) <= window_size:
                continue
            arr = np.asarray(seq, dtype=np.int64)
            windows = np.lib.stride_tricks.sliding_window_view(arr, window_size)[:len(seq) - window_size]
            y_true = arr[window_size:window_size + len(windows)]
            xb = torch.from_numpy(windows).to(device)
            logits = model(xb)
            topk = torch.topk(logits, k=3, dim=1).indices.cpu().numpy()
            miss = ~(topk == y_true[:, None]).any(axis=1)
            if not miss.any():
                continue  # model didn't actually flag this one as anomalous

            template_seq = [templates.get(e, f"<unknown:{e}>") for e in seq]
            incidents.append({
                "case_id": f"{dataset}_{sid}",
                "source": dataset,
                "ground_truth": {"label": "anomaly", "seq_id": sid, "detector": "lstm_seed42"},
                "log_lines": [{"ts": "", "service": dataset, "level": "INFO", "msg": t} for t in template_seq],
                "query_text": " ".join(template_seq),
            })
    return incidents


def main():
    RCA.mkdir(parents=True, exist_ok=True)
    all_incidents = build_testbed_incidents()
    for ds in ["hdfs", "bgl", "thunderbird"]:
        all_incidents += sample_real_anomalies(ds, N_REAL_SAMPLES_PER_DATASET)

    out_path = RCA / "incidents" / "incidents.json"
    out_path.write_text(json.dumps(all_incidents, indent=2), encoding="utf-8")
    print(f"built {len(all_incidents)} incidents "
          f"({sum(1 for i in all_incidents if i['source']=='testbed')} testbed, "
          f"{sum(1 for i in all_incidents if i['source']!='testbed')} real-detected) -> {out_path}")


if __name__ == "__main__":
    main()
