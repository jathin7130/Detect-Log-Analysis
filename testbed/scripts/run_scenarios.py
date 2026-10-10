"""Drive the fault-injection testbed: continuous background traffic plus a
schedule of known faults, with ground truth recorded for later RCA
evaluation (Phase 7).

Usage: python run_scenarios.py [--duration-s 300] [--request-interval-s 1.0]
"""
import argparse
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

GATEWAY_URL = "http://localhost:8101"
FAULT_ENDPOINTS = {
    "gateway": "http://localhost:8101/_fault",
    "orders": "http://localhost:8102/_fault",
    "inventory": "http://localhost:8103/_fault",
    "payments": "http://localhost:8104/_fault",
}

# (offset_s, target_service, fault_type, duration_s)
SCENARIO = [
    (20, "inventory", "db_failure", 8),
    (50, "orders", "dependency_timeout", 10),
    (85, "payments", "resource_exhaustion", 15),
    (120, "inventory", "pod_restart", 5),
    (150, "orders", "db_failure", 10),
    (185, "gateway", "dependency_timeout", 8),
    (215, "payments", "pod_restart", 5),
    (245, "inventory", "resource_exhaustion", 12),
]

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "incidents"

_stop = threading.Event()
_request_count = {"n": 0, "n_errors": 0}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_generator(interval_s: float):
    while not _stop.is_set():
        try:
            resp = requests.get(f"{GATEWAY_URL}/checkout", timeout=10)
            _request_count["n"] += 1
            if resp.status_code != 200:
                _request_count["n_errors"] += 1
        except requests.exceptions.RequestException:
            _request_count["n"] += 1
            _request_count["n_errors"] += 1
        time.sleep(interval_s)


def run(duration_s: int, request_interval_s: float):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ground_truth = []

    gen_thread = threading.Thread(target=load_generator, args=(request_interval_s,), daemon=True)
    gen_thread.start()

    start = time.time()
    print(f"scenario started at {now_iso()}, duration={duration_s}s")

    for offset_s, service, fault_type, fault_duration_s in SCENARIO:
        if offset_s >= duration_s:
            continue
        sleep_for = start + offset_s - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)

        injected_at = now_iso()
        try:
            requests.post(
                FAULT_ENDPOINTS[service],
                json={"type": fault_type, "duration_s": fault_duration_s},
                timeout=5,
            )
            print(f"[{injected_at}] injected {fault_type} on {service} for {fault_duration_s}s")
        except requests.exceptions.RequestException as e:
            print(f"[{injected_at}] FAILED to inject {fault_type} on {service}: {e}")
            continue

        ground_truth.append({
            "injected_at": injected_at,
            "offset_s": offset_s,
            "service": service,
            "fault_type": fault_type,
            "fault_duration_s": fault_duration_s,
        })

    remaining = start + duration_s - time.time()
    if remaining > 0:
        time.sleep(remaining)

    _stop.set()
    gen_thread.join(timeout=5)

    manifest = {
        "started_at": datetime.fromtimestamp(start, tz=timezone.utc).isoformat(),
        "duration_s": duration_s,
        "request_interval_s": request_interval_s,
        "n_requests": _request_count["n"],
        "n_errors": _request_count["n_errors"],
        "incidents": ground_truth,
    }
    out_path = OUT_DIR / "ground_truth.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"done. {_request_count['n']} requests, {_request_count['n_errors']} errors, "
          f"{len(ground_truth)} incidents injected -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-s", type=int, default=280)
    ap.add_argument("--request-interval-s", type=float, default=1.0)
    args = ap.parse_args()
    run(args.duration_s, args.request_interval_s)
