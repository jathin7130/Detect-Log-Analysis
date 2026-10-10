"""Phase 4's final step: confirm the testbed's logs flow through the same
masking + Drain3 pipeline built for HDFS/BGL/Thunderbird, and that the
recorded fault windows are actually visible as anomalous bursts in the
resulting event-id stream (a basic sanity check before Phase 7 builds RCA
on top of this corpus).
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from mask import mask_content  # noqa: E402

from drain3 import TemplateMiner

ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "logs"
INCIDENTS_DIR = ROOT / "incidents"
OUT_DIR = ROOT / "parsed"

LINE_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (?P<level>\w+) (?P<service>\w+): (?P<content>.*)$")


def parse_all_services():
    miner = TemplateMiner()
    rows = []
    for log_path in sorted(LOGS_DIR.glob("*.log")):
        service = log_path.stem
        with log_path.open(encoding="utf-8") as f:
            for line in f:
                m = LINE_RE.match(line.rstrip("\n"))
                if not m:
                    continue
                masked = mask_content(m.group("content"))
                result = miner.add_log_message(masked)
                rows.append({
                    "timestamp": m.group("ts"),
                    "service": service,
                    "level": m.group("level"),
                    "masked_content": masked,
                    "event_id": result["cluster_id"],
                    "template": result["template_mined"],
                })

    rows.sort(key=lambda r: r["timestamp"])
    return rows, miner


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, miner = parse_all_services()

    parsed_path = OUT_DIR / "testbed_parsed.csv"
    with parsed_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "service", "level", "masked_content", "event_id", "template"])
        writer.writeheader()
        writer.writerows(rows)

    templates_path = OUT_DIR / "testbed_templates.csv"
    with templates_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "template", "occurrences"])
        for cluster in miner.drain.clusters:
            writer.writerow([cluster.cluster_id, cluster.get_template(), cluster.size])

    print(f"parsed {len(rows)} lines across {len(set(r['service'] for r in rows))} services, "
          f"{len(miner.drain.clusters)} templates -> {parsed_path}")

    # Sanity check: do the ERROR/WARNING lines cluster in time around the
    # recorded fault injection windows?
    gt_path = INCIDENTS_DIR / "ground_truth.json"
    if not gt_path.exists():
        print("no ground_truth.json yet, skipping incident-window check")
        return

    ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
    error_rows = [r for r in rows if r["level"] in ("ERROR", "WARNING")]

    print(f"\n{len(error_rows)} ERROR/WARNING lines total across {len(ground_truth['incidents'])} injected incidents")
    for inc in ground_truth["incidents"]:
        inc_time = inc["injected_at"][11:19]  # HH:MM:SS
        nearby = [r for r in error_rows if abs(_to_seconds(r["timestamp"][11:19]) - _to_seconds(inc_time)) <= inc["fault_duration_s"] + 2]
        print(f"  {inc['service']:<10} {inc['fault_type']:<22} at {inc_time} (dur={inc['fault_duration_s']}s): "
              f"{len(nearby)} ERROR/WARNING lines nearby")


def _to_seconds(hms: str) -> int:
    h, m, s = map(int, hms.split(":"))
    return h * 3600 + m * 60 + s


if __name__ == "__main__":
    main()
