"""Mask PII then run Drain3 template mining over a raw Loghub dataset.

Usage:
    python scripts/mask_and_parse.py --dataset hdfs
    python scripts/mask_and_parse.py --dataset bgl

Writes, per dataset, into data/masked/<name>/:
    parsed.csv     - one row per log line: structured fields + masked raw
                     content + the Drain3 template id/template it matched
    templates.csv  - event_id -> template string (the parser's vocabulary)
    drain3_state.json - persisted Drain3 template-tree snapshot
"""
import argparse
import csv
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mask import mask_content  # noqa: E402

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_MASKED = ROOT / "data" / "masked"

HDFS_LINE_RE = re.compile(
    r"^(?P<date>\d{6})\s+(?P<time>\d{6})\s+(?P<pid>\d+)\s+(?P<level>\w+)\s+(?P<component>[^:]+):\s*(?P<content>.*)$"
)
BLOCK_ID_RE = re.compile(r"blk_-?\d+")


def parse_hdfs_line(line: str):
    m = HDFS_LINE_RE.match(line)
    if not m:
        return None
    d = m.groupdict()
    blk = BLOCK_ID_RE.search(d["content"])
    d["block_id"] = blk.group(0) if blk else ""
    return d


def parse_bgl_line(line: str):
    parts = line.rstrip("\n").split(maxsplit=9)
    if len(parts) < 10:
        return None
    label, timestamp, date, node, time_, node_repeat, mtype, component, level, content = parts
    return {
        "label": label,
        "timestamp": timestamp,
        "date": date,
        "node": node,
        "time": time_,
        "type": mtype,
        "component": component,
        "level": level,
        "content": content,
    }


def parse_thunderbird_line(line: str):
    parts = line.rstrip("\n").split(maxsplit=9)
    if len(parts) < 10:
        return None
    label, timestamp, date, user, month, day, time_, location, component, content = parts
    return {
        "label": label,
        "timestamp": timestamp,
        "date": date,
        "user": user,
        "month": month,
        "day": day,
        "time": time_,
        "location": location,
        "component": component.rstrip(":"),
        "content": content,
    }


DATASETS = {
    "hdfs": {
        "raw": DATA_RAW / "HDFS_v1" / "HDFS.log",
        "parser": parse_hdfs_line,
        "fields": ["date", "time", "pid", "level", "component", "block_id"],
        "mask_fields": [],
    },
    "bgl": {
        "raw": DATA_RAW / "BGL" / "BGL.log",
        "parser": parse_bgl_line,
        "fields": ["label", "timestamp", "date", "node", "time", "type", "component", "level"],
        "mask_fields": ["node"],
    },
    "thunderbird": {
        "raw": DATA_RAW / "Thunderbird" / "Thunderbird.log",
        "parser": parse_thunderbird_line,
        "fields": ["label", "timestamp", "date", "user", "month", "day", "time", "location", "component"],
        "mask_fields": ["user", "location"],
    },
}


def build_template_miner(state_path: Path) -> TemplateMiner:
    config = TemplateMinerConfig()
    config.load(str(ROOT / "configs" / "drain3.ini")) if (ROOT / "configs" / "drain3.ini").exists() else None
    config.profiling_enabled = False
    miner = TemplateMiner(config=config)
    return miner


def run(dataset: str, limit: int | None = None):
    spec = DATASETS[dataset]
    raw_path = spec["raw"]
    out_dir = DATA_MASKED / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    miner = build_template_miner(out_dir / "drain3_state.json")
    parser = spec["parser"]
    fields = spec["fields"]
    mask_fields = set(spec.get("mask_fields", []))

    parsed_path = out_dir / "parsed.csv"
    n_total = 0
    n_parsed = 0
    t0 = time.time()

    with raw_path.open("r", encoding="utf-8", errors="replace") as fin, parsed_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        writer = csv.writer(fout)
        writer.writerow(["line_no", *fields, "masked_content", "event_id", "template"])

        for line_no, line in enumerate(fin, start=1):
            if limit and line_no > limit:
                break
            n_total += 1
            rec = parser(line)
            if rec is None:
                continue
            for mf in mask_fields:
                rec[mf] = mask_content(rec[mf])
            masked = mask_content(rec["content"])
            if dataset == "hdfs":
                # block id is kept in its own column for session grouping;
                # strip it from the text handed to Drain3 so templates
                # generalize across blocks instead of one cluster per block id.
                template_input = BLOCK_ID_RE.sub("<BLK>", masked)
            else:
                template_input = masked
            result = miner.add_log_message(template_input)
            writer.writerow(
                [line_no, *[rec.get(f, "") for f in fields], masked, result["cluster_id"], result["template_mined"]]
            )
            n_parsed += 1
            if n_parsed % 500_000 == 0:
                elapsed = time.time() - t0
                print(f"[{dataset}] {n_parsed:,} lines parsed ({elapsed:.0f}s elapsed)", flush=True)

    templates_path = out_dir / "templates.csv"
    with templates_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "template", "occurrences"])
        for cluster in miner.drain.clusters:
            writer.writerow([cluster.cluster_id, cluster.get_template(), cluster.size])

    elapsed = time.time() - t0
    print(
        f"[{dataset}] done: {n_parsed:,}/{n_total:,} lines parsed, "
        f"{len(miner.drain.clusters)} templates mined, {elapsed:.0f}s total",
        flush=True,
    )
    print(f"[{dataset}] -> {parsed_path}")
    print(f"[{dataset}] -> {templates_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--limit", type=int, default=None, help="only process first N lines (debug)")
    args = ap.parse_args()
    run(args.dataset, limit=args.limit)
