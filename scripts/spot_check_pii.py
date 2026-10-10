"""Spot-check masked output for residual PII.

Samples rows from each dataset's parsed.csv and scans the masked text/fields
for anything the masking regexes in src/mask.py might have missed: raw IPs,
emails, credential-looking strings, and unmasked hostnames. Reports counts
and a few example lines so a human can eyeball them.
"""
import csv
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

ROOT = Path(__file__).resolve().parent.parent
DATA_MASKED = ROOT / "data" / "masked"

SAMPLE_SIZE = 20000
SEED = 42

RESIDUAL_CHECKS = {
    "raw_ipv4": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "email_like": re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b"),
    "user_at_host": re.compile(r"\b[\w.+-]+@[\w.-]+\b"),
    "credential_kv": re.compile(r"\b(?:password|passwd|token|secret|api[_-]?key)\s*[=:]", re.IGNORECASE),
    "unix_path": re.compile(r"(?:/[\w\-.$]+){2,}"),
    "hex_token_32plus": re.compile(r"\b[0-9a-fA-F]{32,}\b"),
}

DATASETS = ["hdfs", "bgl", "thunderbird"]


def sample_rows(path: Path, n: int, seed: int):
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def check_dataset(name: str):
    path = DATA_MASKED / name / "parsed.csv"
    if not path.exists():
        print(f"[{name}] parsed.csv not found, skipping")
        return

    rows = sample_rows(path, SAMPLE_SIZE, SEED)
    print(f"\n=== {name}: spot-checking {len(rows)} sampled rows out of file ===")

    hits = {k: [] for k in RESIDUAL_CHECKS}
    for row in rows:
        text_fields = " | ".join(v for k, v in row.items() if k not in ("line_no", "event_id"))
        for check_name, pattern in RESIDUAL_CHECKS.items():
            if pattern.search(text_fields):
                hits[check_name].append(text_fields[:160])

    for check_name, examples in hits.items():
        print(f"  {check_name}: {len(examples)} / {len(rows)} sampled rows flagged")
        for ex in examples[:3]:
            print(f"    e.g. {ex}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    args = ap.parse_args()
    for ds in args.datasets:
        check_dataset(ds)
