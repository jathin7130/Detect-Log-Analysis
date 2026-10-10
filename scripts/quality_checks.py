"""Data-quality checks on the built sequences: coverage, length distribution,
and label balance per dataset. Writes a summary to results/quality_report.md.
"""
import csv
import statistics as stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_MASKED = ROOT / "data" / "masked"
DATA_SEQ = ROOT / "data" / "sequences"
RESULTS = ROOT / "results"

DATASETS = ["hdfs", "bgl", "thunderbird"]


def template_coverage(dataset: str):
    parsed_path = DATA_MASKED / dataset / "parsed.csv"
    total = 0
    matched = 0
    with parsed_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row.get("event_id"):
                matched += 1
    return total, matched


def sequence_stats(dataset: str):
    seq_path = DATA_SEQ / f"{dataset}.csv"
    lengths = []
    labels = []
    with seq_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lengths.append(int(row["length"]))
            labels.append(int(row["label"]))
    n = len(lengths)
    n_anom = sum(labels)
    return {
        "n_sequences": n,
        "n_anomaly": n_anom,
        "n_normal": n - n_anom,
        "anomaly_rate": n_anom / n if n else 0,
        "len_min": min(lengths) if lengths else 0,
        "len_max": max(lengths) if lengths else 0,
        "len_mean": stats.mean(lengths) if lengths else 0,
        "len_median": stats.median(lengths) if lengths else 0,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    lines = ["# Data Quality Report\n"]
    for ds in DATASETS:
        lines.append(f"## {ds}\n")

        total, matched = template_coverage(ds)
        coverage_pct = 100 * matched / total if total else 0
        lines.append(f"- Template coverage: {matched:,} / {total:,} lines ({coverage_pct:.2f}%)")

        s = sequence_stats(ds)
        lines.append(f"- Sequences: {s['n_sequences']:,} total")
        lines.append(
            f"- Label balance: {s['n_normal']:,} normal / {s['n_anomaly']:,} anomaly "
            f"({100*s['anomaly_rate']:.3f}% anomaly rate)"
        )
        lines.append(
            f"- Sequence length: min={s['len_min']}, max={s['len_max']}, "
            f"mean={s['len_mean']:.1f}, median={s['len_median']:.1f}"
        )
        lines.append("")

        print(f"[{ds}] coverage={coverage_pct:.2f}%  n_seq={s['n_sequences']:,}  "
              f"anomaly_rate={100*s['anomaly_rate']:.3f}%  "
              f"len(min/mean/max)={s['len_min']}/{s['len_mean']:.1f}/{s['len_max']}")

    report_path = RESULTS / "quality_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nFull report -> {report_path}")


if __name__ == "__main__":
    main()
