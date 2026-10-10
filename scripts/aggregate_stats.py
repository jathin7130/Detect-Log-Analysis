"""Aggregate per-seed results into mean ± 95% CI per (dataset, model).

Classical baselines (Isolation Forest / Random Forest) have no seed
variation in how we ran them (single deterministic fit each), so they're
reported as single-run numbers, not mean±CI — only the neural models
(trained across 3 seeds) get a confidence interval, which is the honest
way to report it rather than manufacturing a CI from n=1.
"""
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

METRICS = ["precision", "recall", "f1", "pr_auc", "false_alarms_per_10k_events"]
DATASETS = ["hdfs", "bgl", "thunderbird"]
SEEDED_MODELS = ["lstm", "transformer", "lstm_combined"]
SINGLE_RUN_MODELS = ["isolation_forest", "random_forest"]

# two-sided 95% t critical values for small df, keyed by n-1
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def mean_ci(values: list[float], confidence_975_table=T_975):
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, float("nan")
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(variance)
    t_crit = confidence_975_table.get(n - 1, 1.96)  # fall back to normal approx
    half_width = t_crit * sd / math.sqrt(n)
    return mean, half_width


def collect_seeded(dataset: str, model: str):
    per_metric = {m: [] for m in METRICS}
    seeds_found = []
    # Exact match on a numeric seed, e.g. "hdfs_lstm_seed42_eval.json" -- a
    # plain glob("..._seed*_eval.json") also matches "..._seed42_drift_eval.json"
    # (the drift-robustness eval, a different file, not a 4th seed run).
    pattern = re.compile(rf"^{re.escape(dataset)}_{re.escape(model)}_seed(\d+)_eval\.json$")
    for path in sorted(RESULTS.glob(f"{dataset}_{model}_seed*_eval.json")):
        if not pattern.match(path.name):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        seeds_found.append(data.get("seed"))
        for m in METRICS:
            if m in data:
                per_metric[m].append(data[m])
    return per_metric, seeds_found


def collect_single(dataset: str, model: str):
    path = RESULTS / f"{dataset}_{model}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    report = {}
    md_lines = ["# Phase 3 — Aggregated Stats (mean ± 95% CI across seeds)\n"]
    md_lines.append("| Dataset | Model | " + " | ".join(METRICS) + " | n_seeds |")
    md_lines.append("|---|---|" + "---|" * (len(METRICS) + 1))

    for ds in DATASETS:
        report[ds] = {}
        for model in SEEDED_MODELS:
            per_metric, seeds = collect_seeded(ds, model)
            if not seeds:
                continue
            row = {}
            cells = []
            for m in METRICS:
                vals = per_metric[m]
                if not vals:
                    cells.append("—")
                    continue
                mean, half_width = mean_ci(vals)
                row[m] = {"mean": mean, "ci95_half_width": half_width, "values": vals}
                if math.isnan(half_width):
                    cells.append(f"{mean:.3f}")
                else:
                    cells.append(f"{mean:.3f} ± {half_width:.3f}")
            report[ds][model] = {"metrics": row, "seeds": seeds}
            md_lines.append(f"| {ds} | {model} | " + " | ".join(cells) + f" | {len(seeds)} |")

        for model in SINGLE_RUN_MODELS:
            data = collect_single(ds, model)
            if not data:
                continue
            report[ds][model] = {"metrics": {m: data[m] for m in METRICS if m in data}, "seeds": "n/a (single run)"}
            cells = [f"{data[m]:.3f}" for m in METRICS if m in data]
            md_lines.append(f"| {ds} | {model} | " + " | ".join(cells) + " | 1 (no seed variation) |")

    out_json = RESULTS / "phase3_aggregate_stats.json"
    out_md = RESULTS / "phase3_aggregate_stats.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"-> {out_json}")
    print(f"-> {out_md}")
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
