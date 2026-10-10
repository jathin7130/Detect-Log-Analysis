"""Closes a second gap: the proposal's sub-question 4 asks "how accuracy
changes when latency, throughput and computational cost are taken into
account together with predictive performance" -- Phase 3 reported accuracy
metrics and separately logged training times, but never consolidated them
into one cost-vs-accuracy view. This does that.
"""
import json
import re
import statistics as stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RCA = ROOT / "rca"
LOGS = ROOT / "logs"

ELAPSED_RE = re.compile(r"\(([\d.]+)s elapsed\)")

DATASETS = ["hdfs", "bgl", "thunderbird"]
SEEDS = [42, 123, 777]


def classical_cost(dataset: str, model: str):
    path = RESULTS / f"{dataset}_{model}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return {"train_time_sec": d.get("train_time_sec"), "f1": d.get("f1")}


def parse_elapsed_from_log(dataset: str, model: str, seed: int):
    """train_lstm.py/train_transformer.py print elapsed time per epoch but
    don't persist it into history.json -- the log file is the only record."""
    log_dir = LOGS / f"{model}_runs"
    log_path = log_dir / f"{dataset}_seed{seed}.log"
    if not log_path.exists():
        return None
    matches = ELAPSED_RE.findall(log_path.read_text(encoding="utf-8"))
    return float(matches[-1]) if matches else None


def neural_cost(dataset: str, model: str):
    train_times, f1s = [], []
    for seed in SEEDS:
        eval_path = RESULTS / f"{dataset}_{model}_seed{seed}_eval.json"
        elapsed = parse_elapsed_from_log(dataset, model, seed)
        if elapsed is not None:
            train_times.append(elapsed)
        if eval_path.exists():
            e = json.loads(eval_path.read_text(encoding="utf-8"))
            f1s.append(e["f1"])
    return {
        "train_time_sec_mean": stats.mean(train_times) if train_times else None,
        "f1_mean": stats.mean(f1s) if f1s else None,
    }


def rca_latency():
    path = RCA / "results" / "rca_results.json"
    if not path.exists():
        return None
    results = json.loads(path.read_text(encoding="utf-8"))
    by_source = {}
    for r in results:
        by_source.setdefault(r["source"], []).append(r["elapsed_s"])
    out = {"overall_mean_s": stats.mean([r["elapsed_s"] for r in results]), "n_cases": len(results)}
    for src, times in by_source.items():
        out[f"{src}_mean_s"] = stats.mean(times)
    return out


def main():
    lines = ["# Cost / Latency vs. Accuracy (closes proposal sub-question 4)\n"]

    lines.append("## Training cost vs. accuracy\n")
    lines.append("| Dataset | Model | Train time | F1 | Cost per 0.1 F1 |")
    lines.append("|---|---|---|---|---|")
    table = {}
    for ds in DATASETS:
        for model in ["isolation_forest", "random_forest"]:
            c = classical_cost(ds, model)
            if c and c["train_time_sec"] is not None and c["f1"]:
                cost_per_point = c["train_time_sec"] / (c["f1"] * 10)
                lines.append(f"| {ds} | {model} | {c['train_time_sec']:.1f}s | {c['f1']:.3f} | {cost_per_point:.2f}s |")
                table.setdefault(ds, {})[model] = c
        for model in ["lstm", "transformer"]:
            c = neural_cost(ds, model)
            if c["train_time_sec_mean"] is not None and c["f1_mean"]:
                cost_per_point = c["train_time_sec_mean"] / (c["f1_mean"] * 10)
                lines.append(f"| {ds} | {model} | {c['train_time_sec_mean']:.1f}s (mean/3 seeds) | "
                              f"{c['f1_mean']:.3f} | {cost_per_point:.2f}s |")
                table.setdefault(ds, {})[model] = c

    lines.append("\n*\"Cost per 0.1 F1\" = training seconds spent per 0.1 of F1 achieved — a cheap, "
                  "legible way to compare \"was the extra compute worth it\" across wildly different "
                  "training costs; lower is more cost-efficient.*\n")

    lines.append("\n## RCA latency (time-to-diagnosis)\n")
    rca = rca_latency()
    if rca:
        lines.append(f"- Overall mean: **{rca['overall_mean_s']:.1f}s per incident** (n={rca['n_cases']}, "
                      f"retrieval + local LLM inference + gate check, end to end)")
        for k, v in rca.items():
            if k.endswith("_mean_s") and k != "overall_mean_s":
                lines.append(f"- {k.replace('_mean_s', '')}: {v:.1f}s")
        lines.append("\nThis is the proposal's \"automated time-to-diagnosis\" metric (Section 3.3) -- "
                      "note this is a reproducible *pipeline* latency measurement, not a claim about real "
                      "MTTR reduction, consistent with the proposal's own caveat that a real MTTR claim "
                      "needs production/human-operator experiments this project doesn't run.")

    out_path = RESULTS / "cost_latency_table.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    out_json = RESULTS / "cost_latency_table.json"
    out_json.write_text(json.dumps({"training": table, "rca_latency": rca}, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {out_path}\n-> {out_json}")


if __name__ == "__main__":
    main()
