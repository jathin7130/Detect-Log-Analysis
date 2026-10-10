"""Phase 6/8 — website. FastAPI + plain HTML/CSS/JS (no frontend framework,
per the "fast html" stack choice) serving:
  - a dashboard over REAL Phase 2/3 results
  - a demo page over REAL Phase 7 RCA output (scripts/run_rca.py's 20 cases)
    — the mock gate output from Phase 6 is gone; this now reads
    rca/results/rca_results.json directly, nothing hardcoded.
"""
import json
import math
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TESTBED = ROOT / "testbed"
RCA = ROOT / "rca"
WEBAPP = Path(__file__).resolve().parent

app = FastAPI(title="DevOps Log Intelligence")
app.mount("/static", StaticFiles(directory=WEBAPP / "static"), name="static")


def _sanitize_nan(obj):
    """NaN isn't valid JSON (and FastAPI/Starlette's JSONResponse enforces
    that strictly) -- aggregate_stats.py legitimately produces NaN for a
    95% CI half-width when a model only has 1 seed (lstm_combined), since
    a confidence interval from n=1 is undefined, not zero."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    return obj


def _load(path: Path, default):
    if not path.exists():
        return default
    return _sanitize_nan(json.loads(path.read_text(encoding="utf-8")))


@app.get("/")
def index():
    return FileResponse(WEBAPP / "templates" / "dashboard.html")


@app.get("/demo")
def demo_page():
    return FileResponse(WEBAPP / "templates" / "demo.html")


@app.get("/phases")
def phases_page():
    return FileResponse(WEBAPP / "templates" / "phases.html")


@app.get("/api/detector_metrics")
def detector_metrics():
    """Real Phase 3 aggregate stats (mean ± 95% CI across seeds) plus the
    Phase 2 classical baselines, merged into one structure the dashboard
    can render as a per-dataset bar comparison."""
    return _load(RESULTS / "phase3_aggregate_stats.json", {})


@app.get("/api/cross_domain")
def cross_domain():
    """All 6 directed cross-domain transfer pairs."""
    pairs = {}
    for path in sorted(RESULTS.glob("cross_domain_*.json")):
        key = path.stem.replace("cross_domain_", "")
        pairs[key] = _load(path, {})
    return pairs


@app.get("/api/drift")
def drift():
    out = {}
    for ds in ["hdfs", "bgl", "thunderbird"]:
        out[ds] = _load(RESULTS / f"{ds}_lstm_seed42_drift_eval.json", {})
    return out


@app.get("/api/testbed_incidents")
def testbed_incidents():
    return _load(TESTBED / "incidents" / "ground_truth.json", {"incidents": []})


def _rca_results():
    path = RCA / "results" / "rca_results.json"
    if not path.exists():
        raise HTTPException(500, "rca/results/rca_results.json not found — run scripts/run_rca.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def _rca_incidents_by_id():
    """rca_results.json doesn't persist the original log_lines (only
    evidence + diagnosis) -- they live in the incident corpus instead."""
    path = RCA / "incidents" / "incidents.json"
    if not path.exists():
        return {}
    incidents = json.loads(path.read_text(encoding="utf-8"))
    return {i["case_id"]: i for i in incidents}


# Curated defaults for the demo's initial view: one clean "answer" case and
# the one genuine self-abstain the hand-check found, so a first-time visitor
# sees both gate behaviors without having to hunt for them in the dropdown.
DEFAULT_ANSWER_CASE = "testbed_5"
DEFAULT_ABSTAIN_CASE = "bgl_bgl_5057"


@app.get("/api/rca_summary")
def rca_summary():
    """Aggregate gate metrics across all 20 RCA cases, for the dashboard
    overview card. The 16/20-correct figure comes from the manual
    hand-check (not something the gate can compute about itself), so it's
    hardcoded here rather than derived -- it's a record of that review,
    not a live-computed number.
    """
    results = _rca_results()
    n = len(results)
    n_gate_answer = sum(1 for r in results if r["gate"]["gate_decision"] == "answer")
    n_overrode = sum(1 for r in results if r["gate"]["gate_overrode_llm"])
    avg_coverage = sum(r["gate"]["evidence_coverage"] for r in results) / n
    return {
        "n_cases": n,
        "n_gate_answer": n_gate_answer,
        "n_gate_abstain": n - n_gate_answer,
        "n_gate_overrode_llm": n_overrode,
        "avg_evidence_coverage": avg_coverage,
        "hand_check_n_correct": 16,
        "hand_check_n_flagged": 2,
        "hand_check_n_abstain": 2,
    }


@app.get("/api/demo_cases")
def demo_cases():
    """Lists every real RCA case for the demo page's selector dropdown."""
    results = _rca_results()
    return {
        "default_answer_case": DEFAULT_ANSWER_CASE,
        "default_abstain_case": DEFAULT_ABSTAIN_CASE,
        "cases": [
            {
                "case_id": r["case_id"],
                "source": r["source"],
                "gate_decision": r["gate"]["gate_decision"],
                "gate_overrode_llm": r["gate"]["gate_overrode_llm"],
            }
            for r in results
        ],
    }


@app.get("/api/demo_incident")
def demo_incident(case_id: str = DEFAULT_ANSWER_CASE):
    """Real Phase 7 RCA output for one case — retrieval evidence, the raw
    log window, the LLM's own output, and the gate's independently-checked
    decision (which can and does differ from what the LLM claimed, see
    gate_overrode_llm)."""
    results = _rca_results()
    case = next((r for r in results if r["case_id"] == case_id), None)
    if case is None:
        raise HTTPException(404, f"no RCA case with id {case_id!r}")

    incident = _rca_incidents_by_id().get(case_id, {})

    return {
        "case_id": case["case_id"],
        "source": case["source"],
        "ground_truth": case["ground_truth"],
        "log_lines": incident.get("log_lines", []),
        "evidence": case["evidence"],
        "gate": {
            "llm_decision": case["gate"]["llm_decision"],
            "decision": case["gate"]["gate_decision"],
            "gate_overrode_llm": case["gate"]["gate_overrode_llm"],
            "diagnosis": case["llm_output"].get("diagnosis", ""),
            "evidence_coverage": case["gate"]["evidence_coverage"],
            "confidence": case["gate"]["confidence"],
            "mock": False,
        },
    }
