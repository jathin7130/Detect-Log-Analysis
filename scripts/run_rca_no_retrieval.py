"""Closes the third gap: the proposal's sub-question 3 asks whether
retrieval-grounded reasoning improves RCA "without leading to an increase in
unsupported claims" -- which requires a WITH-retrieval vs WITHOUT-retrieval
comparison. scripts/run_rca.py only ever ran with retrieval; this is the
without-retrieval arm on the identical 20 incidents, same LLM, same gate.

Evidence is replaced with an explicit "no evidence was retrieved" message
rather than just omitted, so the LLM can't quietly fall back on its own
unstated assumptions about what evidence might say -- the point is to see
what happens when it has to diagnose from the raw incident alone.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from llm_client import generate, extract_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RCA = ROOT / "rca"

NO_EVIDENCE_PROMPT_TEMPLATE = """You are a DevOps root-cause-analysis assistant. You diagnose the root cause of incidents using ONLY the evidence provided below. You must not invent causes that aren't supported by the evidence.

INCIDENT LOG LINES:
{log_lines}

RETRIEVED EVIDENCE:
No evidence was retrieved for this incident.

Instructions:
- If the evidence clearly supports a specific root cause, give that diagnosis and cite the evidence ids that support it.
- If the evidence does NOT clearly support a specific root cause, you MUST set "decision" to "abstain" rather than guess.
- Respond with ONLY a JSON object, no other text, in this exact shape:
{{"diagnosis": "<one or two sentences>", "cited_evidence_ids": ["<id1>", "<id2>"], "decision": "answer" or "abstain", "confidence": <float 0.0-1.0>}}
"""


def format_log_lines(log_lines, max_lines=15):
    priority = [l for l in log_lines if l["level"] in ("ERROR", "WARNING")]
    context = [l for l in log_lines if l["level"] not in ("ERROR", "WARNING")]
    lines = (priority + context)[:max_lines]
    lines.sort(key=lambda l: l["ts"])
    return "\n".join(f"[{l['ts']}] {l['level']} {l['service']}: {l['msg']}" for l in lines)


def apply_gate_no_evidence(llm_output: dict) -> dict:
    """With zero evidence available, ANY "answer" is unsupported by
    definition -- there is nothing it could have validly cited. The gate
    doesn't need citation-matching here; the rule collapses to: answer
    without evidence is always an unsupported claim, forced to abstain."""
    llm_decision = llm_output.get("decision", "abstain")
    cited = llm_output.get("cited_evidence_ids", [])
    confidence = float(llm_output.get("confidence", 0.0))

    unsupported_claim = (llm_decision == "answer")  # no evidence existed to support it
    gate_decision = "abstain"  # with zero evidence, the gate can never grant "answer"
    gate_overrode = (llm_decision == "answer")

    return {
        "llm_decision": llm_decision,
        "gate_decision": gate_decision,
        "gate_overrode_llm": gate_overrode,
        "n_cited": len(cited),
        "evidence_coverage": 0.0,  # no evidence was available to cover
        "unsupported_claim": unsupported_claim,
        "confidence": confidence,
    }


def main():
    incidents = json.loads((RCA / "incidents" / "incidents.json").read_text(encoding="utf-8"))
    results = []

    for inc in incidents:
        prompt = NO_EVIDENCE_PROMPT_TEMPLATE.format(log_lines=format_log_lines(inc["log_lines"]))
        try:
            raw_response = generate(prompt)
            llm_output = extract_json(raw_response)
            parse_error = None
        except Exception as e:
            llm_output = {"diagnosis": "", "cited_evidence_ids": [], "decision": "abstain", "confidence": 0.0}
            parse_error = str(e)
            raw_response = ""

        gate_result = apply_gate_no_evidence(llm_output)
        result = {
            "case_id": inc["case_id"],
            "source": inc["source"],
            "ground_truth": inc["ground_truth"],
            "llm_raw_response": raw_response,
            "llm_output": llm_output,
            "gate": gate_result,
            "parse_error": parse_error,
        }
        results.append(result)
        print(f"[{inc['case_id']}] llm={gate_result['llm_decision']} gate={gate_result['gate_decision']} "
              f"(overrode={gate_result['gate_overrode_llm']}) conf={gate_result['confidence']:.2f}"
              + (f" PARSE_ERROR: {parse_error}" if parse_error else ""))

    out_path = RCA / "results" / "rca_results_no_retrieval.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n{len(results)} cases -> {out_path}")


if __name__ == "__main__":
    main()
