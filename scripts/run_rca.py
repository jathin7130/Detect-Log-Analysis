"""Phase 7 — run RCA across all incidents through the full pipeline:
retrieve evidence (Phase 5's FAISS index) -> local LLM diagnosis (Ollama,
llama3.1:8b) -> reliability gate.

The gate is NOT just trusting the LLM's own self-reported decision. It
independently checks:
  1. does the LLM cite at least one evidence id that actually exists in
     what it was given (an "answer" with zero valid citations is an
     unsupported claim by definition, regardless of what the model says
     about its own confidence)
  2. is the self-reported confidence above threshold
Either check failing forces ABSTAIN, overriding the LLM if necessary --
that's the point of a gate sitting outside the model being gated.
"""
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from llm_client import generate, extract_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RETRIEVAL = ROOT / "retrieval"
RCA = ROOT / "rca"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K_EVIDENCE = 5
CONFIDENCE_THRESHOLD = 0.5

PROMPT_TEMPLATE = """You are a DevOps root-cause-analysis assistant. You diagnose the root cause of incidents using ONLY the evidence provided below. You must not invent causes that aren't supported by the evidence.

INCIDENT LOG LINES:
{log_lines}

RETRIEVED EVIDENCE (each has an id):
{evidence_block}

Instructions:
- If the evidence clearly supports a specific root cause, give that diagnosis and cite the evidence ids that support it.
- If the evidence does NOT clearly support a specific root cause, you MUST set "decision" to "abstain" rather than guess.
- Respond with ONLY a JSON object, no other text, in this exact shape:
{{"diagnosis": "<one or two sentences>", "cited_evidence_ids": ["<id1>", "<id2>"], "decision": "answer" or "abstain", "confidence": <float 0.0-1.0>}}
"""


def format_log_lines(log_lines, max_lines=15):
    """Chronological truncation alone can silently cut the actual
    ERROR/WARNING lines out of a busy window (background traffic fills
    the first N lines before anything goes wrong) -- always keep those,
    then fill the rest of the budget with context lines.
    """
    priority = [l for l in log_lines if l["level"] in ("ERROR", "WARNING")]
    context = [l for l in log_lines if l["level"] not in ("ERROR", "WARNING")]
    lines = (priority + context)[:max_lines]
    lines.sort(key=lambda l: l["ts"])
    return "\n".join(f"[{l['ts']}] {l['level']} {l['service']}: {l['msg']}" for l in lines)


def format_evidence(evidence):
    return "\n".join(f"[{e['doc_id']}] (score={e['score']:.3f}) {e['text']}" for e in evidence)


def retrieve(index, docs, model, query_text, k=TOP_K_EVIDENCE):
    qvec = model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    scores, idx = index.search(qvec, k=k)
    results = []
    for score, doc_idx in zip(scores[0], idx[0]):
        d = docs[doc_idx]
        results.append({"doc_id": d["doc_id"], "score": float(score), "text": d["text"], "source": d["source"]})
    return results


def _normalize_citation(c: str) -> str:
    """The prompt displays evidence as "(id=X, score=Y)" and the model
    sometimes echoes the "id=" label back as part of the citation string
    instead of just the bare id -- strip it rather than let a cosmetic
    formatting choice get treated as an unsupported claim."""
    c = c.strip()
    if c.lower().startswith("id="):
        c = c[3:]
    return c.strip(" '\"")


def apply_gate(llm_output: dict, evidence: list) -> dict:
    valid_ids = {e["doc_id"] for e in evidence}
    cited = llm_output.get("cited_evidence_ids", [])
    valid_cited = [c for c in cited if _normalize_citation(c) in valid_ids]
    evidence_coverage = len(valid_cited) / max(len(cited), 1) if cited else 0.0

    llm_decision = llm_output.get("decision", "abstain")
    confidence = float(llm_output.get("confidence", 0.0))

    unsupported_claim = (llm_decision == "answer" and len(valid_cited) == 0)
    low_confidence = confidence < CONFIDENCE_THRESHOLD

    gate_decision = "answer" if (llm_decision == "answer" and not unsupported_claim and not low_confidence) else "abstain"
    gate_overrode = (llm_decision == "answer" and gate_decision == "abstain")

    return {
        "llm_decision": llm_decision,
        "gate_decision": gate_decision,
        "gate_overrode_llm": gate_overrode,
        "n_cited": len(cited),
        "n_valid_cited": len(valid_cited),
        "evidence_coverage": evidence_coverage,
        "unsupported_claim": unsupported_claim,
        "confidence": confidence,
    }


def main():
    index = faiss.read_index(str(RETRIEVAL / "index.faiss"))
    docs = json.loads((RETRIEVAL / "documents.json").read_text(encoding="utf-8"))
    embed_model = SentenceTransformer(MODEL_NAME)

    incidents = json.loads((RCA / "incidents" / "incidents.json").read_text(encoding="utf-8"))
    results = []

    for inc in incidents:
        t0 = time.time()
        evidence = retrieve(index, docs, embed_model, inc["query_text"])
        prompt = PROMPT_TEMPLATE.format(
            log_lines=format_log_lines(inc["log_lines"]),
            evidence_block=format_evidence(evidence),
        )
        try:
            raw_response = generate(prompt)
            llm_output = extract_json(raw_response)
            parse_error = None
        except Exception as e:
            llm_output = {"diagnosis": "", "cited_evidence_ids": [], "decision": "abstain", "confidence": 0.0}
            parse_error = str(e)
            raw_response = ""

        gate_result = apply_gate(llm_output, evidence)
        elapsed = time.time() - t0

        result = {
            "case_id": inc["case_id"],
            "source": inc["source"],
            "ground_truth": inc["ground_truth"],
            "evidence": evidence,
            "llm_raw_response": raw_response,
            "llm_output": llm_output,
            "gate": gate_result,
            "parse_error": parse_error,
            "elapsed_s": elapsed,
        }
        results.append(result)

        print(f"[{inc['case_id']}] llm={gate_result['llm_decision']} gate={gate_result['gate_decision']} "
              f"(overrode={gate_result['gate_overrode_llm']}) coverage={gate_result['evidence_coverage']:.2f} "
              f"conf={gate_result['confidence']:.2f} ({elapsed:.1f}s)" + (f" PARSE_ERROR: {parse_error}" if parse_error else ""))

    out_path = RCA / "results" / "rca_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n{len(results)} cases -> {out_path}")


if __name__ == "__main__":
    main()
