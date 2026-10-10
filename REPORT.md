# AI-Powered DevOps Log Analysis — Results Report

**Status: FINISHED** (2026-10-08). All 9 phases complete; this report is the
final write-up of results and findings.

Implementation and experimental results for the hybrid log-intelligence pipeline
described in the CA2 proposal (classical ML → sequence/transformer anomaly
detection → retrieval-grounded, gated RCA). This report documents what was
built, what was found, every scoping decision made along the way and why, and
how to reproduce it. `PROGRESS.md` is the running build log this report was
distilled from — consult it for granular, dated detail on any step.

**Datasets**: HDFS, BGL, Thunderbird (Loghub). **Models**: Isolation Forest,
Random Forest, DeepLog-style LSTM, compact Transformer, LSTM+semantic-embedding
ablation. **RCA**: local `llama3.1:8b` (Ollama) + FAISS retrieval + an
independently-checking reliability gate.

---

## 1. Data pipeline (Phases 0–1)

Environment verified against a real python.org install (not the Windows Store
stub) with GPU-enabled PyTorch confirmed on an RTX 4060. All three datasets
downloaded in full (HDFS 11.18M lines, BGL 4.75M lines, Thunderbird capped at
10M lines per the plan), masked for PII (IPs, hostnames, user@host pairs,
file paths, credential-like strings), parsed into event templates with
Drain3, built into labeled sequences, and split 70/15/15 (stratified by
label, seed 1337, frozen before any model touched it).

| Dataset | Templates | Sequences | Anomaly rate |
|---|---|---|---|
| HDFS | 44 | 575,061 (per-block) | 2.93% |
| BGL | 1,819 | 47,135 (100-event windows) | 10.24% |
| Thunderbird | 1,467 | 99,808 (100-event windows) | 38.36% |

**PII spot-check caught two real gaps** before they became silent data-quality
problems: HDFS block IDs were leaking into Drain3 templates as literals
(fixed — masked to `<BLK>` before template mining, kept as a real value in
its own column for session grouping), and Thunderbird's `component` field
carried unmasked install paths (fixed — `<PATH>`). Final re-check: 0/20,000
sampled rows flagged across all three datasets.

## 2. Classical baselines (Phase 2)

TF-IDF/count features over template-id sequences, fit on train split only.

| Dataset | Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| HDFS | Isolation Forest | 0.543 | 0.918 | 0.682 | 0.709 |
| HDFS | Random Forest | 0.999 | 0.999 | 0.999 | 1.000 |
| BGL | Isolation Forest | 0.197 | 0.934 | 0.325 | 0.252 |
| BGL | Random Forest | 0.986 | 0.978 | 0.982 | 0.997 |
| Thunderbird | Isolation Forest | 0.397 | 0.988 | 0.566 | 0.288 |
| Thunderbird | Random Forest | 1.000 | 0.999 | 0.999 | 1.000 |

Random Forest's near-perfect scores are consistent with the literature on
these benchmarks (specific templates are strong anomaly indicators), not a
data leak — the split stayed frozen throughout.

## 3. Sequence models: LSTM + Transformer (Phase 3)

DeepLog-style next-event prediction (window=5, top-3 hit/miss scoring),
trained on normal sequences only, 3 seeds (42/123/777) per dataset per model.

| Dataset | Model | F1 (mean ± 95% CI, n=3) | PR-AUC |
|---|---|---|---|
| HDFS | LSTM | 0.265 ± 0.084 | 0.146 ± 0.067 |
| HDFS | Transformer | 0.271 ± 0.055 | 0.147 ± 0.037 |
| BGL | LSTM | 0.928 ± 0.014 | 0.971 ± 0.001 |
| BGL | Transformer | 0.926 ± 0.003 | 0.970 ± 0.003 |
| Thunderbird | LSTM | 0.823 ± 0.033 | 0.784 ± 0.041 |
| Thunderbird | Transformer | 0.815 ± 0.045 | 0.769 ± 0.085 |

**Findings:**
- **Transformer ≈ LSTM everywhere** (within ~0.01–0.03 F1) — the added
  attention mechanism doesn't buy meaningful improvement over the simpler
  recurrent model on any of these three datasets.
- **On HDFS, both neural models lose to the simple Isolation Forest
  baseline** (0.265–0.271 vs. 0.682). See §6 for the sensitivity-check
  follow-up on this.

### Ablation (template-only vs. template+semantic)

Single seed (42), 2 variants — the raw-text-only variant from the original
3-variant plan was dropped as a scoping decision (see §8).

| Dataset | Template-only | +Semantic embedding |
|---|---|---|
| HDFS | 0.300 | 0.275 |
| BGL | 0.932 | 0.927 |
| Thunderbird | 0.836 | 0.796 |

Adding a frozen semantic embedding of each template's text doesn't help and
mildly hurts, most clearly on Thunderbird — likely because it's a
deterministic function of the template id already, adding dimensionality
without new information.

### Cross-domain transfer (6/6 directed pairs)

Every pair shows **100% vocabulary disjunction** — HDFS, BGL and Thunderbird
share zero templates in any direction, confirmed at full scale (not just
sampled). A naive P/R/F1 comparison of before/after limited-target-adaptation
looked *identical* in every pair, which would read as "adaptation did
nothing" — except it's a metric artifact: with every window collapsed to the
same "unseen" input, threshold-based P/R/F1 can only express "flag everyone,"
regardless of whether the model's constant prediction is "always anomalous"
(before) or "always normal" (after). Adding a raw, threshold-independent
`mean_raw_score` diagnostic exposed the real signal: it swings from ~1.0 to
~0.0 in every pair, confirming the model's predictions genuinely flipped —
the gate just couldn't express a trivial classifier's output as a useful F1.
Caught during Phase 3 directly via the results JSON showing identical before/
after numbers despite training loss crashing from ~0.75 to ~0.001.

### Drift robustness

Benign drift (token renaming, new-template insertion, frequency tweaks) on
normal test sequences, measured against the LSTM's already-committed
operating threshold.

| Dataset | FPR (clean) | FPR (drifted) | Relative increase |
|---|---|---|---|
| HDFS | 2.6% | 99.9% | **37x** (catastrophic) |
| BGL | 0.4% | 5.4% | 11.7x (severe) |
| Thunderbird | 17.3% | 66.0% | 2.8x (moderate) |

HDFS essentially flags everything once its (already small, 44-template)
vocabulary drifts even slightly — a dramatic, concrete demonstration of the
enterprise-readiness concern the proposal's literature review raises about
LogRobust/LogBERT-style brittleness.

## 4. Fault-injection testbed (Phase 4)

Four Flask microservices (`gateway → orders → {inventory, payments}`) in
Docker Compose, logging realistic text lines through the same masking +
Drain3 pipeline as the main datasets (no separate ingestion path). Four fault
types (`db_failure`, `dependency_timeout`, `resource_exhaustion`,
`pod_restart`), each independently verified — `pod_restart` confirmed via an
actual container restart + uptime check, not just a log line. An 8-incident
scenario was run (280s, continuous background traffic: 245 requests, 34
errors) with ground truth recorded automatically. All 8 incidents show a
clear ERROR/WARNING line-count spike at their exact injection timestamp when
run through the parsing pipeline (counts: 32, 9, 19, 3, 30, 4, 3, 21).

## 5. Retrieval layer (Phase 5)

`all-MiniLM-L6-v2` embeddings over 3,360 documents (every HDFS/BGL/
Thunderbird template + every testbed template + the 8 testbed incident
descriptions), FAISS `IndexFlatIP` (cosine via normalized vectors). Sanity-
checked with 6 made-up incident queries — every top-1 match was semantically
on point (e.g. "memory usage is very high" → testbed's exact
`resource_exhaustion` template at 0.714 cosine similarity).

## 6. RCA + reliability gate (Phase 7)

Local `llama3.1:8b` via Ollama — the proposal's ethics section commits to a
local open-source model rather than an external provider, and this is the
only LLM path used anywhere in the project.

**Pipeline**: incident → FAISS retrieval (top-5) → LLM diagnosis (structured
JSON: diagnosis, cited evidence ids, decision, confidence) → **gate**, which
independently re-checks that cited ids actually exist in what the model was
given and that confidence clears a threshold, **overriding the LLM to
abstain if either check fails** regardless of what the model claims about
itself.

**20 cases**: 8 real testbed faults (known service+fault_type ground truth)
+ 12 genuine LSTM-detected anomalies from HDFS/BGL/Thunderbird (true
positives, not random windows).

| Metric | Value |
|---|---|
| LLM self-reported "answer" rate | 19/20 (95%) |
| Gate "answer" rate | 18/20 (90%) |
| Gate override rate | 1/20 (5%) |
| Avg evidence coverage | 0.95 |
| Avg confidence | 0.76 |

**Manual hand-check, all 20 cases**: 16 clearly correct, 1 correct-but-
conservatively-gated (confidence dipped below threshold on rerun), 1 honest
self-abstain (LLM itself said the evidence didn't support a diagnosis), 2
flagged. The 2 flagged cases are **not hallucination** — both are
`pod_restart` incidents where the LLM correctly named the affected service
but described the symptom an *upstream caller* sees (a connection error)
rather than the actual mechanism (the process crashed and restarted). That's
a real, explainable gap: the crashing service's own log lines ("unhandled
exception... terminating" / "service started") weren't in the evidence shown
to the LLM for those cases, which only had the observer services' logs.

**Honest methodological caveat**: the retrieval corpus contains literal
descriptions of all 8 testbed incidents (built from the same ground truth in
Phase 5), so those 8 cases mainly test retrieval+gate *mechanics* — a
realistic behavior for a production system's incident history, but not a
test of causal inference under ambiguity. The 12 real-dataset cases have no
equivalent answer-sheet in the corpus and are the fairer test of genuine
reasoning; all 12 produced plausible, template-grounded diagnoses.

Two real bugs were caught and fixed during this phase: a log-window
truncation that could silently drop the actual ERROR lines from a busy
window, and an over-strict citation-matching check that caused 3 false gate
overrides when the LLM echoed the prompt's `"id=X"` display format back
verbatim — both fixed and the affected cases rerun to confirm the underlying
diagnoses were correct all along.

## 7. Website (Phases 6 & 8)

FastAPI + plain HTML/CSS/JS (no frontend framework), dark DevOps/SRE
monitoring aesthetic. Dashboard and demo are both wired to real results —
nothing on either page is hardcoded or mocked. The demo page lets you browse
all 20 real RCA cases, including the one gate-override case (clearly banner-
flagged) and the one honest self-abstain. Verified end-to-end in-browser on
desktop and mobile.

## 8. Scoping decisions (what changed from the original plan, and why)

| Original | Changed to | Reason |
|---|---|---|
| 5 seeds per neural run | 3 seeds | Time budget; 3 still gives a meaningful mean±CI without the long tail of diminishing-returns reruns |
| 3 ablation variants (template / raw-text-only / combined) | 2 variants (template / combined) | Raw-text-only dropped as a deliberate scope cut; template-only vs. combined is the comparison that actually answers "does adding semantic signal help" |
| Fixed window_size=5/topk=3 for all LSTM evaluation | Unchanged as the headline config, but a window_size=10/topk=5 **sensitivity check** was run on HDFS specifically after the original result looked surprising | See §9 — the check mattered, the headline numbers were kept unchanged on principle |
| Early stopping patience=3, min_delta=1e-4 | patience=2, min_delta=5e-4 | The original setting let BGL/Thunderbird runs wander for many epochs on sub-0.001 "improvements"; tightened after measuring real per-epoch cost mid-run |

## 9. HDFS sensitivity check — a finding that needed a second look

The headline result (LSTM/Transformer losing to Isolation Forest on HDFS)
was flagged for a robustness check before being written up as clean: HDFS
sequences average only 19.4 events vs. BGL/Thunderbird's fixed 100, so the
same window_size=5 might be losing more relative context there specifically.

A sensitivity run (window_size=10, topk=5, HDFS only, seed 42) found:

| Config | F1 | PR-AUC | FA/10k seq |
|---|---|---|---|
| Main config (window=5, topk=3) | 0.300 | 0.177 | 255.2 |
| Sensitivity check (window=10, topk=5) | **0.545** | **0.373** | **9.3** |

F1 nearly doubles, false alarms drop ~27x. The gap to Isolation Forest
(0.682) narrows substantially but doesn't close — so the qualitative
conclusion holds ("LSTM is not clearly better than the simplest baseline on
HDFS"), but "loses badly" is not an accurate characterization. **The
headline numbers were kept at the main-config value** (changing
hyperparameters per-dataset after seeing results would bias the cross-
dataset comparison) — this is documented as a limitation/sensitivity note,
not folded silently into the main table. Full detail in
`sensitivity_check/README.md`.

## 10. Known limitations

- **HDFS detector performance is hyperparameter-sensitive** (§9) — the main
  config may understate what these models can do on HDFS specifically.
- **Testbed RCA cases have an answer-sheet in the retrieval corpus** (§6) —
  mainly tests pipeline mechanics for those 8, not pure inference.
- **`pod_restart` root-cause identification is incomplete** when only
  observer-service logs are available (§6) — a real gap, not a random error.
- **Random Forest's near-perfect scores** likely reflect how separable these
  specific benchmark templates are, not a general claim about supervised
  detection in production (where label availability itself is the real
  constraint, as the proposal's own Research Niche section argues).
- **RCA evaluation is 20 cases**, by design (time-boxed hand-check, not a
  claim of statistical power) — the proposal's own evaluation plan treats
  this as a qualitative reliability-gate demonstration, not a benchmark.

## 11. Reproducing this

See `README.md` for environment setup and the exact script sequence per
phase. Every seed used is listed there. `PROGRESS.md` has the full dated
build log if any step's exact command or timing is needed.
