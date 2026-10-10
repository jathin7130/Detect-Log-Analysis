# DevOps Log Analysis — Progress Checklist

**STATUS: FINISHED (2026-10-08)** — all 9 phases complete. Final write-up: `REPORT.md`.

**THESIS: DONE (2026-10-09)** — `THESIS.md` is the integrated academic document: original CA2 proposal (Sections 1–3) merged with new Results (4), Discussion (5), and Conclusion/Limitations/Future Work (6), plus an updated References list. Three gaps identified in review were closed first and are reflected in the document: paired bootstrap significance tests (`scripts/paired_significance_test.py`, `results/paired_tests/`), a consolidated cost/latency table (`scripts/build_cost_latency_table.py`, `results/cost_latency_table.md`), and a with/without-retrieval RCA ablation (`scripts/run_rca_no_retrieval.py`, `rca/results/retrieval_ablation_summary.md`). Formatting and proofreading are explicitly out of scope here (handled separately).

Tracks the phased plan (Phase 0–9). Check items off as they're completed.

## Phase 0 — Setup (Day 0)

- [x] Environment setup — venv created at `.venv` (Python 3.11.9, python.org install)
- [x] Check `pyvenv.cfg` points to real python.org install — confirmed: `home = ...\AppData\Local\Programs\Python\Python311`
- [x] Verify GPU is used by PyTorch — confirmed: `torch 2.11.0+cu128`, `cuda available: True`, device = NVIDIA GeForce RTX 4060 Laptop GPU
- [x] Set up tracking checklist — this file
- [x] CPU-only deps installed (`requirements.txt`): pandas, numpy, scikit-learn, drain3, sentence-transformers, faiss-cpu, matplotlib, pytest. `torch`/`transformers` came in as sentence-transformers deps (CPU wheel, no CUDA index used — didn't touch the GPU).

## Phase 1 — Full-scale data prep

- [x] Thunderbird download — full archive fetched (1.9GB, `data/raw/Thunderbird.tar.gz`)
- [x] Mask + parse full HDFS — 11,175,629/11,175,629 lines, 44 templates, ~7 min (`data/masked/hdfs/`)
- [x] Mask + parse full BGL — 4,713,493/4,747,963 lines (34K lines didn't match the expected field layout — likely malformed/rare-variant lines, non-blocking), 1,819 templates, ~5 min (`data/masked/bgl/`)
- [x] Mask + parse Thunderbird (10M cap) — 9,980,758/10,000,000 lines, 1,467 templates, ~13 min (`data/masked/thunderbird/`)
- [x] Spot-check masked output for PII — all 3 datasets clean after two fixes found during spot-checking:
      (1) HDFS block IDs were leaking into Drain3 templates as literals → now masked to `<BLK>` before template mining, kept as a real value in its own column for session grouping.
      (2) Thunderbird's `component` field carried install paths (e.g. `/apps/x86_64/.../gmetad[1684]`) that content-masking didn't touch since it's a separate column → patched to `<PATH>[1684]`.
      Final re-check: 0/20,000 sampled rows flagged on all 3 datasets across IP/email/credential/path/hex-token checks (16/20k hex-token hits on BGL are benign hardware part-number strings, reviewed, not PII).
- [x] Build sequences — HDFS: session-based by block_id, labeled from the official `anomaly_label.csv` (575,061 sequences). BGL/Thunderbird: no natural session id, so fixed 100-event non-overlapping windows, labeled anomalous if any line in the window carries a non-"-" alert label (BGL: 47,135 sequences; Thunderbird: 99,808 sequences).
- [x] Run quality checks — `results/quality_report.md`:
      - HDFS: 100% template coverage, 2.928% anomaly rate, seq length 2–298 (mean 19.4)
      - BGL: 100% template coverage, 10.237% anomaly rate, seq length 93–100 (mean 100.0)
      - Thunderbird: 100% template coverage, 38.359% anomaly rate, seq length 58–100 (mean 100.0)
- [x] Freeze 70/15/15 split — stratified by label, seed 1337, one manifest per dataset in `data/splits/`:
      - HDFS: train 402,542 / val 86,258 / test 86,261
      - BGL: train 32,993 / val 7,069 / test 7,073
      - Thunderbird: train 69,865 / val 14,970 / test 14,973

## Phase 2 — Classical baselines

- [x] Build TF-IDF / template-count features — vectorizers fit on train split only (no val/test leakage); vocab sizes: HDFS 44, BGL 1,470, Thunderbird 1,402 (`data/features/`)
- [x] Train Isolation Forest (×3 datasets) — unsupervised, contamination set from train anomaly rate, threshold picked on val (max F1): HDFS F1=0.682, BGL F1=0.325, Thunderbird F1=0.566. BGL/Thunderbird's high false-alarm rates reflect how hard unsupervised detection is on their bursty alert patterns in a 100-event window.
- [x] Train Random Forest (×3 datasets) — supervised, class_weight="balanced", threshold picked on val (max F1): HDFS F1=0.999, BGL F1=0.982, Thunderbird F1=0.999. Near-perfect scores are expected/consistent with the literature — specific event templates are strong anomaly indicators on these benchmarks, not a data leak (split is frozen, sequence-level, no train/test overlap).
- [x] Evaluate, lock baseline numbers — `results/baselines_locked.json` / `.md`, the reference numbers Phase 3's LSTM/transformer results get compared against:

  | Dataset | Model | Precision | Recall | F1 | PR-AUC | FA/10k seq |
  |---|---|---|---|---|---|---|
  | hdfs | isolation_forest | 0.543 | 0.918 | 0.682 | 0.709 | 226.1 |
  | hdfs | random_forest | 0.999 | 0.999 | 0.999 | 1.000 | 0.3 |
  | bgl | isolation_forest | 0.197 | 0.934 | 0.325 | 0.252 | 3905.0 |
  | bgl | random_forest | 0.986 | 0.978 | 0.982 | 0.997 | 14.1 |
  | thunderbird | isolation_forest | 0.397 | 0.988 | 0.566 | 0.288 | 5755.0 |
  | thunderbird | random_forest | 1.000 | 0.999 | 0.999 | 1.000 | 0.7 |

## Phase 3 — LSTM + Transformer training

- [x] Implement DeepLog-style LSTM — `src/lstm_model.py`
- [x] Add mixed precision + early-stopping hooks — `scripts/train_lstm.py`, reused by transformer/ablation trainers
- [x] Smoke-test LSTM — verified GPU+AMP active on all 3 datasets
- [x] Full LSTM runs (3 datasets × 3 seeds) — all 9 done. F1: HDFS 0.23–0.30 (underperforms its own Isolation Forest baseline), BGL 0.92–0.93 (rivals Random Forest), Thunderbird 0.81–0.84 (beats Isolation Forest, below Random Forest)
- [x] Implement compact transformer — `src/transformer_model.py`, LogBERT/LogFormer-inspired, mean-pooled encoder
- [x] Smoke-test transformer — verified on all 3 datasets
- [x] Full transformer runs — all 9 done, ~2h53min. Essentially tied with LSTM on all 3 datasets (within ~0.01–0.03 F1) — extra complexity doesn't buy meaningful improvement here
- [x] Write drift-injection script — `scripts/inject_drift.py` (token substitution, new-template insertion, frequency tweak on normal test sequences)
- [x] Cross-domain transfer runs — all 6 directed pairs done. **Finding: 100% vocabulary disjunction** between all 3 datasets (zero shared templates in any direction) — zero-shot transfer is fundamentally impossible at the template-id level. Confirmed via `mean_raw_score` diagnostic after catching a threshold-selection metric-masking issue (P/R/F1 looked identical before/after adaptation despite the model's predictions genuinely flipping from "always anomalous" to "always normal")
- [x] Ablation variants (2: template-only, combined) — template-only wins on all 3 datasets; combined (+semantic template embedding) is tied-to-worse, most notably on Thunderbird (0.836 → 0.796)
- [x] Drift evaluation — `scripts/evaluate_drift.py`. Caught + fixed a real GPU crash: drift-injected ids exceeded the model's embedding table size (out-of-bounds CUDA assert), fixed by clipping to the "unseen" index. Benign drift causes a substantial false-alarm spike on all 3 datasets: HDFS 2.6%→99.9% FPR (37x, catastrophic), BGL 0.4%→5.4% (11.7x), Thunderbird 17.3%→66.0% (2.8x)
- [x] Aggregate stats (mean ± 95% CI) — `results/phase3_aggregate_stats.md`. Caught + fixed a glob bug (`seed*_eval.json` was also matching `seed42_drift_eval.json`, a different file, polluting the mean/CI with a phantom 4th "seed")
- [x] HDFS sensitivity check (post-hoc robustness check) — `sensitivity_check/README.md`. Tested window_size=10/topk=5 vs the main-config window_size=5/topk=3 on HDFS LSTM seed42: F1 nearly doubles (0.300→0.545), FA/10k drops ~27x (255→9.3). **Verdict**: the original "LSTM loses badly to Isolation Forest on HDFS" finding was partly a hyperparameter artifact — the gap narrows substantially (0.545 vs IF's 0.682) though doesn't fully close. Main-config numbers kept as the headline result (changing hyperparameters per-dataset after seeing results would bias the comparison); this is documented as a limitation/sensitivity note for the write-up instead.

## Phase 4 — Fault-injection testbed

- [x] Stand up Docker Compose — `testbed/docker-compose.yml`, 4 containers (gateway/orders/inventory/payments), shared Dockerfile with build-arg service selection
- [x] Define 4 microservices with logging — Flask apps, text-line logs (not JSON) so they flow through the same Drain3 pipeline as HDFS/BGL/Thunderbird; realistic dependency chain gateway→orders→{inventory,payments}
- [x] Wire fault hooks — `POST /_fault {type, duration_s}` per service; 4 fault types (db_failure, dependency_timeout, resource_exhaustion, pod_restart), all individually verified (pod_restart confirmed via actual container restart + uptime check)
- [x] Format testbed logs for pipeline — reused `src/mask.py` + Drain3 directly, no new ingestion path needed
- [x] Run fault scenarios, record ground truth — 280s run, continuous background traffic (245 requests, 34 errors), 8 scheduled incidents across all 4 fault types and all 4 services, ground truth at `testbed/incidents/ground_truth.json`
- [x] Test sample through parsing pipeline — `testbed/scripts/verify_pipeline.py`: 1,675 lines parsed into 22 templates; **all 8 injected incidents show a clear ERROR/WARNING spike at their exact injection timestamp** (counts: 32, 9, 19, 3, 30, 4, 3, 21 nearby lines), confirming the corpus is usable for Phase 7's RCA evaluation

## Phase 5 — Retrieval layer

- [x] Pick embedding model — `all-MiniLM-L6-v2` (same model as the Phase 3 ablation's template embeddings — one consistent semantic space)
- [x] Embed historical templates/incidents — 3,360 docs: all HDFS/BGL/Thunderbird templates + testbed templates + the 8 testbed ground-truth incidents (`retrieval/documents.json`)
- [x] Build FAISS index — `IndexFlatIP` over normalized vectors (cosine sim), `retrieval/index.faiss`
- [x] Manual relevance sanity-check — `scripts/check_retrieval.py`, 6 made-up incident queries, every top-1 match is semantically on point (e.g. "memory usage is very high" → testbed's exact `resource_exhaustion` template at 0.714 cosine sim; "hardware component reported a parity error" → all top-3 are genuine BGL parity-error templates)

## Phase 6 — Website scaffolding

- [x] Pick stack, scaffold API — FastAPI + plain HTML/CSS/JS (no frontend framework), `webapp/app.py`. Launch config added to the top-level `.claude/launch.json` as `devops-log-webapp` (port 8205)
- [x] Dashboard shell — `webapp/templates/dashboard.html`. **Not mocked** — wired directly to real Phase 2/3/4 results (detector F1 by dataset, drift robustness, cross-domain transfer matrix, testbed incidents) since the data already existed; no reason to fake it
- [x] Demo shell with canned example — `webapp/templates/demo.html`, a real recorded testbed incident (db_failure/inventory) with real log lines and real FAISS retrieval evidence; only the LLM diagnosis/reliability-gate step is mocked (clearly badged), pending Phase 7
- [x] Styling both views into one site — `webapp/static/css/style.css`, dark DevOps/SRE monitoring aesthetic (monospace, status-color palette, card grid), verified responsive on desktop and mobile
- [x] API skeleton with stub endpoints — `/api/detector_metrics`, `/api/cross_domain`, `/api/drift`, `/api/testbed_incidents` (real data), `/api/demo_incident` (stub, swaps to real once Phase 7 exists)
- [x] **Post-launch addition (2026-10-10)**: `/phases` page — `webapp/templates/phases.html`, a detailed walkthrough of all 10 phases (what/why/key steps/findings per phase, including Phase 3's 6 sub-investigations), added at the user's request for more explanatory depth on the site. Hit and fixed a real bug while verifying it: the browser's parsed stylesheet had stopped exactly at the end of the pre-edit CSS file (`.footer-note`), silently dropping every new `.phase-*` rule appended after it — a stale cached stylesheet object, not a syntax error. Fixed by cache-busting the stylesheet link (`style.css?v=2`) across all 3 pages; confirmed fixed by checking `document.styleSheets[0].cssRules.length` before/after.

## Phase 7 — RCA + reliability gate

- [x] Feed real incidents + evidence to LLM — local `llama3.1:8b` via Ollama (ethics-section commitment: no external provider). 20 incidents: 8 real testbed faults (known service+fault_type) + 12 genuine LSTM-detected anomalies from HDFS/BGL/Thunderbird. `scripts/build_rca_incidents.py`, `scripts/run_rca.py`, `src/llm_client.py`
- [x] Implement the gate — independently re-checks the LLM's own citations actually exist in the evidence it was given, and enforces a confidence threshold; overrides the LLM to abstain if either check fails, regardless of what the LLM claims about itself. `apply_gate()` in `scripts/run_rca.py`
- [x] Run RCA across all incidents — all 20 cases processed, ~4s/case
- [x] Compute evidence coverage / unsupported-claim rate — avg coverage 0.95, avg confidence 0.76, gate override rate 1/20 (5%)
- [x] Manual hand-check 15–20 cases — all 20 reviewed. **16/20 clearly correct, 1 correct-but-conservatively-gated (confidence dipped below threshold), 1 honest self-abstain, 2 flagged.** The 2 flagged cases are both `pod_restart` incidents where the LLM correctly named the service but described the symptom an upstream caller sees (connection error) rather than the actual mechanism (process crash) — a real, explainable limitation (the crashing service's own log lines weren't in the evidence shown), not hallucination.
- Caught + fixed 2 real bugs during this phase: (1) log-window truncation could silently drop the actual ERROR lines from a busy window — fixed to prioritize ERROR/WARNING lines; (2) the gate's citation matching was too strict (exact string match), causing 3 false overrides when the LLM echoed the prompt's `"id=X"` display format back verbatim instead of the bare id — fixed with format-robust normalization, verified by rerunning and confirming those 3 cases' diagnoses were correct all along.
- Noted limitation for the write-up: the retrieval corpus contains literal descriptions of all 8 testbed incidents (built from the same ground truth in Phase 5), so those 8 cases mainly test retrieval+gate *mechanics* rather than pure causal inference — valid behavior for a real system's incident history, but the 12 real-dataset cases (no answer-sheet in the corpus) are the fairer test of genuine reasoning.

## Phase 8 — Wire website to real results

- [x] Swap mock dashboard data for real metrics — was already real since Phase 6; added a real RCA summary card (`/api/rca_summary`: 16/20 hand-check, override count, abstain count)
- [x] Point demo at real trained backend — `/api/demo_incident` and `/api/demo_cases` now read directly from `rca/results/rca_results.json` + `rca/incidents/incidents.json`; the old hardcoded mock response is gone entirely
- [x] Curate default abstain/answer examples — demo page opens on `testbed_5` (clean answer) by default, with a dropdown over all 20 real cases and a hint pointing at `bgl_bgl_5057` (the genuine self-abstain); the one gate-override case (`testbed_2`) is flagged directly in the dropdown
- [x] Full end-to-end UI test — verified in-browser (desktop + mobile): dashboard RCA card renders real numbers, demo page correctly renders all 3 gate states (plain answer, gate-overrode-LLM with the override banner, honest self-abstain), case switching works, no console errors
- [x] Publish — running locally via the project's launch config (`devops-log-webapp`, port 8205); no external hosting needed for this deliverable

## Phase 9 — Write-up & QA

- [x] Document results + scoping changes — `REPORT.md`: full narrative across all 8 phases, every scoping decision explicitly tabulated with rationale, all bugs-caught-and-fixed documented as part of the quality story, known limitations section
- [x] Clean up reproducibility details — `README.md` with exact script sequence per phase; audited `requirements.txt` against actual imports used (added `scipy`, which was imported directly but only transitively declared); all 6 fixed seeds used across the project documented in one table
- [x] Final end-to-end run-through — verified all key deliverables present (69 result JSONs, 31 model checkpoints, RCA/retrieval/testbed artifacts all in place); webapp re-verified live with zero errors (one console error investigated and confirmed to be stale browser history from an earlier-session bug already fixed, not a live issue — checked via fresh `curl` against all 4 main endpoints, all 200 OK)

**All 9 phases complete.**

---

### Notes
- `requirements.txt` — CPU-only deps, safe to install anytime.
- `requirements-gpu.txt` — torch/transformers; hold off running any CUDA check (`torch.cuda.is_available()`) until the GPU is free. Installing the package itself is harmless, but we're holding off for now per instruction.
