# AI-Powered DevOps Log Analysis

Hybrid log-intelligence pipeline: classical ML → LSTM/Transformer anomaly
detection → cross-domain/drift robustness testing → fault-injection testbed
→ retrieval-augmented, gated RCA with a local LLM.

**Results and findings**: see [`REPORT.md`](REPORT.md).
**Full dated build log**: see [`PROGRESS.md`](PROGRESS.md).

## Setup

Requires Python 3.11 (a real python.org install, not the Windows Store
stub — `pyvenv.cfg` should show `home = ...\Programs\Python\Python311`), a
CUDA-capable GPU for the neural model steps, Docker Desktop for Phase 4, and
[Ollama](https://ollama.com) with `llama3.1:8b` pulled for Phase 7.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
# GPU-enabled torch (verified on this project: RTX 4060, driver CUDA 13.4):
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe scripts/check_gpu.py   # confirm cuda available: True
```

## Reproduction, in phase order

### Phase 1 — data

```bash
# download HDFS_v1.zip, BGL.zip, Thunderbird.tar.gz from the Loghub Zenodo
# record (8196385) into data/raw/, then:
python scripts/mask_and_parse.py --dataset hdfs
python scripts/mask_and_parse.py --dataset bgl
python scripts/mask_and_parse.py --dataset thunderbird
python scripts/spot_check_pii.py --datasets hdfs bgl thunderbird
python scripts/build_sequences.py --dataset hdfs
python scripts/build_sequences.py --dataset bgl
python scripts/build_sequences.py --dataset thunderbird
python scripts/quality_checks.py
python scripts/freeze_split.py --dataset hdfs   # seed 1337, stratified 70/15/15
python scripts/freeze_split.py --dataset bgl
python scripts/freeze_split.py --dataset thunderbird
```

### Phase 2 — classical baselines

```bash
python scripts/build_features.py --dataset <hdfs|bgl|thunderbird>
python scripts/train_isolation_forest.py --dataset <...>   # seed 42
python scripts/train_random_forest.py --dataset <...>      # seed 42
python scripts/lock_baselines.py
```

### Phase 3 — LSTM, Transformer, ablation, cross-domain, drift

```bash
bash scripts/run_all_lstm.sh          # 3 datasets x seeds 42/123/777
bash scripts/run_all_transformer.sh   # same
bash scripts/run_all_eval.sh          # scores every checkpoint against test
bash scripts/run_ablation.sh          # template-only vs +semantic, seed 42

bash scripts/run_all_cross_domain_build.sh   # all 6 directed pairs
python scripts/cross_domain_evaluate.py --source <a> --target <b> --seed 42

python scripts/inject_drift.py --dataset <...>              # seed 2026
python scripts/evaluate_drift.py --model lstm --dataset <...> --seed 42

python scripts/aggregate_stats.py     # mean ± 95% CI, all seeded models
```

The shell drivers deliberately don't use `set -e` — the local torch/CUDA
build exits non-zero on teardown even after a fully successful run; checked
and confirmed harmless by validating output files after each run.

### Phase 4 — fault-injection testbed

```bash
cd testbed
docker compose up -d --build
python scripts/run_scenarios.py --duration-s 280 --request-interval-s 1.0
python scripts/verify_pipeline.py
```

### Phase 5 — retrieval

```bash
python scripts/build_template_embeddings.py --dataset <...>
python scripts/build_retrieval_index.py
python scripts/check_retrieval.py     # manual relevance sanity-check
```

### Phase 7 — RCA + reliability gate

Requires `ollama serve` running with `llama3.1:8b` pulled.

```bash
python scripts/build_rca_incidents.py   # 8 testbed + 12 real detected anomalies
python scripts/run_rca.py               # retrieval -> LLM -> gate, all 20 cases
```

### Phases 6 & 8 — website

```bash
.venv/Scripts/python.exe -m uvicorn webapp.app:app --port 8205
# http://localhost:8205 (dashboard), http://localhost:8205/demo (RCA walkthrough)
```

## Seeds used (for exact reproduction)

| Purpose | Seed |
|---|---|
| Train/val/test split (`freeze_split.py`) | 1337 |
| Classical baselines (IF/RF) | 42 |
| Neural models (LSTM/Transformer/ablation) | 42, 123, 777 |
| Drift injection (`inject_drift.py`) | 2026 |
| Cross-domain adapt/val/test split (`cross_domain_evaluate.py`) | 2026 |
| PII spot-check sampling | 42 |

## Project layout

```
data/            raw/masked/parsed/sequence/split data (gitignored where large)
models/          trained checkpoints
results/         every evaluation's JSON + the aggregate stats
testbed/         Docker Compose fault-injection microservices
retrieval/       FAISS index + document corpus
rca/             RCA incident corpus + results
webapp/          FastAPI dashboard + demo site
sensitivity_check/  the HDFS window-size robustness check (see REPORT.md §9)
scripts/         everything above, one script per step
src/             shared library code (masking, models, eval utilities)
```
