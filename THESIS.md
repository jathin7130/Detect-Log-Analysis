AI-Powered DevOps Log Analysis: A Hybrid
Framework for Anomaly Detection and
Evidence-Grounded Root Cause Analysis
=========================================

Yadlapalli Jathin Chowdary — 25202871
MSc in Artificial Intelligence - Research in Computing CA2
National College of Ireland

> **Note on this document**: Sections 1–3 are the original CA2 proposal.
> Sections 4–6 and the updated References are new, written after
> implementing and running the full pipeline. Formatting and proofreading
> are intentionally left for a final pass outside this document.

## Abstract

Modern DevOps generates enormous volumes of diverse log data, which
are essential for failure diagnosis but difficult to analyze manually in real
time. This proposal constructs and tests a hybrid AI log intelligence
framework that integrates classical machine learning and sequence and
transformer-based anomaly detection with an evidence-based language
model for root cause analysis. Using public log benchmarks and a managed microservice fault-injection test to evaluate the quality of detection,
generalization across systems, false alarms, diagnosis, latency, and cost,
this study aims to address a critical gap. The main unique feature is an
integrated corporate evaluation that is more important than developing
a new foundation model. Detected events would go through a retrieval
mechanism and a trust gate, which would require supporting logging and
allow the system to be silent if the trust is insufficient. We expect to
create a reproducible decision support framework within real DevOps environments to indicate when the semi-automated log analysis is accurate,
generalized, and trusted enough to be implemented.

*(Results summary, added post-implementation: the framework was built and
evaluated as proposed. The central finding is not "one model wins" but a
decision matrix — classical Random Forest dominates on raw accuracy across
all three benchmarks, but at a cost-per-accuracy-point that is 2–3 orders of
magnitude lower than the neural sequence models, which in turn only
clearly outperform a simple Isolation Forest baseline on two of three
datasets. Cross-domain transfer between unrelated log systems was found to
be completely non-viable at the template level (100% vocabulary
disjunction, 6/6 directed pairs), and all detectors were found to be
measurably brittle to even mild benign log drift. The evidence-grounded RCA
gate was shown, via a controlled with/without-retrieval ablation, to be the
mechanism that makes a trustworthy "answer" possible at all — without
retrieval, 100% of attempted diagnoses were unsupported by construction and
correctly suppressed.)*

## 1 Introduction

Software delivery has drastically improved thanks to the integration of container
orchestration, microservice architecture, continuous integration, and continuous
delivery. However, the improvements have come with trade-offs. There are now
more points of failure, and more telemetry is produced from standard operations.
Logs contain one of the richest records of operational behavior. This is due to
logs preserving the sequences of events, messages about errors, and contextual
information. These elements often go missing within aggregate metrics. The
CA1 study identified the core practical problem. The volume, velocity, and
variability of the formatting of modern logs have made static rules, keyword
searches, and manual inspections more brittle. The time and resources that
are lost due to service failures and the subsequent troubleshooting prolong the
importance of the problem. The aim of this research is to automate the detection
of anomalies and the subsequent root-cause analysis (RCA) within the DevOps
logs. The problem will still be measured experimentally.

Prior studies illustrate a trend evolving from parsing and recurrent sequence
models through semantic transformers and language-model-based analysis. Drain
enables quick online conversion of raw data into event templates (He et al.,
2017), while DeepLog learns normal sequences of events and helps identify atypical behaviors (Du et al., 2017). Subsequent works LogRobust and LogBERT
focused on the flexible template problem and used semantic or self-supervised
representations (Zhang et al., 2019; Guo, Yuan and Wu, 2021). However, achieving high benchmark F1-scores does not correlate with enterprise readiness. Real
DevOps use also depends on the false-positive rate, behavior under log drift,
system transfer, diagnostic usefulness, inference and compute costs, and the
avoidance of unsupported inferences.

The proposed approach is a hybrid log-intelligence pipeline. In this pipeline,
raw logs will be sanitized and parsed, while still keeping the original text. Classical and modern neural network-based detectors will be compared on the same
data splits. The best performing model will be used to create incident windows
of the anomalous event sequence and surrounding context. Next, a retrieval
mechanism will fetch the most pertinent past templates, descriptions of errors,
and experiment metadata. After this, a tuned language model will suggest and
rank potential root causes. A reliability gate will check if the diagnosis presented cites the retrieved data, if not, it will abstain and present a confidence
score. The main goal of this design is to provide a high-throughput detection
pipeline. This design prevents the use of a language model as a brute force
parser for each line of the logs.

**Research Question.** How effective can a hybrid AI pipeline that combines machine-learning/deep-learning anomaly detection and evidence-gathered
language-model reasoning be in terms of automating anomaly detection and
root-cause analysis across different DevOps logs? What are the technical factors that most affect the pipeline's readiness for deployment in real-world enterprises?

**Operational sub-questions.** The study will examine (1) which detector provides the best balance between precision and recall with regard to false
alarms; (2) how performance is affected by cross-system transfer and by controlled log drift; (3) whether retrieval-grounded language-model reasoning enhances top-k root-cause identification without leading to an increase in unsupported claims; and (4) how accuracy changes when latency, throughput and
computational cost are taken into account together with predictive performance.

The proposed contribution is not yet another independent anomaly detector; rather, it is a reproducible, end-to-end evaluation framework which links
together detection, cross-domain robustness and evidence-grounded root cause
analysis within a single experimental protocol. The study will also provide a
controlled fault-injection corpus with known root causes, include an explicit
mechanism for abstaining from diagnoses when uncertainty is high, and offer
practical advice on when to use classical machine learning, transformers or language models. Section 2 gives a critical review of current developments and
identifies the area of research that remains unexplored. Section 3 sets out the
methodology, the resources, the evaluation strategy, the ethical considerations
and the project schedule. **Section 4 reports the results of implementing this
methodology; Section 5 discusses them against the literature introduced below;
Section 6 concludes.**

## 2 Literature Review

### 2.1 Log parsing and sequence-based anomaly detection

Typically, automated log analysis starts with converting highly varied logs into
event templates with a constant structure. In particular, Drain uses an algorithm with a fixed-depth tree to perform online parsing, showing good results
in terms of both accuracy and runtime on multiple large data sets of log files
(He et al., 2017). DeepLog works at the level of sequences and trains the LSTM
network to recognize normal patterns of event transitions, detecting unusual
next events and enabling workflow-level analysis (Du et al., 2017). Those approaches are complementary but not competitive. First, Drain performs parsing
before modeling while DeepLog proves that temporal ordering is predictive itself. Moreover, a common feature of both approaches that matters for the
current research question is their development in the pre-microservices era and
without using modern LLM technology for operations management; therefore,
a combination of template generation and detection can fail in case of templates
evolution or parameter omission during parsing.

Empirical evidence, provided by Khan et al. (2024), confirms the importance
of this shortcoming. They compare different parsers and anomaly detectors and
show that parsing accuracy is not sufficient to serve as a criterion for the quality of anomaly detectors; it is also important whether generated templates can
be distinguished from each other. This complicates the assumption of "better
parser – better detector" and explains the necessity of ablation for the current experiment, which will involve comparison of templates, raw texts, and
combined representations.

### 2.2 Robustness to evolving and heterogeneous logs

Production logs are anything but static. With software updates rolling out,
new statements, variables, and sequences pop up, which means a closed-world
model might mistakenly flag harmless changes as incidents. LogRobust tackles
this challenge by focusing on event semantics and utilizing an attention-based
Bi-LSTM. This approach demonstrates that semantic representations can help
lessen the impact of previously unseen events (Zhang et al., 2019). On the other
hand, LogBERT takes a step further into self-supervised contextual learning by
identifying normal sequence patterns using BERT-style objectives and spotting
deviations from those patterns (Guo, Yuan, and Wu, 2021). While LogBERT
captures a richer bidirectional context compared to LogRobust, its transformer
architecture tends to be more computationally intensive and still relies on specific preprocessing and threshold choices tied to benchmarks. So, even with its
enhanced representational capacity, it doesn't completely resolve the trade-offs
enterprises face between accuracy, speed, and maintainability.

LogFormer directly addresses cross-domain generalization by decoupling source-domain pre-training from parameter-efficient target-domain adaptation and incorporating attention to information that might get lost during parsing (Guo et
al., 2024). This is particularly important for DevOps, as organizations rarely run
just one stable service. The approach indicates that transferable log knowledge
is indeed possible, but its evaluation still focuses on anomaly-detection benchmarks rather than a comprehensive detection-to-root cause analysis workflow.
Therefore, the proposed project prioritizes cross-domain degradation and low-data adaptation as key outcomes, rather than just reporting F1-scores within a
single dataset.

### 2.3 Benchmarking, generalisability and operational validity

The introduction of Loghub has really enhanced the reproducibility of comparative research by uniting logs from various sources like distributed systems, supercomputers, operating systems, and other software environments (Zhu et al.,
2023). Its datasets, including HDFS, BGL, and Thunderbird, are particularly
valuable because they vary significantly in terms of source, structure, and scale.
However, it's important to note that a benchmark collection can't completely
capture the complexities of software evolution, deployment changes, correlated
microservice failures, or the operator's need to pinpoint a cause instead of just
labeling an incident. Uddin et al. (2026), in their review of AI-based microservice log analysis, also highlight a gap between the promising results of research
and the readiness for deployment, especially concerning contextual understanding, realistic benchmarking, and thorough diagnosis across services. For the
current research question, these insights present a challenge regarding validity:
a detector might perform well on a static labeled dataset but offer little help
when a new incident arises. To tackle this, the proposed study aims to merge
established benchmark datasets with controlled fault injection in a compact
containerized microservice environment. The benchmark aspect ensures comparability, while the fault-injection component provides known causes, timing,
and service dependencies necessary for evaluating root cause analysis (RCA).
This dual approach helps avoid the misconception that benchmark classification
accuracy alone indicates a reduction in real production mean time to recovery
(MTTR).

### 2.4 Language models for anomaly analysis and root-cause reasoning

Recent advancements are shifting log analysis from just detection to incorporating language-model reasoning. LINEADAPTER introduces a smart way to
adapt smaller language models for log anomaly detection and root cause analysis (RCA), showing better precision-recall trade-offs while cutting down on the
costs associated with full fine-tuning (Liu et al., 2025). This is particularly exciting for DevOps teams working with limited resources, as these smaller models
can be run locally. However, generative diagnosis presents a unique challenge
compared to classification: an explanation might sound convincing even if it
doesn't have solid backing from the logs. Therefore, an effective operational
system needs to assess not just if a proposed cause is accurate, but also if the
evidence supporting it can be traced back reliably. The LLM4Log review by
Ma, Yang, and Chen (2026) brings together a rapidly expanding body of work
on parsing, anomaly detection, failure prediction, RCA, and summarization. It
points out key themes like retrieval grounding, parameter-efficient tuning, and
verification, while also flagging potential risks in deployment such as context
limits, latency, privacy concerns, and hallucination. Uddin et al. (2026) arrive at a similar conclusion from the microservice angle: while AI can enhance
automation, for enterprises to adopt it successfully, there needs to be stronger
validation and contextual reasoning. Collectively, these reviews suggest that
merely feeding raw production logs into a large model isn't enough and lacks
methodological rigor. The research focus should therefore be on developing a
gated, evidence-based architecture where the LLM only analyzes incident windows selected by detectors and the relevant supporting evidence.

### 2.5 Research Niche

The literature reviewed highlights some solid individual components—like
online parsing, sequential anomaly detection, semantic robustness, cross-domain
transformers, and the emerging language-model RCA. However, most studies
tend to focus on optimizing just one stage and primarily evaluate it using predictive metrics on a limited set of static datasets. What's really missing is
a comprehensive, enterprise-focused comparison that explores how reliable automation remains when the log source changes, and whether the root-cause explanations generated are backed by actual evidence. The proposed research aims
to fill this gap with four interconnected contributions: a common benchmark
for classical, recurrent, and transformer detectors; experiments on cross-system
and drift scenarios; a fault-injection RCA testbed with known causes; and an
evidence-grounded reliability gate that can choose to abstain rather than produce an unsupported diagnosis. This integrated approach directly tackles the
CA1 concern regarding the disconnect between cutting-edge AI research and its
practical application in DevOps.

## 3 Research Method & Specification

The study adopted a quantitative experimental design that includes comparative benchmarking, controlled fault injection, and ablation analysis. For
anomaly detection, the focus is on a log sequence or a time-bounded event window, depending on how the dataset is typically organized. The root cause analysis (RCA) centers around a controlled incident where the fault
that was injected is known. There was no need for any human-subject experiments. All models were assessed using fixed train/validation/test splits, ensuring the
test set remained untouched until the model and threshold selections were finalized.

### 3.1 Research Method

1. **Data preparation and representation.** HDFS, BGL, and Thunderbird logs from Loghub, masked for sensitive information (IP addresses, usernames, host identifiers, file paths, credential-like strings), parsed into event templates with Drain/Drain3 while keeping a parallel raw-text version for semantic models, formed into sequences using session identifiers (HDFS block IDs) or fixed windows otherwise.
2. **Baseline anomaly detectors.** Template-count/TF-IDF features with Isolation Forest and Random Forest, and a DeepLog-style LSTM sequence predictor.
3. **Transformer detector.** A compact transformer inspired by LogBERT/LogFormer, prioritizing reproducibility over replicating every architectural detail.
4. **Generalization and drift tests.** Source-to-target cross-domain transfer and controlled drift injection (benign token renaming, new-template insertion, frequency tweaks).
5. **Evidence-grounded RCA.** A containerized microservice testbed generating incidents with known causes (dependency timeouts, failed database connections, pod restarts, resource exhaustion), feeding a retrieval-augmented, gated LLM diagnosis step.

### 3.2 Research Resources

Python, pandas, NumPy, scikit-learn, PyTorch, Hugging Face-style tooling, Drain3,
sentence-transformer embeddings with FAISS, Docker for the fault-injection
testbed, Git for version control. *(Implementation note: the retrieval-and-RCA
language model used was a locally-hosted open-source model — `llama3.1:8b`
served via Ollama — per the ethics commitment in Section 3.4 below, rather than
an external API.)*

### 3.3 Evaluation

Anomaly detection was evaluated on precision, recall, F1-score, PR-AUC, false-positive rate, and false alarms per 10,000 events, with thresholds selected on validation data. Five repeated seeds were proposed for trainable neural models;
**implementation note: this was reduced to three seeds as an explicit scoping
decision** (see Section 4 and the Scoping Decisions table), reporting mean
performance with 95% confidence intervals. Generalization was assessed via
absolute/relative loss in F1/PR-AUC when transferring across log sources. RCA
was assessed on top-1/top-3 accuracy and mean reciprocal rank where ground
truth was available, evidence coverage, and unsupported-claim rate. **Paired
bootstrap intervals and non-parametric paired tests to compare models on the
same instances** were planned; these were not performed during the initial
Phase 3 run (independent per-seed confidence intervals were reported instead)
and were added as a follow-up — see Section 4.4.

### 3.4 Ethical Considerations of the Research

The project mainly relies on public technical datasets and synthetic or controlled
testbed logs. System logs might still include identifiers, network addresses, usernames, file paths, tokens, or sensitive configuration
details; Public Loghub data was treated as potentially sensitive despite being publicly accessible. Only the necessary subsets were downloaded, automated masking applied before any model saw the data, and published artifacts
restricted to sanitized examples. The system is designed as a decision-support prototype rather
than an autonomous remediation tool, with a reliability gate and an
abstention rule to highlight uncertainty, every RCA output tied
to retrievable log evidence, and **a local open-source language model used
throughout (`llama3.1:8b` via Ollama) to avoid sending log excerpts to an
external provider**, consistent with the plan stated here.

### 3.5 Project Plan

The project was planned over twelve capstone weeks, moving from literature
consolidation and ethics/data preparation through parsing, baseline development, transformer training, RCA module development, cross-domain tests and
ablations, prototype integration, and write-up. See `PROGRESS.md` for the
complete, dated execution log against this plan, and the Scoping Decisions
table in Section 4 for where execution diverged from the original Gantt chart
and why.

---

## 4 Results

*(New section — implementation and experimental results.)*

### 4.1 Data pipeline

Environment verified against a real python.org install with GPU-enabled
PyTorch confirmed. All three datasets downloaded in full (HDFS 11.18M lines,
BGL 4.75M lines, Thunderbird capped at 10M lines), masked, parsed with
Drain3, built into labeled sequences, and split 70/15/15 (stratified, seed 1337,
frozen before any model touched it).

**Table 4.1 — Dataset summary**

| Dataset | Templates | Sequences | Anomaly rate |
|---|---|---|---|
| HDFS | 44 | 575,061 (per-block) | 2.93% |
| BGL | 1,819 | 47,135 (100-event windows) | 10.24% |
| Thunderbird | 1,467 | 99,808 (100-event windows) | 38.36% |

The PII spot-check caught two real gaps before they became silent
data-quality problems: HDFS block IDs leaking into Drain3 templates as
literals (fixed by masking to `<BLK>` before template mining while retaining
the real value in its own column for session grouping), and Thunderbird's
`component` field carrying unmasked install paths (fixed). Final re-check: 0 of
20,000 sampled rows flagged across all three datasets.

### 4.2 Classical baselines

**Table 4.2 — Classical baseline performance**

| Dataset | Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| HDFS | Isolation Forest | 0.543 | 0.918 | 0.682 | 0.709 |
| HDFS | Random Forest | 0.999 | 0.999 | 0.999 | 1.000 |
| BGL | Isolation Forest | 0.197 | 0.934 | 0.325 | 0.252 |
| BGL | Random Forest | 0.986 | 0.978 | 0.982 | 0.997 |
| Thunderbird | Isolation Forest | 0.397 | 0.988 | 0.566 | 0.288 |
| Thunderbird | Random Forest | 1.000 | 0.999 | 0.999 | 1.000 |

### 4.3 Sequence models: LSTM and Transformer

DeepLog-style next-event prediction (window=5, top-3 hit/miss scoring),
trained on normal sequences only, three seeds (42/123/777) per dataset per
model.

**Table 4.3 — Neural detector performance (mean ± 95% CI, n=3 seeds)**

| Dataset | Model | F1 | PR-AUC |
|---|---|---|---|
| HDFS | LSTM | 0.265 ± 0.084 | 0.146 ± 0.067 |
| HDFS | Transformer | 0.271 ± 0.055 | 0.147 ± 0.037 |
| BGL | LSTM | 0.928 ± 0.014 | 0.971 ± 0.001 |
| BGL | Transformer | 0.926 ± 0.003 | 0.970 ± 0.003 |
| Thunderbird | LSTM | 0.823 ± 0.033 | 0.784 ± 0.041 |
| Thunderbird | Transformer | 0.815 ± 0.045 | 0.769 ± 0.085 |

**Table 4.4 — Ablation: template-only vs. +semantic embedding (seed 42)**

| Dataset | Template-only | +Semantic embedding |
|---|---|---|
| HDFS | 0.300 | 0.275 |
| BGL | 0.932 | 0.927 |
| Thunderbird | 0.836 | 0.796 |

Adding a frozen semantic embedding of each template's text does not help and
mildly hurts, most clearly on Thunderbird.

### 4.4 Statistical significance (paired bootstrap)

Section 3.3 specified paired bootstrap intervals and non-parametric paired
tests. These were run as a follow-up: for each dataset, every model pair was
scored on the identical frozen test split (same instance order), and a paired
bootstrap (B=2,000 resamples) was used to test whether the F1 difference
between models is distinguishable from zero on the *same* test instances —
which the independent per-seed CIs in Table 4.3 cannot do.

**Table 4.5 — Paired bootstrap: LSTM vs. other models (seed 42, B=2,000)**

| Dataset | Comparison | F1 diff | 95% CI | p-value | Significant? |
|---|---|---|---|---|---|
| HDFS | LSTM vs. Transformer | +0.021 | [+0.013, +0.031] | <0.0001 | Yes |
| HDFS | LSTM vs. Isolation Forest | −0.383 | [−0.401, −0.364] | <0.0001 | Yes |
| HDFS | LSTM vs. Random Forest | −0.699 | [−0.715, −0.683] | <0.0001 | Yes |
| BGL | LSTM vs. Transformer | +0.004 | [−0.000, +0.009] | 0.073 | **No** |
| BGL | LSTM vs. Isolation Forest | +0.606 | [+0.585, +0.627] | <0.0001 | Yes |
| BGL | LSTM vs. Random Forest | −0.050 | [−0.064, −0.037] | <0.0001 | Yes |
| Thunderbird | LSTM vs. Transformer | +0.027 | [+0.023, +0.031] | <0.0001 | Yes |
| Thunderbird | LSTM vs. Isolation Forest | +0.270 | [+0.262, +0.278] | <0.0001 | Yes |
| Thunderbird | LSTM vs. Random Forest | −0.163 | [−0.170, −0.156] | <0.0001 | Yes |

A key nuance: on HDFS and Thunderbird, LSTM-vs-Transformer differences are
*statistically* significant despite being practically tiny (+0.02–0.03 F1) —
large test sets (tens of thousands of instances) give enough statistical power
to detect small effects. BGL is the one dataset where LSTM and Transformer
are genuinely statistically indistinguishable. By contrast, the LSTM-vs-classical
gaps are both significant *and* large, especially the HDFS LSTM-vs-Random-Forest
gap (−0.699). See Section 5.1 for discussion.

### 4.5 Cost and latency vs. accuracy

Closes sub-question 4 — Section 3.3 asked how accuracy changes when
latency, throughput and computational cost are considered together.

**Table 4.6 — Training cost vs. accuracy**

| Dataset | Model | Train time | F1 | Cost per 0.1 F1 |
|---|---|---|---|---|
| HDFS | Isolation Forest | 4.7s | 0.682 | 0.69s |
| HDFS | Random Forest | 13.3s | 0.999 | 1.33s |
| HDFS | LSTM | 955.3s (mean/3 seeds) | 0.265 | 360.47s |
| HDFS | Transformer | 1299.6s (mean/3 seeds) | 0.271 | 479.81s |
| BGL | Isolation Forest | 0.7s | 0.325 | 0.23s |
| BGL | Random Forest | 1.1s | 0.982 | 0.12s |
| BGL | LSTM | 773.6s (mean/3 seeds) | 0.928 | 83.33s |
| BGL | Transformer | 591.7s (mean/3 seeds) | 0.926 | 63.90s |
| Thunderbird | Isolation Forest | 1.5s | 0.566 | 0.27s |
| Thunderbird | Random Forest | 6.3s | 0.999 | 0.63s |
| Thunderbird | LSTM | 1383.0s (mean/3 seeds) | 0.823 | 168.09s |
| Thunderbird | Transformer | 1484.3s (mean/3 seeds) | 0.815 | 182.02s |

*"Cost per 0.1 F1" = training seconds per 0.1 of F1 achieved; lower is more
cost-efficient.* Classical baselines are 2–3 orders of magnitude cheaper to
train than the neural models across every dataset, and on HDFS/Thunderbird
Random Forest is simultaneously cheaper *and* more accurate.

**RCA latency**: mean 4.1 seconds per incident end-to-end (retrieval + local
LLM inference + gate check), n=20. Consistent with the proposal's own caveat
(Section 3.3), this is reported as a reproducible pipeline-latency measurement,
not a claim about real-world MTTR reduction, which would require
production/human-operator experiments outside this project's scope.

### 4.6 Cross-domain transfer and drift robustness

All six directed cross-domain pairs (HDFS↔BGL, HDFS↔Thunderbird,
BGL↔Thunderbird) showed **100% template vocabulary disjunction** at full
scale — these three systems share zero templates in either direction. A naive
before/after-adaptation F1 comparison looked identical in every pair (a
threshold-selection artifact: with every window collapsed to the same "unseen"
token, a classifier can only express "flag everyone," regardless of whether its
constant prediction is "always anomalous" or "always normal"). A
threshold-independent `mean_raw_score` diagnostic exposed the real signal: it
swings from ~1.0 to ~0.0 in every pair, confirming the model's predictions
genuinely flipped after limited target adaptation, even though training loss
crashed from ~0.75 to ~0.001 in every case.

**Table 4.7 — Drift robustness (LSTM, seed 42)**

| Dataset | FPR (clean) | FPR (drifted) | Relative increase |
|---|---|---|---|
| HDFS | 2.6% | 99.9% | 37× (catastrophic) |
| BGL | 0.4% | 5.4% | 11.7× (severe) |
| Thunderbird | 17.3% | 66.0% | 2.8× (moderate) |

### 4.7 Fault-injection testbed

Four Flask microservices (`gateway → orders → {inventory, payments}`) in
Docker Compose, logging through the same masking+Drain3 pipeline as the
main datasets. Four fault types (`db_failure`, `dependency_timeout`,
`resource_exhaustion`, `pod_restart`) each independently verified.
An 8-incident scenario (280s, 245 requests, 34 errors) was run with ground
truth recorded automatically; all 8 incidents show a clear ERROR/WARNING
line-count spike at their exact injection timestamp.

### 4.8 Retrieval layer

`all-MiniLM-L6-v2` embeddings over 3,360 documents (every dataset template +
every testbed template + the 8 testbed incident descriptions), FAISS
`IndexFlatIP`. Sanity-checked with 6 held-out queries; every top-1 match was
semantically on point (e.g. "memory usage is very high" → testbed's exact
`resource_exhaustion` template at 0.714 cosine similarity).

### 4.9 RCA and the reliability gate

Pipeline: incident → FAISS retrieval (top-5) → local LLM (`llama3.1:8b`)
diagnosis → gate, which independently re-checks that cited evidence ids
actually exist and that confidence clears a threshold, overriding the LLM to
abstain if either check fails.

**Table 4.8 — RCA gate metrics (20 cases: 8 testbed + 12 real detected anomalies)**

| Metric | Value |
|---|---|
| LLM self-reported "answer" rate | 19/20 (95%) |
| Gate "answer" rate | 18/20 (90%) |
| Gate override rate | 1/20 (5%) |
| Avg. evidence coverage | 0.95 |
| Avg. confidence | 0.76 |

Manual hand-check of all 20 cases: 16 clearly correct, 1 correct-but-
conservatively-gated, 1 honest self-abstain, 2 flagged. The 2 flagged cases are
both `pod_restart` incidents where the LLM correctly named the affected
service but described the symptom an *upstream caller* sees (a connection
error) rather than the actual mechanism (the process crashed and restarted) —
a real, explainable gap (the crashing service's own log lines weren't in the
evidence shown), not hallucination.

### 4.10 Retrieval ablation (with vs. without retrieval)

Closes sub-question 3. Same 20 incidents, same LLM, same gate — the only
change is whether the prompt receives the FAISS top-5 evidence or an explicit
"no evidence was retrieved" message.

**Table 4.9 — With vs. without retrieval**

| | With retrieval | Without retrieval |
|---|---|---|
| LLM "answer" rate | 19/20 (95%) | 18/20 (90%) |
| **Gate "answer" rate** | **18/20 (90%)** | **0/20 (0%)** |
| Gate override rate | 1/20 (5%) | 18/20 (90%) |
| Avg. confidence | 0.76 | **0.85** |

Without retrieval, every attempted answer is unsupported by construction (no
evidence exists to cite), and the gate correctly abstains on all 18 attempted
answers. Confidence *rose* when grounding was removed — the opposite of
good calibration, and the exact hallucination signature the gate is designed to
catch regardless of what the model reports about itself. One case
(`testbed_0`) also flipped from a correct diagnosis (with retrieval) to an
actively wrong one (without), not merely an unsupported one.

### 4.11 Scoping decisions

**Table 4.10 — What changed from the original plan, and why**

| Original | Changed to | Reason |
|---|---|---|
| 5 seeds per neural run | 3 seeds | Time budget; 3 still gives a meaningful mean±CI |
| 3 ablation variants (template / raw-text-only / combined) | 2 variants | Raw-text-only dropped; template-only vs. combined is the comparison that answers "does semantic signal help" |
| Paired bootstrap / non-parametric tests (Section 3.3) promised from the start | Delivered as a follow-up (Section 4.4), after independent per-seed CIs were reported first | Identified as a gap during review; closed before this document was finalized |
| Fixed window_size=5/topk=3 for all LSTM evaluation | Kept as the headline config; a window_size=10/topk=5 sensitivity check was run on HDFS specifically | The HDFS result looked surprising (losing to Isolation Forest); the check found F1 nearly doubles (0.300→0.545) with different hyperparameters, narrowing but not closing the gap — headline numbers kept as-is on principle (changing hyperparameters per-dataset after seeing results would bias the cross-dataset comparison); documented as a sensitivity note, not folded silently into the main table |
| Early stopping patience=3, min_delta=1e-4 | patience=2, min_delta=5e-4 | The original setting let BGL/Thunderbird runs wander for many epochs on sub-0.001 "improvements"; tightened after measuring real per-epoch cost mid-run |

---

## 5 Discussion

### 5.1 Does more model complexity help? (Sub-question 1)

Section 2.1 (Khan et al., 2024) argues that parsing quality alone doesn't
predict detector quality. This project's results extend that caution one level
up the complexity stack: *model* complexity alone doesn't predict detector
quality either. The Transformer — architecturally closer to LogBERT/LogFormer
(Guo, Yuan and Wu, 2021; Guo et al., 2024) than the LSTM — never produces a
practically meaningful improvement over the simpler DeepLog-style LSTM
(Du et al., 2017) on any of the three datasets (Table 4.3), and the paired
bootstrap test (Table 4.5) confirms this isn't just an artifact of seed variance:
on BGL the two are genuinely statistically indistinguishable, and on HDFS and
Thunderbird the statistically-significant differences are small enough (+0.02–
0.03 F1) to be a rounding error in practice. Meanwhile, both neural models
lose decisively to the simplest baseline, Isolation Forest, on HDFS specifically
(Table 4.5, −0.383 and worse), and lose by a wide margin to Random Forest
everywhere. The HDFS sensitivity check (Section 4.11) shows this specific
result is sensitive to the window-size/top-k hyperparameters — a caution
Khan et al.'s own methodology would predict — but even the improved
configuration doesn't close the gap to Isolation Forest.

The practical reading, consistent with the proposal's decision-matrix framing
in Section 3.3, is that *none* of these models is a context-free "best" choice.
Random Forest's near-perfect scores (Table 4.2) are a product of these
specific benchmarks' templates being highly separable given a label — exactly
the condition under which Uddin et al. (2026) warn that benchmark accuracy
doesn't transfer to deployment readiness, since production systems rarely have
Random Forest's training-time supervision available for genuinely novel
failure modes.

### 5.2 Generalization and drift (Sub-question 2)

The cross-domain result (Section 4.6) is the sharpest confirmation in this
project of the gap LogFormer (Guo et al., 2024) and Uddin et al. (2026) both
flag: a detector's cross-domain transferability cannot be assumed from its
within-domain accuracy. HDFS, BGL and Thunderbird are different enough as
systems that they share *zero* templates — a more extreme case than
LogFormer's own cross-domain evaluation, which moves between related
log sources. The methodological point this project adds is about measurement,
not just the finding itself: a naive P/R/F1 comparison of the model before and
after limited target adaptation looked unchanged in every pair, which would
have been reported as "adaptation did nothing" had the `mean_raw_score`
diagnostic (Section 4.6) not been added to check that conclusion — the
model's predictions had, in fact, completely flipped. This is a direct,
small-scale demonstration of the exact failure mode Khan et al. describe for
parsing-quality metrics, now shown to apply to evaluation metrics for adapted
models too: a metric can look stable while the thing it's supposed to measure
has changed completely.

The drift results (Table 4.7) are the project's clearest link to LogRobust
(Zhang et al., 2019) and LogBERT's (Guo, Yuan and Wu, 2021) motivating
concern — static, closed-world models misfiring on harmless change. HDFS's
37× false-positive blowup under mild synthetic drift is a concrete illustration
of exactly the brittleness those papers were written to address, and the
severity gradient across datasets (HDFS catastrophic, BGL severe, Thunderbird
moderate) suggests vocabulary size matters: HDFS's 44 templates leave far less
room to reinterpret a renamed or new template as "close enough to something
already known" than BGL's 1,819 or Thunderbird's 1,467.

### 5.3 Retrieval-grounded RCA without unsupported claims (Sub-question 3)

The retrieval ablation (Section 4.10) directly answers this sub-question, and
does so more sharply than expected going in. The LLM4Log review (Ma, Yang
and Chen, 2026) frames hallucination risk as a calibration problem — models
sounding confident regardless of whether they have grounds to be. This
project's ablation reproduces that exact failure in miniature and quantifies it:
removing retrieval raised the model's self-reported confidence (0.76 → 0.85)
while removing the only thing that could have justified any of it. That is the
clearest evidence in this project that **the gate, not the model's own
self-assessment, has to be the thing doing the trust-checking** — a
design choice Section 3.4's ethics framing already committed to on principle,
and which this result validates empirically rather than just by argument.
LINEADAPTER's (Liu et al., 2025) framing of parameter-efficient local models as
viable for RCA is also supported here in a narrow but real sense: a locally-hosted
8B model, with no fine-tuning at all, produced correct diagnoses on 16/20
hand-checked cases once properly grounded — the gate and retrieval
architecture did more work than the model's raw capability.

### 5.4 Cost, latency and deployment readiness (Sub-question 4)

Table 4.6 is this project's most direct engagement with Uddin et al.'s (2026)
deployment-readiness concern. The headline number — Random Forest reaching
F1=0.999 on HDFS in 13.3 seconds, versus the LSTM reaching F1=0.265 in 955
seconds — is not a close call. Even where the neural models are genuinely
competitive on accuracy (BGL, Thunderbird), they are roughly two to three
orders of magnitude more expensive to train for that accuracy. This
reframes the "is the complexity worth it" question from Section 5.1 in cost
terms: not only does the added model complexity fail to reliably improve
accuracy, it does so while consuming vastly more compute, which is precisely
the practical-adoption friction Uddin et al. describe as separating research
results from enterprise deployment.

### 5.5 On statistical vs. practical significance

A theme that runs through Sections 5.1–5.4 and is made explicit by the paired
bootstrap results (Table 4.5) deserves its own note: with large enough test
sets, nearly any real difference becomes statistically detectable, whether or
not it is practically meaningful. The HDFS and Thunderbird LSTM-vs-
Transformer comparisons are the clearest example — both "significant" at
p<0.0001, both representing a difference (0.02–0.03 F1) that would not change
a single deployment decision. This project treats the BGL result (genuinely
non-significant, the one case where the test sample size and effect size
happened to align such that the null hypothesis isn't rejected) and the
HDFS/Thunderbird results side by side deliberately, rather than reporting only
significance flags, because the gap between the two readings is itself the
point: a decision-matrix evaluation (Section 3.3's own stated goal) needs
effect sizes, not just p-values.

---

## 6 Conclusion

### 6.1 Answering the Research Question

**RQ**: *How effective can a hybrid AI pipeline that combines
machine-learning/deep-learning anomaly detection and evidence-gathered
language-model reasoning be in terms of automating anomaly detection and
root-cause analysis across different DevOps logs? What are the technical
factors that most affect the pipeline's readiness for deployment in
real-world enterprises?*

The pipeline is effective at detection on two of three benchmarks (BGL,
Thunderbird show strong neural-model F1) and at RCA when evidence is
available (90% of 20 hand-checked cases pass the reliability gate, 16 of those
judged correct on manual review). It is measurably *not* effective at
cross-domain generalization (100% vocabulary disjunction across all three
systems) or at robustness to mild drift (up to 37× false-alarm inflation on
HDFS). The technical factors most affecting enterprise readiness, in order of
how decisively this project's evidence speaks to them, are: **(1) computational
cost relative to accuracy gained** — classical models are 2–3 orders of
magnitude cheaper and frequently more accurate; **(2) drift brittleness**,
which scales inversely with a dataset's template vocabulary size; **(3) the
complete absence of cross-domain transfer** between unrelated systems,
meaning a model trained on one service tells you nothing about another; and
**(4) the necessity of an external, non-self-reported trust mechanism** for any
LLM-based diagnosis step, since the model's own confidence was shown to
move in the wrong direction exactly when it should have moved in the right
one.

### 6.2 Answering the operational sub-questions

1. **Which detector balances precision/recall/false alarms best?** No single
   detector wins across all three datasets (Table 4.2, 4.3) — Random Forest
   dominates raw accuracy everywhere but at a cost Section 5.4 quantifies;
   among the sequence models, neither LSTM nor Transformer is reliably
   better than the other (Table 4.5), and both lose to even a simple
   Isolation Forest on HDFS.
2. **How is performance affected by cross-domain transfer and drift?**
   Cross-domain transfer is completely non-viable between these three
   systems (100% vocabulary disjunction, 6/6 pairs); drift causes severe to
   catastrophic false-alarm inflation, worse on smaller-vocabulary datasets.
3. **Does retrieval-grounded reasoning enhance RCA without increasing
   unsupported claims?** Yes — and more than that, retrieval is what makes
   a gated "answer" possible at all; without it, the gate correctly
   suppresses 100% of attempted diagnoses because none could be
   validly grounded, and the model's confidence, not its accuracy,
   degrades in exactly the wrong direction when grounding is removed.
4. **How does accuracy change when latency/throughput/cost are
   considered?** Classical baselines remain highly competitive once
   training cost is factored in, often dominating the neural models on
   both axes simultaneously (Table 4.6); RCA latency (4.1s/incident) is
   reproducible but explicitly not a stand-in for a real MTTR claim.

### 6.3 Limitations

- **HDFS detector performance is hyperparameter-sensitive** (Section 4.11)
  — the headline configuration may understate what the LSTM/Transformer
  can do on HDFS specifically; the qualitative conclusion (not clearly better
  than Isolation Forest) held up under a sensitivity check, but the margin
  narrowed substantially.
- **8 of the 20 RCA hand-check cases have an answer already present in the
  retrieval corpus** (built from the same ground truth), so those 8 mainly
  test retrieval+gate mechanics rather than pure causal inference under
  ambiguity; the 12 real-dataset cases, which have no equivalent
  answer-sheet, are the fairer test and all 12 produced plausible,
  template-grounded diagnoses.
- **`pod_restart` root-cause identification is incomplete** when only
  observer-service logs are available — a real, explainable gap in evidence
  availability, not a reasoning failure.
- **RCA evaluation is 20 cases** by design — a time-boxed qualitative
  reliability-gate demonstration, not a claim of statistical power; the
  proposal's own evaluation plan treats it this way.
- **Random Forest's near-perfect scores likely reflect benchmark template
  separability**, not a general claim about supervised detection in
  production, where label availability (not algorithm choice) is usually the
  binding constraint.

### 6.4 Future Work

- **Per-dataset hyperparameter tuning as its own controlled experiment** —
  the HDFS sensitivity check (window=10, topk=5) suggests meaningful
  headroom exists; a systematic sweep, reported separately from the
  fixed-protocol headline comparison, would quantify how much of the
  LSTM/Transformer gap to classical baselines on HDFS is tunable away.
- **A richer cross-domain test** using log sources that are related rather
  than completely disjoint (e.g. two versions of the same application, or
  two microservices within one organization), to test transfer under
  realistic partial-overlap conditions rather than this project's
  worst-case (100% disjoint) scenario.
- **Extending the retrieval ablation to the combined/semantic-embedding
  model variant**, to see whether retrieval's effect on the gate interacts
  with the detector representation choice, not just the RCA stage.
- **A production-adjacent MTTR study** with human operators, which both
  this project and the original proposal explicitly flag as the only way to
  convert the 4.1s/incident pipeline-latency figure into a real
  response-time claim.
- **Testing the reliability gate's sensitivity to its own threshold** (currently
  0.5 confidence, exact-citation-match) — the retrieval ablation shows the
  gate's strictness is what makes it trustworthy, but the specific threshold
  value was fixed by convention rather than tuned, and a sweep would
  clarify the precision/abstention trade-off it implies.

---

## Updated References

*(Additions beyond the original proposal's literature review, for tools and
models used in the implementation.)*

Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy, G., Mazaré, P.-E.,
Lomeli, M., Hosseini, L. and Jégou, H. (2024) 'The Faiss library',
arXiv:2401.08281. Preprint. https://arxiv.org/abs/2401.08281.

Dubey, A. et al. (2024) 'The Llama 3 Herd of Models', arXiv:2407.21783.
Preprint. https://arxiv.org/abs/2407.21783. *(llama3.1:8b, used locally via
Ollama for the RCA/reliability-gate stage.)*

Ollama (2026) *Ollama: Get up and running with large language models
locally*. Available at: https://ollama.com (Accessed: 2026).

Pedregosa, F. et al. (2011) 'Scikit-learn: Machine Learning in Python',
*Journal of Machine Learning Research*, 12, pp. 2825–2830.

Paszke, A. et al. (2019) 'PyTorch: An Imperative Style, High-Performance
Deep Learning Library', *Advances in Neural Information Processing Systems*,
32.

Reimers, N. and Gurevych, I. (2019) 'Sentence-BERT: Sentence Embeddings
using Siamese BERT-Networks', *Proceedings of the 2019 Conference on
Empirical Methods in Natural Language Processing (EMNLP)*, pp. 3982–3992.
https://doi.org/10.18653/v1/D19-1410. *(`all-MiniLM-L6-v2`, used for all
semantic embeddings: the ablation's template embeddings and the FAISS
retrieval corpus.)*

*(Original proposal references — Du et al. 2017; Guo, Yang, Liu, Bai, Wang,
Li, Zheng, Zhang, Peng and Tian 2024; Guo, Yuan and Wu 2021; He, Zhu, Zheng
and Lyu 2017; Khan, Shin, Bianculli and Briand 2024; Liu, Bao, Zhang, Li,
Ranjan and Jha 2025; Ma, Yang and Chen 2026; Uddin, Weerasinghe, Gajewski,
Akbarsharifi, Akbarsharifi, Stoner, Cerny and He 2026; Zhang, Xu, Lin, Qiao,
Zhang, Dang, Xie, Yang, Cheng, Li, Chen, He, Yao, Lou, Chintalapati, Shen
and Zhang 2019; Zhu, He, He, Liu and Lyu 2023 — carried over unchanged;
see the original proposal document for full entries.)*
