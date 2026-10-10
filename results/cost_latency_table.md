# Cost / Latency vs. Accuracy (closes proposal sub-question 4)

## Training cost vs. accuracy

| Dataset | Model | Train time | F1 | Cost per 0.1 F1 |
|---|---|---|---|---|
| hdfs | isolation_forest | 4.7s | 0.682 | 0.69s |
| hdfs | random_forest | 13.3s | 0.999 | 1.33s |
| hdfs | lstm | 955.3s (mean/3 seeds) | 0.265 | 360.47s |
| hdfs | transformer | 1299.6s (mean/3 seeds) | 0.271 | 479.81s |
| bgl | isolation_forest | 0.7s | 0.325 | 0.23s |
| bgl | random_forest | 1.1s | 0.982 | 0.12s |
| bgl | lstm | 773.6s (mean/3 seeds) | 0.928 | 83.33s |
| bgl | transformer | 591.7s (mean/3 seeds) | 0.926 | 63.90s |
| thunderbird | isolation_forest | 1.5s | 0.566 | 0.27s |
| thunderbird | random_forest | 6.3s | 0.999 | 0.63s |
| thunderbird | lstm | 1383.0s (mean/3 seeds) | 0.823 | 168.09s |
| thunderbird | transformer | 1484.3s (mean/3 seeds) | 0.815 | 182.02s |

*"Cost per 0.1 F1" = training seconds spent per 0.1 of F1 achieved — a cheap, legible way to compare "was the extra compute worth it" across wildly different training costs; lower is more cost-efficient.*


## RCA latency (time-to-diagnosis)

- Overall mean: **4.1s per incident** (n=20, retrieval + local LLM inference + gate check, end to end)
- testbed: 4.2s
- hdfs: 3.9s
- bgl: 4.1s
- thunderbird: 4.2s

This is the proposal's "automated time-to-diagnosis" metric (Section 3.3) -- note this is a reproducible *pipeline* latency measurement, not a claim about real MTTR reduction, consistent with the proposal's own caveat that a real MTTR claim needs production/human-operator experiments this project doesn't run.