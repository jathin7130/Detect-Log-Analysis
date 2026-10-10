# Retrieval Ablation — closes proposal sub-question 3

**Sub-question 3**: "whether retrieval-grounded language-model reasoning
enhances top-k root-cause identification without leading to an increase in
unsupported claims."

Same 20 incidents, same LLM (`llama3.1:8b`), same reliability gate logic.
Only difference: the "without retrieval" arm's prompt tells the model no
evidence was retrieved, instead of giving it the FAISS top-5. Note the raw
incident log lines are still shown in both arms — this isolates the effect
of *retrieval* specifically, not all context.

## Headline numbers

| | WITH retrieval | WITHOUT retrieval |
|---|---|---|
| LLM self-reported "answer" rate | 19/20 (95%) | 18/20 (90%) |
| **Gate "answer" rate** | **18/20 (90%)** | **0/20 (0%)** |
| Gate override rate | 1/20 (5%) | 18/20 (90%) |
| Avg. self-reported confidence | 0.76 | **0.85** |

## What this shows

**Retrieval is what makes a gated "answer" possible at all.** Without it,
every single attempted answer is an unsupported claim by construction —
there is nothing in the prompt the model could have validly cited — so the
gate correctly abstains on 100% of them. With retrieval, 90% clear the gate.
Retrieval isn't a minor accuracy boost here; it's the difference between a
system that can ever answer under the gate's own rules and one that can't.

**Confidence went up, not down, when grounding was removed** (0.76 → 0.85).
That's the opposite of what you'd want from a well-calibrated model, and
it's exactly the hallucination risk the gate exists to catch (the LLM4Log
review's central concern) — a model doesn't reliably know when it's
guessing, which is precisely why an external, independent check (not the
model's own self-reported confidence) is necessary.

## Spot-check: did removing retrieval make diagnoses wrong, or just ungrounded?

Both, depending on the case:

- **`testbed_0`** (ground truth: db_failure on inventory): with retrieval,
  correctly diagnosed. Without retrieval, the LLM guessed "orders service
  experiencing high latency" — **wrong**, not just unsupported.
- **HDFS/Thunderbird real-anomaly cases**: diagnoses stayed largely similar
  with or without retrieval, because for these cases the raw log window
  itself already contains the informative masked template text (that's what
  an anomaly window *is*) — retrieval adds confirming context rather than
  being the only source of signal.

So retrieval's value is uneven across case types: it's load-bearing for
incidents where the raw window alone is ambiguous (e.g. an observer
service's logs, which only show symptoms), and more of a confirmation layer
when the window itself is already diagnostic. Both are legitimate RCA
scenarios in practice, and the gate's behavior — refusing to answer without
grounding regardless of which kind of case it is — doesn't depend on
knowing which case type it's looking at.

## Answer to sub-question 3

**Yes, with a caveat.** Retrieval-grounded reasoning does enhance RCA
without increasing unsupported claims — in fact it's the only way any
claims get through the gate at all. The caveat: this project's gate
achieves that by being strict (any answer with zero valid citations is
abstained, full stop), not by the LLM reliably self-assessing when it needs
evidence — the confidence numbers show the model doesn't do that on its
own.
