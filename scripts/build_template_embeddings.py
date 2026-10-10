"""Precompute a semantic (sentence-transformer) embedding for each dataset's
distinct Drain3 templates — the "raw-text signal" half of the combined
ablation variant (template-id embedding + semantic template embedding, vs.
template-id embedding alone).

Each template's own string stands in for "raw text" here: within a
template cluster the masked content is near-identical modulo the
parameters Drain3 wildcarded out, so the template string is a reasonable,
cheap proxy for embedding every line in that cluster individually.

Output: data/features/<dataset>/template_embeddings.npy, shape
        (max_event_id + 1, embed_dim); row 0 (padding/unseen) is all zeros.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA_MASKED = ROOT / "data" / "masked"
DATA_FEATURES = ROOT / "data" / "features"

MODEL_NAME = "all-MiniLM-L6-v2"


def build(dataset: str, model: SentenceTransformer):
    templates_path = DATA_MASKED / dataset / "templates.csv"
    ids, texts = [], []
    with templates_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(row["event_id"]))
            texts.append(row["template"])

    max_id = max(ids)
    embed_dim = model.get_sentence_embedding_dimension()
    table = np.zeros((max_id + 1, embed_dim), dtype=np.float32)

    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    for eid, vec in zip(ids, vectors):
        table[eid] = vec

    out_dir = DATA_FEATURES / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "template_embeddings.npy"
    np.save(out_path, table)
    print(f"[{dataset}] {len(ids)} templates embedded, table shape={table.shape} -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hdfs", "bgl", "thunderbird"])
    args = ap.parse_args()
    model = SentenceTransformer(MODEL_NAME)
    build(args.dataset, model)
