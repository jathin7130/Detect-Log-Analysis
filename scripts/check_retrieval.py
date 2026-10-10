"""Manual relevance sanity-check for the retrieval layer: a handful of
made-up incident queries, each with an intuitively-correct kind of
match, so a human (or future reader of this output) can confirm the
embeddings aren't nonsense before Phase 7 builds the RCA gate on top.
"""
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
RETRIEVAL = ROOT / "retrieval"
MODEL_NAME = "all-MiniLM-L6-v2"

QUERIES = [
    "database connection could not be established",
    "a downstream service call timed out and had to be retried",
    "the process crashed and had to restart",
    "memory usage is very high, system under resource pressure",
    "a block of data was being written and replicated across nodes",
    "a hardware component reported a parity error",
]


def main():
    index = faiss.read_index(str(RETRIEVAL / "index.faiss"))
    docs = json.loads((RETRIEVAL / "documents.json").read_text(encoding="utf-8"))
    model = SentenceTransformer(MODEL_NAME)

    for query in QUERIES:
        qvec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        scores, idx = index.search(qvec, k=3)
        print(f"\nQuery: {query!r}")
        for rank, (score, doc_idx) in enumerate(zip(scores[0], idx[0]), start=1):
            d = docs[doc_idx]
            print(f"  {rank}. [{score:.3f}] ({d['source']}/{d['doc_type']}) {d['text'][:90]}")


if __name__ == "__main__":
    main()
