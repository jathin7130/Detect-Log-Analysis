"""Phase 5 — retrieval layer: embed every known template (HDFS/BGL/Thunderbird
+ the fault-injection testbed) plus the testbed's ground-truth incidents,
and build a FAISS index over them.

This is the corpus the RCA stage (Phase 7) will later query against: given
a new incident window, retrieve the most similar historical templates and
past incident descriptions as evidence for the reliability gate to check
citations against.

Same embedding model as the ablation's template embeddings
(all-MiniLM-L6-v2) — one consistent semantic space across the project
rather than a second one just for retrieval.
"""
import json
import csv
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA_MASKED = ROOT / "data" / "masked"
TESTBED = ROOT / "testbed"
RETRIEVAL = ROOT / "retrieval"

MODEL_NAME = "all-MiniLM-L6-v2"
DATASETS = ["hdfs", "bgl", "thunderbird"]


def load_dataset_templates():
    docs = []
    for ds in DATASETS:
        path = DATA_MASKED / ds / "templates.csv"
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                docs.append({
                    "doc_id": f"{ds}_template_{row['event_id']}",
                    "source": ds,
                    "doc_type": "template",
                    "text": row["template"],
                    "metadata": {"event_id": row["event_id"], "occurrences": row["occurrences"]},
                })
    return docs


def load_testbed_templates():
    path = TESTBED / "parsed" / "testbed_templates.csv"
    if not path.exists():
        return []
    docs = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            docs.append({
                "doc_id": f"testbed_template_{row['event_id']}",
                "source": "testbed",
                "doc_type": "template",
                "text": row["template"],
                "metadata": {"event_id": row["event_id"], "occurrences": row["occurrences"]},
            })
    return docs


def load_testbed_incidents():
    path = TESTBED / "incidents" / "ground_truth.json"
    if not path.exists():
        return []
    manifest = json.loads(path.read_text(encoding="utf-8"))
    docs = []
    for i, inc in enumerate(manifest["incidents"]):
        text = (
            f"Fault of type {inc['fault_type']} injected on service {inc['service']}, "
            f"lasting {inc['fault_duration_s']} seconds."
        )
        docs.append({
            "doc_id": f"testbed_incident_{i}",
            "source": "testbed",
            "doc_type": "incident",
            "text": text,
            "metadata": inc,
        })
    return docs


def main():
    RETRIEVAL.mkdir(parents=True, exist_ok=True)
    docs = load_dataset_templates() + load_testbed_templates() + load_testbed_incidents()
    print(f"corpus: {len(docs)} documents "
          f"({sum(1 for d in docs if d['doc_type']=='template')} templates, "
          f"{sum(1 for d in docs if d['doc_type']=='incident')} incidents)")

    model = SentenceTransformer(MODEL_NAME)
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = embeddings.astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])  # normalized vectors -> inner product == cosine sim
    index.add(embeddings)

    faiss.write_index(index, str(RETRIEVAL / "index.faiss"))
    (RETRIEVAL / "documents.json").write_text(json.dumps(docs, indent=2), encoding="utf-8")

    print(f"index built: {index.ntotal} vectors, dim={embeddings.shape[1]} -> {RETRIEVAL / 'index.faiss'}")


if __name__ == "__main__":
    main()
