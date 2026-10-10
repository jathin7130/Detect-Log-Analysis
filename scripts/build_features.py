"""Build TF-IDF and template-count features over event-id sequences.

Each sequence's event_ids string (e.g. "1 2 1 1 3") is treated as a
"document" of template tokens. Vectorizers are fit on the train split only
and applied to val/test, consistent with the frozen-split rule — no
information from val/test leaks into feature construction.

Output: data/features/<dataset>/{tfidf,counts}_{train,val,test}.npz
        data/features/<dataset>/labels_{train,val,test}.npy
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
DATA_SEQ = ROOT / "data" / "sequences"
DATA_SPLITS = ROOT / "data" / "splits"
DATA_FEATURES = ROOT / "data" / "features"


def build(dataset: str):
    out_dir = DATA_FEATURES / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs = pd.read_csv(DATA_SEQ / f"{dataset}.csv", dtype={"seq_id": str})
    seqs = seqs.set_index("seq_id")

    splits = {}
    for split in ["train", "val", "test"]:
        split_df = pd.read_csv(DATA_SPLITS / f"{dataset}_{split}.csv", dtype={"seq_id": str})
        merged = split_df.merge(seqs[["event_ids"]], on="seq_id", how="left")
        splits[split] = merged

    tokenizer = str.split
    count_vec = CountVectorizer(tokenizer=tokenizer, token_pattern=None, lowercase=False)
    tfidf_vec = TfidfVectorizer(tokenizer=tokenizer, token_pattern=None, lowercase=False)

    count_vec.fit(splits["train"]["event_ids"])
    tfidf_vec.fit(splits["train"]["event_ids"])

    for split, df in splits.items():
        counts = count_vec.transform(df["event_ids"])
        tfidf = tfidf_vec.transform(df["event_ids"])
        labels = df["label"].to_numpy()

        sp.save_npz(out_dir / f"counts_{split}.npz", counts)
        sp.save_npz(out_dir / f"tfidf_{split}.npz", tfidf)
        np.save(out_dir / f"labels_{split}.npy", labels)

    meta = {
        "dataset": dataset,
        "vocab_size": len(count_vec.vocabulary_),
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[{dataset}] vocab={meta['vocab_size']} train={meta['n_train']} "
          f"val={meta['n_val']} test={meta['n_test']} -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hdfs", "bgl", "thunderbird"])
    args = ap.parse_args()
    build(args.dataset)
