"""Location database: frame embeddings plus the metadata that turns a hit into a place.

Exact search over a plain matrix. At the scale this project reaches -- a whole store
sampled every half second is a few thousand frames -- a numpy matmul answers a query
in about 0.1 ms, so an approximate index buys nothing. FAISS was the original plan,
but faiss-cpu and torch each bundle their own OpenMP runtime and abort when both are
loaded, so it costs a crash and saves no time. Swap it in here if the database ever
reaches the millions of rows where approximate search actually pays off.
"""

import csv
from pathlib import Path

import numpy as np

EMB_FILE = "index.npy"
META_FILE = "database.csv"
FIELDS = ["row", "frame", "filename", "time_sec", "label"]


def save(emb, rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / EMB_FILE, emb.astype(np.float32))
    with open(out_dir / META_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load(db_dir):
    emb = np.load(db_dir / EMB_FILE)
    with open(db_dir / META_FILE) as f:
        rows = list(csv.DictReader(f))
    return emb, rows


def search(db_emb, q_emb, k=5):
    """Cosine similarity -- both sides are L2-normalized. Returns (sims, indices)."""
    k = min(k, len(db_emb))
    sims = q_emb @ db_emb.T
    idx = np.argsort(-sims, axis=1)[:, :k]
    return np.take_along_axis(sims, idx, axis=1), idx
