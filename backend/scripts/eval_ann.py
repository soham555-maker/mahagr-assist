"""
eval_ann.py — calibrate HNSW's `efSearch` against REAL query embeddings.

WHY THIS SCRIPT EXISTS
----------------------
HNSW is an APPROXIMATE index: it can miss a true nearest neighbour. How often it
misses depends on `efSearch`, and the honest way to set that is to measure
recall against exact brute-force search on the actual corpus with the actual
queries — not to trust a default.

Two ways to get this wrong, both of which happened while building this:

  1. **Synthetic vectors lie.** A 64-dimension random-vector benchmark reported
     recall@10 = 0.999 at ef=128. The same ef on the real 1024-d GR embeddings
     gave 0.735. Low-dimensional random data is far easier than real embeddings.
  2. **Random query vectors lie the other way.** A random 1024-d vector is
     nearly orthogonal to every corpus vector, so its "nearest neighbours" are
     almost arbitrary and nearly tied — ANN's worst case. Real questions land
     near real clusters, where the graph works much better.

So this script embeds the GOLD SET's real questions with the real model and
compares HNSW's results against an exact IndexFlatIP over the same vectors.

WHAT RECALL ACTUALLY MATTERS HERE
---------------------------------
Not recall@10. The pipeline hands `candidate_k_ann` (60) chunks to a fusion step
that ALSO receives BM25 candidates, then reranks. So the question is whether the
right chunk reaches the candidate pool at all — recall@60 — with BM25 as an
independent second path. Both are reported.

The economics that decide the answer: ANN search costs a fraction of a
millisecond against a ~5 s end-to-end answer. Recall is worth buying.

Usage:
    python scripts/eval_ann.py                       # gold questions, default index
    python scripts/eval_ann.py --k 60 --ef 64 128 256 512
"""

import argparse
import json
import os
import time

# pyrefly: ignore [missing-import]
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=None, help="index dir (default config.INDEX_DIR)")
    ap.add_argument("--gold", default="data/gold/gold.json")
    ap.add_argument("--k", type=int, nargs="+", default=[10, 60],
                    help="recall@k values to report (60 = the real candidate pool)")
    ap.add_argument("--ef", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024])
    args = ap.parse_args()

    # pyrefly: ignore [missing-import]
    import faiss
    from engine import config
    from engine.ingest import IngestionPipeline
    from engine.vector_store import HnswStore

    index_dir = args.index or config.INDEX_DIR
    store = HnswStore.load(index_dir)
    n = len(store)
    print(f"index      {index_dir}  ({n} vectors, dim {store.dim}, M={store.m})")

    questions = [q["q"] for q in json.load(open(args.gold, encoding="utf-8"))["questions"]]
    print(f"queries    {len(questions)} real gold-set questions")
    print(f"embedder   {config.EMBED_MODEL} on {config.EMBED_DEVICE or 'auto'}\n")

    model = IngestionPipeline().model
    q = np.asarray(model.encode(questions, normalize_embeddings=True,
                                convert_to_numpy=True), dtype="float32")

    # Exact ground truth. Reconstructing every vector costs RAM (n x 1024 x 4
    # bytes ~ 280 MB at 70k) but is the only honest baseline — and this is an
    # offline calibration script, not the serving path.
    print("building the exact baseline (brute force) ...")
    flat = faiss.IndexFlatIP(store.dim)
    flat.add(np.vstack([store.index.reconstruct(i) for i in range(n)]))
    maxk = max(args.k)
    t = time.time()
    _, truth = flat.search(q, maxk)
    exact_ms = (time.time() - t) / len(q) * 1000

    header = "efSearch  " + "  ".join(f"recall@{k:<4d}" for k in args.k) + "  ms/query"
    print("\n" + header)
    print("-" * len(header))
    for ef in args.ef:
        store.index.hnsw.efSearch = ef
        t = time.time()
        _, got = store.index.search(q, maxk)
        ms = (time.time() - t) / len(q) * 1000
        cells = []
        for k in args.k:
            rec = np.mean([len(set(g[:k]) & set(t_[:k])) / k
                           for g, t_ in zip(got, truth)])
            cells.append(f"{rec:>9.3f}   ")
        print(f"{ef:>8}  " + "".join(cells) + f"  {ms:>7.3f}")
    print(f"{'exact':>8}  " + "".join(f"{1.0:>9.3f}   " for _ in args.k)
          + f"  {exact_ms:>7.3f}   <- brute force, what HNSW replaced")

    print(f"\nCurrent config.HNSW_EF_SEARCH = {config.HNSW_EF_SEARCH}")
    print("efSearch is a QUERY-time dial — changing it needs no re-ingest.")


if __name__ == "__main__":
    main()
