"""
FaissStore — the one place the whole project talks to the FAISS index.

FAISS stores vectors and nothing else: it answers "give me the k closest vectors
to this one" and hands back positions (0, 1, 2, ...), not text or paper IDs. So a
usable store is FAISS *plus* two parallel Python lists — texts[i] and metadata[i]
— kept in lockstep with the vector at position i in the index. This class owns that
pairing so RAG retrieval (Days 4-5), chat memory (Days 6-7), and plagiarism search
(Days 9-10) all share one add/search/save/load API instead of three divergent ones.

COSINE CONTRACT
---------------
The index is IndexFlatIP (exact, brute-force inner-product search). On vectors that
are already L2-normalized to unit length, an inner product IS cosine similarity, so
search scores read directly as cosine in [-1, 1], highest first. This store assumes
every embedding handed to add() and every query handed to search() is already
normalized — which is exactly what ingest.py produces (encode(..., normalize_
embeddings=True)). Normalization lives at that one source, not here, so there is a
single place to reason about it. Feed it un-normalized vectors and the scores stop
being cosine (ranking still holds, absolute values don't) — that's why the invariant
is documented rather than silently re-enforced.

WHY FLAT, NOT IVF/HNSW
----------------------
A 30-50 paper corpus is a few thousand vectors. Brute-force search over that is
sub-millisecond, so an approximate index (IVF/HNSW) would trade recall for a speed
win we don't need. Same reasoning the plan uses to skip MinHash/LSH: don't add
approximation to solve a scale problem that isn't there.
"""

import json
import os

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import faiss

from engine import config


class FaissStore:
    def __init__(self, dim=config.EMBED_DIM):
        """
        An empty inner-product index of the given embedding dimension, plus the two
        sidecar lists that give each stored vector its text and metadata back.
        dim must match the embedding model's output (config.EMBED_DIM; bge-m3 = 1024).
        An index built at the wrong dim raises on the first add(), not silently.
        """
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.texts = []
        self.metadata = []

    def __len__(self):
        """Number of vectors stored. Kept equal to len(texts) == len(metadata)."""
        return self.index.ntotal

    def add(self, chunks):
        """
        Add pipeline output to the index. chunks is a list of the dicts
        process_pdf returns: [{'text', 'embedding', 'metadata'}, ...].

        Vectors go into FAISS; text and metadata go into the parallel lists at the
        same positions. Because FAISS is append-only here (nothing is ever deleted),
        position i is stable for the life of the store, so the lists stay aligned.
        """
        if not chunks:
            return

        # FAISS is a C++ library under a thin wrapper: it wants one contiguous
        # float32 (n, dim) array, not a list of Python lists or float64 vectors.
        vectors = np.asarray([c['embedding'] for c in chunks], dtype='float32')
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(
                f"expected embeddings of shape (n, {self.dim}), got {vectors.shape}"
            )

        self.index.add(vectors)
        self.texts.extend(c['text'] for c in chunks)
        self.metadata.extend(c['metadata'] for c in chunks)

        # The invariant everything downstream trusts. Assert it here, where a drift
        # is introduced, not later where it would surface as mismatched provenance.
        assert self.index.ntotal == len(self.texts) == len(self.metadata)

    def search(self, query_embedding, k=5, where=None):
        """
        Return the k most cosine-similar stored chunks to query_embedding.

        query_embedding is a single normalized vector (list or 1-D array of length
        dim). Returns a list of {'score', 'text', 'metadata'} dicts, highest score
        first, where score is cosine similarity in [-1, 1]. k is clamped to the
        number of stored vectors, so asking for more than exist just returns all.

        where (optional) is a predicate over a chunk's metadata dict — e.g.
        lambda m: m['content_type'] == 'table' — letting retrieval search one
        modality at a time (Days 4-5). FAISS itself knows nothing about metadata,
        so this is a post-filter: we ask FAISS for the FULL ranked list and keep
        only matching hits, truncated to k. That sidesteps the classic post-filter
        trap (fetch top-10, filter, end up with 2 results while better matches sat
        at rank 11+): filtering the complete ranking loses no recall. It's free
        here because a Flat index already scans every vector per query anyway —
        the same no-machinery-at-this-scale call as flat-over-IVF. At a scale
        where full scans hurt, the upgrade path is FAISS IDSelector pre-filtering
        or per-modality indexes.
        """
        if self.index.ntotal == 0:
            return []

        query = np.asarray(query_embedding, dtype='float32').reshape(1, self.dim)
        fetch_k = self.index.ntotal if where is not None else min(k, self.index.ntotal)
        scores, indices = self.index.search(query, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS pads with -1 if it can't fill k; skip those
                continue
            if where is not None and not where(self.metadata[idx]):
                continue
            results.append({
                'score': float(score),
                'text': self.texts[idx],
                'metadata': self.metadata[idx],
                'index': int(idx),   # position in the store; lets hybrid fuse dense+BM25 by chunk
            })
            if len(results) == k:
                break
        return results

    def hit_for_index(self, idx, query_embedding):
        """Build a hit dict for a chunk selected by SOMETHING OTHER than dense
        search (BM25), computing its cosine against the query so the retrieval
        contract's threshold/floor still has a comparable score. The stored
        vector is exact (IndexFlatIP.reconstruct); both it and the query are
        unit-normalized, so the dot product is cosine — same contract as
        search()."""
        query = np.asarray(query_embedding, dtype='float32').reshape(self.dim)
        vec = self.index.reconstruct(int(idx))
        return {
            'score': float(np.dot(vec, query)),
            'text': self.texts[idx],
            'metadata': self.metadata[idx],
            'index': int(idx),
        }

    def save(self, dir_path):
        """
        Persist the index and its sidecar to dir_path/ (created if missing).

        Two files: corpus.index (the FAISS binary) and corpus_meta.json (dim +
        parallel texts + metadata). Each is written to a .tmp file and then renamed,
        so a crash mid-write leaves the previous good copy intact rather than a
        half-written, length-mismatched pair. Embeddings live only in the FAISS
        file, so the JSON stays small and cleanly serializable.
        """
        os.makedirs(dir_path, exist_ok=True)

        index_path = os.path.join(dir_path, "corpus.index")
        tmp_index = index_path + ".tmp"
        faiss.write_index(self.index, tmp_index)
        os.replace(tmp_index, index_path)  # atomic on the same filesystem

        meta_path = os.path.join(dir_path, "corpus_meta.json")
        tmp_meta = meta_path + ".tmp"
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(
                {"dim": self.dim, "texts": self.texts, "metadata": self.metadata},
                f, ensure_ascii=False,
            )
        os.replace(tmp_meta, meta_path)

        return dir_path

    @classmethod
    def load(cls, dir_path):
        """
        Reload a store saved by save(). Rebuilds the object, then asserts the
        index/text/metadata lengths still agree before handing it back — a
        corrupted or mismatched save is caught here, not at first query.
        """
        index_path = os.path.join(dir_path, "corpus.index")
        meta_path = os.path.join(dir_path, "corpus_meta.json")

        with open(meta_path, "r", encoding="utf-8") as f:
            sidecar = json.load(f)

        store = cls.__new__(cls)  # skip __init__: we're restoring, not starting fresh
        store.dim = sidecar["dim"]
        store.index = faiss.read_index(index_path)
        store.texts = sidecar["texts"]
        store.metadata = sidecar["metadata"]

        assert store.index.ntotal == len(store.texts) == len(store.metadata), \
            "loaded index and sidecar are out of sync — the save is corrupt"
        return store
