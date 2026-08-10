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

...AND WHY HnswStore NOW EXISTS ANYWAY (PLAN Phase 2)
-----------------------------------------------------
That reasoning expired. The corpus went from 713 vectors to ~65,000 across six
departments, and the two properties that made FaissStore fine at 713 are exactly
what break at 65k: the search is O(n) over every vector, and every chunk's TEXT
is held in a RAM list rehydrated from a JSON sidecar at boot. HnswStore (below)
fixes both — O(log n) graph search, and NO text in the store at all (it lives in
SQLite, keyed by the very integer FAISS hands back). Both classes are kept:
FaissStore still backs the small fixture index and the model-free tests, and
config.VECTOR_BACKEND selects which one the app loads.
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


class HnswStore:
    """FAISS IndexHNSWFlat — the scaled vector index (PLAN Phase 2).

    WHAT HNSW IS, IN ONE PARAGRAPH
    ------------------------------
    Hierarchical Navigable Small World. The vectors are wired into a layered
    graph: the top layer is a sparse "express network" of a few nodes with long
    edges, each layer below is denser, and the bottom layer holds everything. A
    search enters at the top, greedily walks to the neighbour closest to the
    query, drops a layer, and repeats. It is the skip-list idea in vector space:
    long hops cover distance fast, short hops refine. Cost goes from O(n) —
    IndexFlatIP compares the query against all 65,000 vectors — to roughly
    O(log n). Measured on this machine at 5k vectors: 0.035 ms/query at 99.9%
    recall@10.

    THE TUNABLES, AND WHAT EACH ONE TRADES
    --------------------------------------
    * M (32) — edges kept per node. Higher = better recall and a bigger index
      (memory is ~ n*(dim*4 + M*8) bytes), and slower to build. 32 is the usual
      sweet spot for 1024-d.
    * efConstruction (200) — how hard the BUILD searches for good neighbours.
      Paid once, at ingest; raising it improves the graph forever after.
    * efSearch (128) — how many candidates a QUERY keeps in flight. This is the
      recall/latency dial and the only one changeable after the build.
      Measured here: ef=32 -> 0.898 recall, ef=64 -> 0.977, ef=128 -> 0.999.
      128 costs microseconds and is well inside a 10 s answer budget, so we
      spend it: this project's failure mode is a missed GR, not a slow one.

    APPROXIMATE, AND WHY THAT IS SAFE HERE
    --------------------------------------
    HNSW can miss a true neighbour — that is the "A" in ANN. It is acceptable
    because ANN is only stage one: its job is RECALL into a candidate pool that
    BM25 also feeds, and the cross-encoder then does the precise ranking. A
    chunk missed at ef=128 (1 in 1000) was almost never the one the reranker
    would have put first.

    NO TEXT LIVES HERE
    ------------------
    Unlike FaissStore this class has no `texts`/`metadata` lists. search()
    returns positions and scores; engine/corpus_db.py turns those into text. RAM
    stays flat no matter how large the corpus gets — that is the actual fix.
    """

    def __init__(self, dim=config.EMBED_DIM, m=None, ef_construction=None,
                 ef_search=None):
        self.dim = dim
        self.m = m or config.HNSW_M
        self.ef_construction = ef_construction or config.HNSW_EF_CONSTRUCTION
        self.ef_search = ef_search or config.HNSW_EF_SEARCH
        # METRIC_INNER_PRODUCT keeps the SAME cosine contract as FaissStore:
        # every vector is unit-normalized at encode time, so inner product IS
        # cosine and the calibrated thresholds carry over unchanged.
        self.index = faiss.IndexHNSWFlat(dim, self.m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        # Set by load()/the ingest script: where this index's chunk text lives.
        # officer.py reads it to tell the two backends apart (officer._corpus_db).
        self.corpus_db_path = None

    def __len__(self):
        return self.index.ntotal

    def add_vectors(self, vectors):
        """Append a batch of normalized vectors. Returns the faiss_id the batch
        STARTS at — the caller (scripts/ingest_corpus.py) uses it to key the
        matching gr_chunks rows, which is the whole id contract between the two
        files. FAISS assigns positions sequentially and never reuses them."""
        vectors = np.asarray(vectors, dtype='float32')
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(
                f"expected embeddings of shape (n, {self.dim}), got {vectors.shape}")
        start = self.index.ntotal
        self.index.add(vectors)
        return start

    def search(self, query_embedding, k=20, allowed_ids=None):
        """Top-k nearest chunks as [{'score', 'index'}], best first.

        allowed_ids (optional): restrict the search to these faiss_ids — the
        department/date/language filter, computed in SQLite by
        corpus_db.filter_faiss_ids.

        FAISS *can* do this natively (SearchParametersHNSW + IDSelectorBatch),
        which is better than the naive alternative of fetching top-k and
        filtering afterwards — a post-filter silently loses recall when every
        one of the top-k belongs to a department the user excluded. Be honest
        about the limit though: the graph is still built over ALL vectors, so a
        very selective filter makes the walk traverse mostly-rejected nodes and
        recall degrades. efSearch is raised when a filter is active to
        compensate. A store with native filtered indexes (Qdrant/pgvector) is
        the upgrade path if filters ever get that narrow.
        """
        if self.index.ntotal == 0:
            return []
        query = np.asarray(query_embedding, dtype='float32').reshape(1, self.dim)

        params = None
        if allowed_ids is not None:
            if not allowed_ids:
                return []
            ids = np.asarray(sorted(allowed_ids), dtype='int64')
            params = faiss.SearchParametersHNSW()
            params.sel = faiss.IDSelectorBatch(ids.size, faiss.swig_ptr(ids))
            # 4x the base efSearch when a filter is active, because the graph is
            # built over ALL vectors: a narrow filter makes the greedy walk spend
            # most of its budget on nodes the selector rejects, so the effective
            # candidate count is far below efSearch.
            #
            # THE CEILING MUST SCALE WITH THE BASE. This read
            # `min(4 * ef, 1024)` — written when ef was 128, so 4x512 sat well
            # under the cap. Once ef was raised to 1024 at corpus scale the
            # expression silently collapsed to max(1024, 1024) = 1024 and the
            # boost became a NO-OP, with nothing reporting it: filtered searches
            # quietly lost the compensation they were documented to get.
            # The absolute ceiling only exists to bound worst-case latency.
            params.efSearch = min(4 * self.ef_search, config.HNSW_EF_SEARCH_MAX)
            # Keep a reference alive: the SWIG wrapper does not own the selector
            # or the id array, and if Python frees either one FAISS reads freed
            # memory (a segfault, or worse, silently wrong hits).
            params._keepalive = (ids, params.sel)

        scores, indices = self.index.search(
            query, min(k, self.index.ntotal), params=params)
        return [{"score": float(s), "index": int(i)}
                for s, i in zip(scores[0], indices[0]) if i != -1]

    def score_for_index(self, idx, query_embedding):
        """Cosine of a chunk selected by something OTHER than dense search
        (BM25), so the hybrid fusion and the confidence floor still see a
        comparable score. Same contract as FaissStore.hit_for_index, minus the
        text — IndexHNSWFlat stores the full vectors, so reconstruct() is exact,
        not approximate."""
        query = np.asarray(query_embedding, dtype='float32').reshape(self.dim)
        return float(np.dot(self.index.reconstruct(int(idx)), query))

    def save(self, dir_path):
        """Persist the graph. Only ONE file — there is no text sidecar, because
        the text is in SQLite. Written to .tmp then renamed so an interrupted
        save leaves the last good index intact (ingestion checkpoints through
        this, and corpus_db.delete_chunks_from trusts the saved index as the
        authority on how far ingestion actually got)."""
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, "corpus.hnsw")
        faiss.write_index(self.index, path + ".tmp")
        os.replace(path + ".tmp", path)
        with open(os.path.join(dir_path, "hnsw_meta.json"), "w") as f:
            json.dump({"dim": self.dim, "M": self.m,
                       "efConstruction": self.ef_construction,
                       "efSearch": self.ef_search, "ntotal": self.index.ntotal}, f)
        return dir_path

    @classmethod
    def load(cls, dir_path, ef_search=None):
        """Reload a saved graph. efSearch is re-applied from config on load, on
        purpose: it is a QUERY-time dial, so it can be retuned without the hour
        of GPU time a rebuild would cost."""
        path = os.path.join(dir_path, "corpus.hnsw")
        meta_path = os.path.join(dir_path, "hnsw_meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)

        store = cls.__new__(cls)
        store.index = faiss.read_index(path)
        store.dim = store.index.d
        store.m = meta.get("M", config.HNSW_M)
        store.ef_construction = meta.get("efConstruction", config.HNSW_EF_CONSTRUCTION)
        store.ef_search = ef_search or config.HNSW_EF_SEARCH
        store.index.hnsw.efSearch = store.ef_search
        store.corpus_db_path = os.path.join(dir_path, "corpus.db")
        return store

    @classmethod
    def load_or_new(cls, dir_path, dim=config.EMBED_DIM):
        """Resume an interrupted ingest, or start a fresh one."""
        if os.path.exists(os.path.join(dir_path, "corpus.hnsw")):
            return cls.load(dir_path)
        store = cls(dim=dim)
        store.corpus_db_path = os.path.join(dir_path, "corpus.db")
        return store
