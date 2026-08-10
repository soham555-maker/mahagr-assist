"""
Tests for retrieval.CorpusRetriever — the two-stage scale path.

Still model-free: a real HnswStore and a real SQLite corpus are built in a tmp
dir, and the embedding model is replaced by a stub that returns whatever vector
the test wants. That is enough to assert the parts that are actually ours — the
fusion, the group-by-GR cap, the filter push-down, and the abstention gate —
without downloading bge-m3.
"""

import numpy as np
import pytest

from engine import corpus_db
from engine.retrieval import CorpusRetriever, RetrievalConfig
from engine.vector_store import HnswStore

DIM = 16


class StubModel:
    """Stands in for bge-m3. encode() returns the vector the test pre-loaded,
    so a test can aim the query at a specific chunk."""

    def __init__(self):
        self.vec = np.zeros(DIM, dtype="float32")
        self.vec[0] = 1.0

    def encode(self, texts, **kw):
        return np.asarray([self.vec], dtype="float32")


class StubReranker:
    """Scores by a caller-supplied table keyed on chunk text, so a test can
    force a known post-rerank order and exercise the threshold gate."""

    def __init__(self, scores):
        self.scores = scores

    def rerank(self, query, hits):
        for h in hits:
            h.setdefault("dense_score", h["score"])
            h["score"] = self.scores.get(h["text"], 0.0)
        return sorted(hits, key=lambda h: h["score"], reverse=True)


def _unit(i):
    """A distinct unit vector per chunk; index 0 is closest to the query."""
    v = np.zeros(DIM, dtype="float32")
    v[0] = 1.0 - i * 0.02
    v[1 + (i % (DIM - 1))] = 0.2
    return v / np.linalg.norm(v)


@pytest.fixture()
def corpus(tmp_path):
    """Three GRs: 'a' has 5 chunks (a long GR), 'b' and 'c' have 1 each."""
    db_path = str(tmp_path / "corpus.db")
    corpus_db.init(db_path)
    store = HnswStore(dim=DIM, m=8, ef_construction=50, ef_search=64)

    layout = [("a", 5, "Higher and Technical Education Department", "2020-01-01", "mr"),
              ("b", 1, "Tribal Development Department", "2023-06-15", "mr"),
              ("c", 1, "School Education and Sports Department", "2015-03-03", "en")]

    vecs, rows, i = [], [], 0
    with corpus_db.connect(db_path) as conn:
        for doc_id, n, dept, date, lang in layout:
            corpus_db.upsert_document(
                conn, doc_id,
                {"gr_number": f"GR-{doc_id}", "department": dept, "date": date,
                 "language": lang, "title": f"title {doc_id}", "references": [],
                 "supersedes": False, "category": "government resolution (GR)"},
                f"full text {doc_id}", n)
            for j in range(n):
                vecs.append(_unit(i))
                rows.append((i, doc_id, j, 1, 1, "text", f"{doc_id} chunk {j} शिष्यवृत्ती"))
                i += 1
        corpus_db.insert_chunks(conn, rows)
    store.add_vectors(np.asarray(vecs, dtype="float32"))
    store.save(str(tmp_path))
    return store, db_path


def _retriever(corpus, cfg=None, reranker=None):
    store, db_path = corpus
    return CorpusRetriever(store, StubModel(), cfg or RetrievalConfig(),
                           reranker=reranker, db_path=db_path)


def test_hits_carry_text_and_document_metadata(corpus):
    """The whole point of the SQLite hydration step: the vector store holds no
    text, yet a hit must still look exactly like the flat backend's."""
    res = _retriever(corpus).retrieve("शिष्यवृत्ती")
    assert res["chunks"]
    h = res["chunks"][0]
    assert h["text"] and "chunk" in h["text"]
    assert h["metadata"]["order_id"] in {"a", "b", "c"}
    assert h["metadata"]["department"]
    assert set(h) == {"score", "text", "metadata", "index"}


def test_one_long_gr_cannot_monopolise_the_pool(corpus):
    """GR 'a' owns 5 of the 7 chunks and the 5 nearest vectors. Without the
    per-GR cap it would fill the rerank pool and the answer would cite one
    document when three were relevant."""
    cfg = RetrievalConfig(max_chunks_per_gr=2)
    res = _retriever(corpus, cfg).retrieve("शिष्यवृत्ती")
    from collections import Counter
    per_gr = Counter(h["metadata"]["order_id"] for h in res["chunks"])
    assert per_gr["a"] <= 2
    assert len(per_gr) >= 2, "the pool should span several GRs, not just the longest"


def test_raising_the_cap_lets_more_chunks_through(corpus):
    cfg = RetrievalConfig(max_chunks_per_gr=5)
    res = _retriever(corpus, cfg).retrieve("शिष्यवृत्ती")
    from collections import Counter
    assert Counter(h["metadata"]["order_id"] for h in res["chunks"])["a"] > 2


def test_department_filter_restricts_results(corpus):
    res = _retriever(corpus).retrieve(
        "शिष्यवृत्ती", filters={"departments": ["Tribal Development Department"]})
    assert res["chunks"]
    assert {h["metadata"]["order_id"] for h in res["chunks"]} == {"b"}


def test_date_and_language_filters(corpus):
    r = _retriever(corpus)
    assert {h["metadata"]["order_id"]
            for h in r.retrieve("शिष्यवृत्ती", filters={"date_from": "2022-01-01"})["chunks"]} == {"b"}
    assert {h["metadata"]["order_id"]
            for h in r.retrieve("शिष्यवृत्ती", filters={"date_to": "2016-01-01"})["chunks"]} == {"c"}
    assert {h["metadata"]["order_id"]
            for h in r.retrieve("शिष्यवृत्ती", filters={"language": "en"})["chunks"]} == {"c"}


def test_a_filter_matching_nothing_abstains(corpus):
    """It must NOT quietly fall back to the whole corpus — that would answer
    from GRs the officer deliberately excluded."""
    res = _retriever(corpus).retrieve("शिष्यवृत्ती",
                                      filters={"departments": ["No Such Department"]})
    assert res["chunks"] == []
    assert res["low_confidence"] is True


def test_bm25_only_chunks_still_get_a_comparable_score(corpus):
    """A chunk BM25 finds but ANN misses is scored by reconstructing its vector,
    so the calibrated thresholds still mean what they were calibrated to mean."""
    res = _retriever(corpus).retrieve("शिष्यवृत्ती")
    assert all(-1.0001 <= h["score"] <= 1.0001 for h in res["chunks"])


def test_reranker_threshold_gate_and_floor(corpus):
    """Above the threshold -> confident. Nothing above it -> keep the floor and
    flag low_confidence, which is what lets the prompt refuse."""
    cfg = RetrievalConfig(rerank_threshold=0.85, floor_k=2)

    good = StubReranker({t: 0.99 for t in
                         [f"a chunk {j} शिष्यवृत्ती" for j in range(5)] +
                         ["b chunk 0 शिष्यवृत्ती", "c chunk 0 शिष्यवृत्ती"]})
    res = _retriever(corpus, cfg, good).retrieve("शिष्यवृत्ती")
    assert res["low_confidence"] is False and res["chunks"]

    weak = StubReranker({})          # everything scores 0.0, well under 0.85
    res = _retriever(corpus, cfg, weak).retrieve("शिष्यवृत्ती")
    assert res["low_confidence"] is True
    assert len(res["chunks"]) == cfg.floor_k


def test_missing_chunk_rows_are_skipped_not_fatal(corpus):
    """If the DB and index ever drift, retrieval degrades rather than 500s."""
    store, db_path = corpus
    with corpus_db.connect(db_path) as conn:
        conn.execute("DELETE FROM gr_chunks WHERE faiss_id = 0")
    res = CorpusRetriever(store, StubModel(), RetrievalConfig(),
                          db_path=db_path).retrieve("शिष्यवृत्ती")
    assert all(h["index"] != 0 for h in res["chunks"])
