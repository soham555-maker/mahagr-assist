"""
Tests for engine/corpus_db.py — the SQLite half of the scaled corpus.

Model-free like the rest of the suite: SQLite is stdlib, so all of this runs in
milliseconds with no torch, no GPU and no network.
"""

import os

import pytest

from engine import corpus_db


@pytest.fixture()
def conn(tmp_path):
    path = str(tmp_path / "corpus.db")
    corpus_db.init(path)
    with corpus_db.connect(path) as c:
        yield c


def _doc(conn, doc_id, **kw):
    meta = {"gr_number": kw.get("gr_number", f"GR-{doc_id}"),
            "department": kw.get("department", "Higher and Technical Education Department"),
            "date": kw.get("date", "2023-06-15"),
            "category": "government resolution (GR)",
            "language": kw.get("language", "mr"),
            "title": kw.get("title", "शिष्यवृत्ती"),
            "references": kw.get("references", []),
            "supersedes": kw.get("supersedes", False)}
    corpus_db.upsert_document(conn, doc_id, meta, kw.get("text", "body"),
                              kw.get("n_chunks", 1))


def _chunks(conn, rows):
    corpus_db.insert_chunks(conn, rows)


def test_chunk_lookup_returns_document_metadata(conn):
    _doc(conn, "20230615", gr_number="संकीर्ण-२०२३/४५")
    _chunks(conn, [(0, "20230615", 0, 1, 1, "text", "अभियांत्रिकी प्रवेश शुल्क")])

    got = corpus_db.chunks_by_faiss_ids(conn, [0])
    assert got[0]["text"] == "अभियांत्रिकी प्रवेश शुल्क"
    m = got[0]["metadata"]
    assert m["order_id"] == "20230615"
    assert m["gr_number"] == "संकीर्ण-२०२३/४५"
    assert m["department"] == "Higher and Technical Education Department"


def test_missing_faiss_ids_are_simply_absent(conn):
    """A drifted id must not raise — retrieval skips it rather than 500-ing."""
    _doc(conn, "d1")
    _chunks(conn, [(0, "d1", 0, 1, 1, "text", "x")])
    assert corpus_db.chunks_by_faiss_ids(conn, [0, 99]).keys() == {0}
    assert corpus_db.chunks_by_faiss_ids(conn, []) == {}


def test_bm25_matches_whole_devanagari_words(conn):
    """The Devanagari tokenizer bug (HANDOFF §5.4) in its FTS5 form: a Marathi
    word must match as a whole word, not be split at its vowel marks."""
    _doc(conn, "d1")
    _doc(conn, "d2")
    _chunks(conn, [
        (0, "d1", 0, 1, 1, "text", "शासन निर्णय क्रमांक शिष्यवृत्ती योजना"),
        (1, "d2", 0, 1, 1, "text", "admission fee structure for engineering"),
    ])
    assert [i for i, _ in corpus_db.search_bm25(conn, "शिष्यवृत्ती", 5)] == [0]
    assert [i for i, _ in corpus_db.search_bm25(conn, "engineering", 5)] == [1]


def test_bm25_scores_are_higher_is_better(conn):
    """FTS5's bm25() is negative-is-better; corpus_db negates it so it matches
    every other score in the codebase. Guard against that flipping back."""
    _doc(conn, "d1")
    _chunks(conn, [(0, "d1", 0, 1, 1, "text", "शिष्यवृत्ती शिष्यवृत्ती")])
    scores = [s for _, s in corpus_db.search_bm25(conn, "शिष्यवृत्ती", 5)]
    assert scores and all(s > 0 for s in scores)


def test_bm25_survives_fts5_operator_characters(conn):
    """A raw GR number contains '-', '/' and '.', all FTS5 MATCH operators.
    Passing one through unquoted raises 'fts5: syntax error'."""
    _doc(conn, "d1")
    _chunks(conn, [(0, "d1", 0, 1, 1, "text", "संकीर्ण-२०२३/प्र.क्र.४५/तांशि-४ नुसार")])
    for q in ["संकीर्ण-२०२३/प्र.क्र.४५", 'a "quoted" thing', "OR AND NOT", "*", "((("]:
        corpus_db.search_bm25(conn, q, 5)          # must not raise
    assert corpus_db.search_bm25(conn, "!!!", 5) == []   # no tokens -> no hits


def test_filter_faiss_ids_by_department_date_and_language(conn):
    _doc(conn, "d1", department="Tribal Development Department", date="2019-01-01")
    _doc(conn, "d2", department="Higher and Technical Education Department",
         date="2024-01-01", language="en")
    _chunks(conn, [(0, "d1", 0, 1, 1, "text", "a"), (1, "d2", 0, 1, 1, "text", "b")])

    assert corpus_db.filter_faiss_ids(conn) is None          # no facets -> no filtering
    assert corpus_db.filter_faiss_ids(
        conn, departments=["Tribal Development Department"]) == {0}
    assert corpus_db.filter_faiss_ids(conn, date_from="2020-01-01") == {1}
    assert corpus_db.filter_faiss_ids(conn, date_to="2020-01-01") == {0}
    assert corpus_db.filter_faiss_ids(conn, language="en") == {1}
    assert corpus_db.filter_faiss_ids(
        conn, departments=["Tribal Development Department"], language="en") == set()


def test_reingesting_a_document_does_not_duplicate_it(conn):
    """Idempotence: the ingest script re-runs over the whole corpus."""
    _doc(conn, "d1", title="first")
    _doc(conn, "d1", title="second")
    assert corpus_db.stats(conn)["documents"] == 1
    assert corpus_db.get_document(conn, "d1")["title"] == "second"


def test_delete_chunks_from_is_crash_recovery(conn):
    """The saved FAISS index is the authority. If SQLite committed chunk rows
    whose vectors never reached disk, they must be dropped — otherwise every
    later chunk is off by a few positions and every citation names the wrong GR
    while looking entirely plausible."""
    _doc(conn, "d1")
    _chunks(conn, [(i, "d1", i, 1, 1, "text", f"chunk {i}") for i in range(5)])

    assert corpus_db.delete_chunks_from(conn, 3) == 2       # ids 3,4 dropped
    assert corpus_db.stats(conn)["chunks"] == 3
    # the FTS index must be cleaned up in lockstep, or a stale row would still
    # be returned by BM25 and then fail to hydrate
    assert all(i < 3 for i, _ in corpus_db.search_bm25(conn, "chunk", 10))
    assert corpus_db.delete_chunks_from(conn, 3) == 0       # already clean


def test_ingested_ids_drives_resume(conn):
    _doc(conn, "d1")
    _doc(conn, "d2")
    assert corpus_db.ingested_ids(conn) == {"d1", "d2"}


def test_find_document_prefers_an_exact_id(conn):
    """'2023' must not resolve to some longer id that merely contains it when a
    document with exactly that id exists."""
    _doc(conn, "2023")
    _doc(conn, "20231231")
    assert corpus_db.find_document(conn, "2023")["id"] == "2023"
    assert corpus_db.find_document(conn, "20231231")["id"] == "20231231"
    assert corpus_db.find_document(conn, "nope") is None


def test_superseding_documents_requires_an_exact_reference(conn):
    """The SQL prefilter uses LIKE over a JSON list, so 'GR-12' would match
    'GR-123'. The exact re-check in Python is what makes it correct."""
    _doc(conn, "old", gr_number="GR-12")
    _doc(conn, "other", gr_number="GR-123")
    _doc(conn, "new", gr_number="GR-99", references=["GR-12"], supersedes=True)
    _doc(conn, "cites-only", gr_number="GR-98", references=["GR-12"], supersedes=False)

    sup = corpus_db.superseding_documents(conn, "GR-12")
    assert [s["gr_number"] for s in sup] == ["GR-99"]     # not GR-98 (no supersede flag)
    assert corpus_db.superseding_documents(conn, "GR-123") == []
    assert corpus_db.superseding_documents(conn, None) == []


def test_document_chunks_are_in_reading_order(conn):
    _doc(conn, "d1")
    _chunks(conn, [(2, "d1", 1, 1, 1, "text", "second"),
                   (1, "d1", 0, 1, 1, "text", "first")])
    hits = corpus_db.document_chunks(conn, "d1")
    assert [h["text"] for h in hits] == ["first", "second"]
    assert hits[0]["metadata"]["order_id"] == "d1"


def test_stats_reports_corpus_shape(conn):
    _doc(conn, "d1", department="A", date="2019-01-01")
    _doc(conn, "d2", department="B", date="2024-05-05")
    _chunks(conn, [(0, "d1", 0, 1, 1, "text", "a"), (1, "d2", 0, 1, 1, "text", "b")])
    st = corpus_db.stats(conn)
    assert st["documents"] == 2 and st["chunks"] == 2
    assert {d["name"] for d in st["departments"]} == {"A", "B"}
    assert st["date_from"] == "2019-01-01" and st["date_to"] == "2024-05-05"


def test_db_file_is_created_on_init(tmp_path):
    path = str(tmp_path / "nested" / "corpus.db")
    corpus_db.init(path)
    assert os.path.exists(path)
