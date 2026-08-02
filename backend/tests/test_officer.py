"""The supersession graph and document grouping — pure metadata logic."""

from engine import officer


class StubStore:
    """Minimal stand-in for FaissStore: just the two parallel sidecar lists the
    officer metadata functions read (no vectors needed)."""
    def __init__(self, metadata):
        self.metadata = metadata
        self.texts = [f"body of {m['order_id']} chunk {m['chunk_index']}" for m in metadata]


# GR-2024 supersedes GR-2023 and cites it.
DOCS = StubStore([
    {"order_id": "gr2023", "gr_number": "GR-2023/45", "date": "2023-06-15",
     "title": "Fee GR 2023", "chunk_index": 0},
    {"order_id": "gr2023", "gr_number": "GR-2023/45", "date": "2023-06-15",
     "title": "Fee GR 2023", "chunk_index": 1},
    {"order_id": "gr2024", "gr_number": "GR-2024/12", "date": "2024-04-10",
     "title": "Revised Fee GR 2024", "chunk_index": 0,
     "supersedes": True, "references": ["GR-2023/45"]},
])


def test_list_documents_dedupes_by_doc():
    docs = officer.list_documents(DOCS)
    assert set(docs.keys()) == {"gr2023", "gr2024"}


def test_document_chunks_returns_all_chunks_in_order():
    chunks = officer.document_chunks(DOCS, "GR-2023/45")
    assert len(chunks) == 2
    assert [c["metadata"]["chunk_index"] for c in chunks] == [0, 1]


def test_supersession_reports_superseded_by():
    info = officer.supersession(DOCS, "GR-2023/45")
    assert info["found"] is True
    assert len(info["superseded_by"]) == 1
    assert info["superseded_by"][0]["gr_number"] == "GR-2024/12"


def test_supersession_reports_cites_and_flag():
    info = officer.supersession(DOCS, "GR-2024/12")
    assert info["declares_supersession"] is True
    assert info["cites"][0]["gr_number"] == "GR-2023/45"
    assert info["cites"][0]["in_corpus"] == "gr2023"   # the cited GR IS in the corpus


def test_supersession_not_found():
    assert officer.supersession(DOCS, "GR-9999/99")["found"] is False
