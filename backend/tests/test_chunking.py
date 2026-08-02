"""Text splitting/chunking (the text-corpus ingest path)."""

from engine.ingest import IngestionPipeline

# no model needed for these methods; __new__ skips loading the embedder
pipe = IngestionPipeline.__new__(IngestionPipeline)


def test_split_pages_on_markers():
    pages = pipe._split_pages("# Page 1\nalpha beta\n# Page 2\ngamma delta")
    assert [p[0] for p in pages] == [1, 2]
    assert "alpha" in pages[0][1] and "gamma" in pages[1][1]


def test_split_pages_without_markers_is_single_page():
    pages = pipe._split_pages("just some text, no markers")
    assert len(pages) == 1 and pages[0][0] == 1


def test_chunk_pages_records_page_spans():
    pages = [(1, "word " * 300), (2, "term " * 300)]
    chunks = pipe.chunk_pages(pages, chunk_size=250, overlap=50)
    assert len(chunks) >= 2
    assert chunks[0]["page_start"] == 1
    # every chunk carries a page range and non-empty text
    assert all(c["text"] and c["page_start"] <= c["page_end"] for c in chunks)


def test_chunk_pages_overlap_must_be_smaller_than_size():
    import pytest
    with pytest.raises(ValueError):
        pipe.chunk_pages([(1, "a b c")], chunk_size=50, overlap=50)
