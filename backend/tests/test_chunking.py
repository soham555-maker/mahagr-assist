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


# --------------------------------------------------------------------------- #
# EMBED_MAX_SEQ — an OOM guard, written after a crash 15,000 documents into a
# 23,000-document ingest. See config.EMBED_MAX_SEQ.
# --------------------------------------------------------------------------- #

class _FakeModel:
    """Stands in for a loaded SentenceTransformer. Only max_seq_length matters."""
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length


def test_embed_max_seq_caps_a_models_default_window():
    """bge-m3 reports max_seq_length=8192. Attention memory is QUADRATIC in
    sequence length and a batch pads to its longest member, so leaving 8192 in
    place makes peak VRAM a function of the worst single chunk in the corpus —
    which is how a 6 GB card died asking for a 1.76 GiB attention tensor.
    """
    from engine import config
    from engine.ingest import IngestionPipeline

    p = IngestionPipeline(model=_FakeModel(8192))
    assert p.model.max_seq_length == config.EMBED_MAX_SEQ
    assert config.EMBED_MAX_SEQ <= 2048, "the cap must actually bound attention"


def test_embed_max_seq_never_raises_a_smaller_window():
    """min(), not assignment: a model that natively supports only 512 tokens
    must not be told it can take 1024 — that would silently truncate nothing
    and mislead about what was embedded."""
    p = IngestionPipeline(model=_FakeModel(512))
    assert p.model.max_seq_length == 512


def test_embed_max_seq_applies_to_a_shared_model():
    """The cap is a property of what is safe to embed on this hardware, not of
    who constructed the object. The API process hands its already-loaded model
    to the pipeline, and that path must be capped too."""
    shared = _FakeModel(8192)
    IngestionPipeline(model=shared)
    assert shared.max_seq_length < 8192
