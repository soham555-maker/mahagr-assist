"""Integration tests for chunk_text with text_table_detect."""

from engine.ingest import IngestionPipeline


def test_chunk_text_creates_both_prose_and_table_chunks():
    text = """\
# Page 1
Prose before table.
| H1 | H2 |
| d1 | d2 |
Prose after table.
"""
    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    chunks = pipeline.chunk_text(text)
    
    assert len(chunks) >= 2
    table_chunks = [c for c in chunks if c["content_type"] == "table"]
    prose_chunks = [c for c in chunks if c["content_type"] == "text"]
    
    assert len(table_chunks) == 1
    assert "In this row: H1 is d1, H2 is d2." in table_chunks[0]["text"]
    
    assert len(prose_chunks) >= 1
    assert "Prose before table." in prose_chunks[0]["text"]
    assert "Prose after table." in prose_chunks[-1]["text"]
    assert "| H1 |" not in prose_chunks[0]["text"]


def test_chunk_text_no_tables_creates_only_prose_chunks():
    text = """\
# Page 1
Prose only.
"""
    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    chunks = pipeline.chunk_text(text)
    
    assert len(chunks) == 1
    assert chunks[0]["content_type"] == "text"
    assert "Prose only." in chunks[0]["text"]
