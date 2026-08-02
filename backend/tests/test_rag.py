"""Grounding helpers: citation parsing, budget trimming, block/citation formatting."""

from engine import rag


def _hit(text, score=1.0, **meta):
    base = {"content_type": "text", "page_start": 1, "page_end": 1}
    base.update(meta)
    return {"text": text, "score": score, "metadata": base}


def test_parse_citations_valid_and_phantom():
    valid, phantom = rag.parse_citations("As in [1] and [3], but [5] does not exist.", n_blocks=3)
    assert valid == [1, 3]
    assert phantom == [5]  # cited a block that was never sent — a groundedness alarm


def test_trim_to_budget_drops_lowest_scoring_tail():
    chunks = [_hit("word " * 100, score=0.9), _hit("word " * 100, score=0.5)]
    kept, dropped = rag.trim_to_budget(chunks, budget=140)  # ~1 chunk fits
    assert len(kept) == 1 and dropped == 1
    assert kept[0]["score"] == 0.9  # highest kept


def test_trim_to_budget_never_empties():
    kept, _ = rag.trim_to_budget([_hit("w " * 1000, score=0.9)], budget=1)
    assert len(kept) == 1  # an over-budget single chunk still goes through


def test_format_block_includes_gr_number_and_date():
    header = rag.format_block(1, _hit("body", gr_number="संकीर्ण-२०२४/x", date="2024-04-10", title="Fee GR"))
    assert header.startswith("[1]")
    assert "संकीर्ण-२०२४/x" in header and "2024-04-10" in header


def test_resolve_citations_carries_gr_fields():
    used = [_hit("body", gr_number="GR/1", date="2023-06-15", department="HTE", language="mr", title="T")]
    src = rag.resolve_citations([1], used)[0]
    assert src["gr_number"] == "GR/1" and src["date"] == "2023-06-15"
    assert src["department"] == "HTE" and src["n"] == 1
