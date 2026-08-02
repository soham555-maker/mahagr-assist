"""Guards on the calibrated retrieval config and the multilingual model choices."""

from engine import config
from engine.retrieval import RetrievalConfig


def test_multilingual_model_choices():
    assert config.EMBED_MODEL == "BAAI/bge-m3"
    assert config.EMBED_DIM == 1024                 # bge-m3 dense width
    assert config.RERANK_MODEL == "BAAI/bge-reranker-v2-m3"


def test_bge_m3_uses_no_query_prefix():
    # bge-m3 takes raw query text; a stale bge-*-en prefix would silently degrade it.
    assert RetrievalConfig().query_prefix == ""


def test_thresholds_are_calibrated_not_placeholders():
    c = RetrievalConfig()
    # rerank_threshold must be > 0 — the 0.0 placeholder never abstained (OOC bug).
    assert 0.0 < c.rerank_threshold < 1.0
    # cosine cutoff sits in the measured keep/reject gap (~0.55).
    assert 0.5 <= c.text_threshold <= 0.65
