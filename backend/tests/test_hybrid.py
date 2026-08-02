"""Tokenization (Devanagari-aware) and Reciprocal Rank Fusion."""

from engine import hybrid


def test_tokenize_ascii():
    assert hybrid.tokenize("DTE Admission 2023") == ["dte", "admission", "2023"]


def test_tokenize_keeps_whole_devanagari_words():
    # The real bug this guards: \w splits Marathi at matras (शासन -> श, सन).
    toks = hybrid.tokenize("शासन निर्णय क्रमांक")
    assert toks == ["शासन", "निर्णय", "क्रमांक"]


def test_tokenize_mixed_script():
    toks = hybrid.tokenize("इतर मागासवर्ग / OBC 6000")
    assert "मागासवर्ग" in toks and "obc" in toks and "6000" in toks
    # the slash is a separator, never part of a token
    assert all("/" not in t for t in toks)


def test_tokenize_empty():
    assert hybrid.tokenize("") == []


def test_rrf_fuse_ranks_items_both_lists_rank_high():
    # 'b' is rank 1 in BOTH lists -> unambiguously wins; 'a' (rank 2 in both) second.
    fused, scores = hybrid.rrf_fuse([["b", "a", "c"], ["b", "a", "d"]])
    assert fused[0] == "b"
    assert fused[1] == "a"
    assert set(fused) == {"a", "b", "c", "d"}
    assert scores["b"] > scores["a"] > scores["c"]


def test_rrf_fuse_single_list_preserves_order():
    fused, _ = hybrid.rrf_fuse([["x", "y", "z"]])
    assert fused == ["x", "y", "z"]
