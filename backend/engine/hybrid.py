"""
hybrid.py — the shared, dependency-light pieces of hybrid retrieval:
tokenization for BM25 and Reciprocal Rank Fusion (RRF). Both are PURE
functions (no model, no I/O), so they're asserted directly in test_hybrid.py,
and both the corpus path (retrieval.py) and the upload path (documents.py)
import them so there's exactly one implementation of each.

WHY RRF INSTEAD OF ADDING SCORES
---------------------------------
Dense (cosine, 0..1) and sparse (BM25, unbounded ~0..20+) scores live on
incompatible scales — adding them lets BM25's magnitude swamp cosine. RRF
sidesteps this by fusing RANK POSITIONS, not scores: an item's fused score is
sum over the lists it appears in of 1/(k + rank). Items both methods rank high
rise to the top; neither scale distorts the other; there's nothing to tune
(k=60 is the standard constant). Same "commensurable-or-don't-mix" discipline
retrieval.py already uses when it merges text/table hits by raw cosine (those
ARE the same scale) — here the scales differ, so we fuse ranks instead.
"""

import re
from collections import defaultdict

RRF_K = 60  # standard RRF constant; larger = flatter rank weighting


def tokenize(text):
    """Lowercase word tokens for BM25. Deliberately simple: no stemming/stopword
    list — BM25's IDF term already down-weights common words, so a query token
    like 'the' contributes almost nothing regardless.

    MULTILINGUAL: matches Latin/digit runs AND the full Devanagari block
    (U+0900-U+097F), so Marathi/Hindi words in Government Resolutions tokenize
    as whole words. Two traps this avoids: the old ASCII [a-z0-9] dropped every
    Marathi word outright; and a plain \\w+ is worse-than-useless here because
    \\w excludes Devanagari combining vowel signs (matras) and the virama, so it
    SHATTERS each word at every matra ("शासन" -> "श","सन"). Including the whole
    block keeps a word contiguous. (Extend the class with more script ranges if
    other languages are ever added.)"""
    return re.findall(r"[a-z0-9ऀ-ॿ]+", text.lower())


def rrf_fuse(ranked_lists, k=RRF_K):
    """
    Reciprocal Rank Fusion over several ranked lists of keys.

    ranked_lists: list of lists, each ordered best-first, of hashable keys
                  (here: chunk indices/positions). A key may appear in some
                  lists and not others.
    Returns (fused_keys, scores): fused_keys is every key seen, ordered by
    descending fused score; scores maps key -> fused score. Rank is 1-based,
    so the #1 item in a list contributes 1/(k+1).
    """
    scores = defaultdict(float)
    for lst in ranked_lists:
        for rank, key in enumerate(lst, start=1):
            scores[key] += 1.0 / (k + rank)
    fused_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return fused_keys, dict(scores)
