"""
reranker.py — the cross-encoder edge of hybrid retrieval (Phase 2).

A cross-encoder reads (query, chunk) TOGETHER in one forward pass, so its
attention can align query terms to chunk content directly — unlike the
bi-encoder (bge-small), which encodes each side separately into a vector and
never sees them in the same input. That makes it far more precise, and far
slower: nothing can be precomputed, so it only ever runs over the small fused
candidate pool the fast hybrid stage already produced (~15 chunks), never the
whole corpus.

WHY IT SHIPS (measured, 2026-07-10 — see docs/hybrid-search.md Step 2a):
the reorder is drastic (mean Kendall's tau 0.438; 62% of gold questions get a
different #1 chunk), and the refusal boundary transforms: worst in-corpus top
score -2.00 vs best out-of-corpus -9.92, a +7.92 gap where cosine had 0.02.
Cost: ~157ms/query on CPU over a 15-chunk pool.

SCORE SCALE: the cross-encoder outputs a raw relevance LOGIT, NOT a 0..1
cosine. Every threshold that reads hit["score"] after reranking must use the
recalibrated cutoff (RetrievalConfig.rerank_threshold), never the old cosine
numbers. The pre-rerank cosine is preserved on the hit as "dense_score".
NOTE: bge-reranker-v2-m3's logit scale differs from the old ms-marco model's,
so rerank_threshold was reset for it and still needs gold-set recalibration
(see retrieval.RetrievalConfig).

MULTILINGUAL: the model is bge-reranker-v2-m3 (config.RERANK_MODEL), which
reads Marathi/Hindi as well as English — a monolingual reranker would demote
the very Marathi chunks the multilingual retriever surfaced.

Both the corpus path (retrieval.Retriever) and the upload path
(documents.hybrid_search) rerank with the SAME instance, so their scores stay
commensurable and rag.merge_extra_chunks can keep merging by score.
"""

from engine import config

RERANK_MODEL = config.RERANK_MODEL  # bge-reranker-v2-m3 — multilingual cross-encoder


class Reranker:
    def __init__(self, model_name=RERANK_MODEL, device=None):
        """device=None uses config.EMBED_DEVICE (auto by default; set to 'cpu'
        to free the GPU for a local Ollama LLM). Same policy as the embedder."""
        # pyrefly: ignore [missing-import]
        from sentence_transformers import CrossEncoder
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device or config.EMBED_DEVICE)

    def rerank(self, query, hits):
        """Score every (query, hit text) pair together and return the hits
        re-sorted by that score, descending. Mutates each hit: 'score' becomes
        the rerank logit (the new primary ranking signal); the previous score
        (cosine) is preserved as 'dense_score'."""
        if not hits:
            return []
        scores = self.model.predict([(query, h["text"]) for h in hits])
        for h, s in zip(hits, scores):
            h.setdefault("dense_score", h["score"])
            h["score"] = float(s)
        return sorted(hits, key=lambda h: h["score"], reverse=True)
