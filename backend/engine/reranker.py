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

COST — WHERE THIS MODEL MUST RUN (measured 2026-08-06, 11-chunk pool of ~1.4k
chars each, RTX 4050):  CPU 27.6s/query  vs  GPU 0.33s/query (~80x).
The "~157ms on CPU" figure this file used to quote was for the ORIGINAL English
ms-marco MiniLM (22M params). bge-reranker-v2-m3 is XLM-R-large — 568M params,
~25x bigger — and Marathi costs several tokens per character, so CPU reranking
alone blew the SRS's <10s response target. Hence config.RERANK_DEVICE is
separate from config.EMBED_DEVICE: the embedder is happy on CPU (~0.1s for one
query), the cross-encoder is not. In fp16 it needs ~1.1 GB VRAM and so still
fits beside a 3B Ollama model on a 6 GB card.

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


def _is_cuda(device):
    """device may be None (= let sentence-transformers auto-pick). Only claim
    CUDA when it is actually going to be used, so a CPU-only box never asks for
    an fp16 load (fp16 on CPU is slower, not faster)."""
    if device is None:
        # pyrefly: ignore [missing-import]
        import torch
        return torch.cuda.is_available()
    return str(device).startswith("cuda")


class Reranker:
    def __init__(self, model_name=RERANK_MODEL, device=None, fp16=None):
        """device=None uses config.RERANK_DEVICE (auto = GPU when present). This
        does NOT follow config.EMBED_DEVICE: see the cost note above — the
        embedder can live on CPU, the cross-encoder cannot.

        fp16=None uses config.RERANK_FP16, and applies only on CUDA."""
        # pyrefly: ignore [missing-import]
        from sentence_transformers import CrossEncoder
        self.model_name = model_name
        device = device or config.RERANK_DEVICE
        want_fp16 = config.RERANK_FP16 if fp16 is None else fp16
        # Load STRAIGHT into fp16 rather than loading fp32 and calling .half():
        # the conversion leaves the discarded fp32 weights sitting in torch's
        # caching allocator, so the process held 2766 MiB for a 1083 MiB model.
        # Loading at the target dtype costs 1174 MiB — 1.6 GB back, which is the
        # difference between fitting an 8-bit LLM beside it on a 6 GB card or not.
        kwargs = ({"model_kwargs": {"torch_dtype": "float16"}}
                  if want_fp16 and _is_cuda(device) else {})
        self.model = CrossEncoder(model_name, device=device, **kwargs)
        self.device = str(getattr(self.model, "device", "cpu"))

    def rerank(self, query, hits):
        """Score every (query, hit text) pair together and return the hits
        re-sorted by that score, descending. Mutates each hit: 'score' becomes
        the rerank logit (the new primary ranking signal); the previous score
        (cosine) is preserved as 'dense_score'.

        BATCH SIZE IS A VRAM CEILING, NOT A SPEED KNOB. sentence-transformers
        defaults to 32 pairs at once; with ~1,900-character Marathi chunks and a
        3B LLM already resident on a 6 GB card, that allocation OOMs — and the
        OOM lands mid-request, not at startup. Batching in config.RERANK_BATCH
        makes peak VRAM independent of how deep the rerank pool is, so the pool
        can be tuned for accuracy without re-checking that it still fits.
        """
        if not hits:
            return []
        scores = self.model.predict([(query, h["text"]) for h in hits],
                                    batch_size=config.RERANK_BATCH)
        for h, s in zip(hits, scores):
            h.setdefault("dense_score", h["score"])
            h["score"] = float(s)
        return sorted(hits, key=lambda h: h["score"], reverse=True)
