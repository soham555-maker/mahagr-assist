"""
config.py — the ONE place the multilingual model choices live.

The whole point of MahaGR Assist over a plain English RAG is that Marathi and
English (and Hindi) documents live in the SAME vector space, so an English
query can retrieve a Marathi Government Resolution and vice-versa. That comes
down to three coupled choices — the embedding model, its output dimension, and
its query convention — plus a matching multilingual reranker. Keeping them here
(instead of hard-coded in ingest.py / vector_store.py / reranker.py) means a
future model swap is a one-file change and the four settings can never drift
out of sync (a mismatched dim or a stale query prefix fails silently — worse
retrieval, no error).

WHAT CHANGED FROM THE ENGLISH BASE (and why)
--------------------------------------------
* EMBED_MODEL  bge-small-en-v1.5 -> bge-m3.
  bge-m3 is multilingual (100+ languages incl. Marathi/Hindi) and one of the
  strongest open retrievers for Indic text. It maps every language into one
  shared space, which is exactly what cross-lingual retrieval needs.
* EMBED_DIM    384 -> 1024.  bge-m3's dense output width. FaissStore is built
  at this dim; an index built at the wrong dim raises on the first add().
* QUERY_PREFIX "Represent this sentence..." -> "" (none).
  bge-*-en-v1.5 was fine-tuned to expect an instruction prefix on the QUERY
  side. bge-m3 was NOT — it takes the raw text on both sides. Carrying the old
  prefix over would quietly degrade every query, so it is emptied here.
* RERANK_MODEL ms-marco-MiniLM (English) -> bge-reranker-v2-m3 (multilingual).
  The cross-encoder must understand Marathi too, or it would demote the very
  Marathi chunks the multilingual retriever surfaced.

OCR (scanned Government Resolutions)
------------------------------------
Many GRs exist only as scanned image PDFs with no text layer. OCR_LANGS is the
Tesseract language string used to read them (see ingest.extract_pages_from_pdf).
Requires the system `tesseract-ocr` package plus the `mar`, `hin`, `eng`
language data files installed.

WHERE .env IS LOADED (and why here)
-----------------------------------
Configuration comes from backend/.env, loaded HERE rather than in rag.py.
config.py is the first engine module every import path touches, and the settings
below are read from os.environ at import time — so if .env were loaded later
(rag.py used to be the only loader) EMBED_DEVICE would already have been read as
None and silently ignored, putting bge-m3 on the GPU next to the local LLM. The
path is derived from __file__ so it resolves from any working directory, and
load_dotenv never overrides a real environment variable.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# --- retrieval / embedding (the coupled four) ---
EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024
QUERY_PREFIX = ""            # bge-m3 needs no instruction prefix (unlike bge-*-en-v1.5)
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# Device for the EMBEDDER (bi-encoder). None = auto (GPU if available, else CPU).
# Set EMBED_DEVICE=cpu when a local LLM (Ollama) shares the same small GPU: at
# serve time the embedder encodes ONE short query per request, which costs ~0.1s
# on CPU, so the VRAM is worth more to the LLM. During bulk ingestion leave it
# unset (and stop Ollama) so the whole corpus is embedded on the GPU.
EMBED_DEVICE = os.environ.get("EMBED_DEVICE") or None

# Load the EMBEDDER in half precision on CUDA. Same technique (and the same
# "load fp16, never .half() afterwards" rule) as RERANK_FP16 below.
# MEASURED on 512 real Marathi GR chunks (avg 1570 chars, RTX 4050):
#     float32  batch 8    9.8 chunks/s   peak 2.62 GB
#     float16  batch 16  36.6 chunks/s   peak 1.49 GB   -> 3.7x faster, 43% VRAM
# Accuracy cost, measured on the same chunks: cosine agreement between the fp32
# and fp16 embedding of the SAME chunk is min 0.99975 / mean 1.00008 — far below
# the score gaps retrieval actually discriminates on (~0.02-0.05), and the
# cross-encoder does the precise ranking afterwards anyway. That 3.7x is the
# difference between a ~2 hour and a ~30 minute corpus build.
# Ignored on CPU, where fp16 is slower rather than faster.
EMBED_FP16 = (os.environ.get("EMBED_FP16", "1").lower()
              not in ("0", "false", "no"))

# Hard cap on the embedder's input length. THIS IS AN OOM GUARD, not a tuning
# knob, and it was written after a crash 15,000 documents into a 23,000-document
# ingest.
#
# bge-m3 advertises an 8192-token window and sentence-transformers adopts that
# as max_seq_length by default. Attention memory grows with the SQUARE of the
# sequence length, and a batch is padded to its LONGEST member — so a single
# pathological chunk sets the cost of the whole batch. One ~1,900-token chunk in
# a batch of 16 asked for a 1.76 GiB attention tensor on a 6 GB card that was
# already holding the model, and the run died. Batch size was not the problem:
# it was 16, the measured-safe value, and had already embedded 125,000 chunks.
#
# The cap is nearly free here because chunking already bounds the input:
# chunk_size is 250 WORDS, which measures p50 494 / p95 662 / p99 768 tokens
# across the corpus. Only 92 of 129,320 chunks (0.071%) exceed 1024, and exactly
# one exceeds 4096.
#
# What truncation costs, stated honestly: for those 92 chunks the EMBEDDING is
# computed from the first 1024 tokens. The full text is still stored in SQLite,
# still returned, still cited and still read by the cross-encoder — so nothing
# is lost from an answer. Only the vector for the tail of a very long chunk is
# less representative. That is a far better trade than a non-deterministic OOM
# in the middle of a multi-hour GPU job.
EMBED_MAX_SEQ = int(os.environ.get("EMBED_MAX_SEQ", "1024"))

# Device for the RERANKER (cross-encoder) — deliberately SEPARATE from
# EMBED_DEVICE, because the two models have opposite cost profiles even though
# they are the same size (~568M params):
#   embedder    1 short query, precomputable corpus side  ->   0.1s on CPU
#   reranker    15 (query, chunk) pairs of ~1.4k chars, nothing precomputable
#               -> MEASURED 27.6s on CPU vs 0.33s on GPU (~80x)
# Sharing one EMBED_DEVICE=cpu switch put the cross-encoder on CPU and made every
# query take ~37s against the SRS's <10s target. The reranker is the one model
# that must stay on the GPU; it fits next to a 3B LLM (see RERANK_FP16).
# None = auto (GPU if available). Set RERANK_DEVICE=cpu only on a GPU-less box.
RERANK_DEVICE = os.environ.get("RERANK_DEVICE") or None

# Load the cross-encoder in half precision on CUDA: ~2.2 GB -> ~1.1 GB VRAM, so
# it coexists with Ollama's ~2.2 GB on a 6 GB card. Measured rerank scores were
# identical to fp32 to 3 decimal places — ranking is unaffected. Ignored on CPU
# (fp16 on CPU is slower, not faster).
RERANK_FP16 = (os.environ.get("RERANK_FP16", "1").lower()
               not in ("0", "false", "no"))

# How many (query, chunk) pairs the cross-encoder scores in ONE forward pass.
# A VRAM CEILING, not a throughput knob: sentence-transformers defaults to 32,
# which with ~1,900-character Marathi chunks and a 3B LLM already resident OOMs
# a 6 GB card MID-REQUEST (not at startup). Bounding it here is what lets
# RetrievalConfig.rerank_pool be tuned for accuracy — measured hit@1 13/20 at a
# pool of 40 vs 12/20 at 15 — without the pool depth deciding whether the
# request survives.
RERANK_BATCH = int(os.environ.get("RERANK_BATCH", "8"))

# --- vector index backend (PLAN Phase 2) ---
# "flat" — FaissStore, IndexFlatIP, exact brute-force search with every chunk's
#          text in a RAM JSON sidecar. Correct and simple; O(n) per query. This
#          is what the original 713-vector demo index uses, and what the
#          model-free tests exercise.
# "hnsw" — HnswStore, IndexHNSWFlat (~O(log n)) with chunk text + metadata in
#          SQLite (engine/corpus_db.py). The multi-department corpus. RAM no
#          longer grows with the corpus, because no text is held in the process.
# The two are deliberately BOTH kept: the backend is a config change, not a
# rewrite, which is the same seam that makes groq<->ollama swappable.
VECTOR_BACKEND = os.environ.get("VECTOR_BACKEND", "flat").lower()

# Where the index lives. Root '/' has ~3 GB free on this machine, so the scaled
# index and its SQLite sidecar default to the big partition (HANDOFF §5).
INDEX_DIR = os.environ.get("MAHAGR_INDEX_DIR", "index")

# HNSW graph parameters — see HnswStore's docstring for what each one trades.
# M and efConstruction are BUILD-time (changing them requires a re-ingest);
# efSearch is QUERY-time and can be retuned freely against measured recall.
HNSW_M = int(os.environ.get("HNSW_M", "32"))
HNSW_EF_CONSTRUCTION = int(os.environ.get("HNSW_EF_CONSTRUCTION", "200"))
HNSW_EF_SEARCH = int(os.environ.get("HNSW_EF_SEARCH", "4096"))
# Absolute ceiling for the FILTERED search path, which asks for 4x the base
# (vector_store.HnswStore.search). It exists only to bound worst-case latency;
# it must never be smaller than HNSW_EF_SEARCH or the boost becomes a no-op —
# which is exactly what happened when a hardcoded 1024 outlived a base raised
# to 1024. Kept as a separate knob so the two cannot drift silently again.
HNSW_EF_SEARCH_MAX = int(os.environ.get("HNSW_EF_SEARCH_MAX",
                                        str(max(8192, 4 * HNSW_EF_SEARCH))))
# RE-CALIBRATED AGAIN 2026-08-10 at 401,573 vectors (the FULL 33-department
# corpus). scripts/eval_ann.py, same 23 real gold questions:
#
#   efSearch   recall@10   recall@60
#       512       0.948       0.953
#      1024       0.957       0.967    <- the wave-A value, already decayed
#      2048       0.961       0.975
#      4096       1.000       0.997    <- chosen
#     exact       1.000       1.000
#
# recall@60 0.967 -> 0.997 and recall@10 to a clean 1.000. ANN is ~2 ms against
# a retrieval stage that measures 2.78 s and an answer that measures 13.9 s, so
# this costs roughly 0.1% of a request. This project's failure mode is a MISSED
# GR, not a slow one; buy the recall.
#
# THE PATTERN TO CARRY FORWARD: this constant has now decayed twice, at every
# corpus growth, and nothing in the system reports it —
#   74k vectors  ef=512  recall@60 0.986
#  157k vectors  ef=512  recall@60 0.962   (same setting, quietly worse)
#  402k vectors  ef=1024 recall@60 0.967   (raised once, decayed again)
# Re-run scripts/eval_ann.py after ANY large ingest. Caveat: recall here is
# measured over 23 gold queries, so treat 1.000 as "no misses observed", not as
# a proof of exactness.
#
# The 2026-08-09 calibration at 156,795 vectors:
#
#   efSearch   recall@10   recall@60   ms/query
#      128       0.904       0.880       0.266
#      256       0.948       0.933       0.360
#      512       0.961       0.962       0.740    <- the old default
#     1024       0.965       0.980       1.105    <- chosen
#     exact      1.000       1.000       7.721
#
# THE POINT OF RE-RUNNING THIS: **HNSW recall degrades as the graph grows.**
# efSearch=512 measured recall@60 0.986 at 74k vectors and only 0.962 at 157k —
# the same setting, quietly worse, with nothing in the system reporting it. A
# greedy walk over a bigger small-world graph visits a smaller FRACTION of it
# for a fixed candidate budget, so efSearch is not a set-once constant: it has
# to be re-measured after any large ingest. Expect to raise it again at ~400k
# vectors (the full 33-department corpus).
#
# WHY 1024: recall@60 0.962 -> 0.980 for +0.37 ms, against a ~10 s answer. That
# is free. recall@10 is nearly flat (0.961 -> 0.965), which is exactly what the
# two-stage design predicts — stage one only has to get the right chunk into
# the 60-candidate pool, and the cross-encoder does the precision work after.
#
# The 2026-08-06 calibration at 64,744 vectors, kept because it is the evidence
# for the paragraph above:
#
#   efSearch   recall@10   recall@60   ms/query
#       32       0.813       0.720       0.053
#      128       0.970       0.927       0.242    <- the old default
#      256       0.983       0.970       0.317
#      512       0.987       0.986       0.533    <- chosen
#     1024       0.987       0.990       0.902
#     exact      1.000       1.000       8.340
#
# (At the time, 512 was chosen because recall@60 went 0.927 -> 0.986 for
# +0.29 ms. The reasoning was right; the constant simply did not survive the
# corpus doubling.)
#
# TWO HONEST NOTES:
#  * An earlier benchmark using RANDOM query vectors reported only 0.735 at
#    ef=128 and was far too pessimistic — a random vector in 1024-d is nearly
#    orthogonal to every real embedding, which is ANN's worst case. Always
#    calibrate with real query embeddings.
#  * At 64k vectors exact brute force is still only 8.3 ms, so HNSW is NOT
#    what makes this system fast today — the LLM dominates. HNSW is what stops
#    search cost growing LINEARLY as the corpus scales toward the full ~100k+
#    GR set. Claim it as a scaling property, not a current speed win.

# --- OCR for scanned documents ---
# Tesseract lang string: Marathi + Hindi + English. Install the traineddata:
#   sudo pacman -S tesseract tesseract-data-mar tesseract-data-hin tesseract-data-eng
OCR_LANGS = "mar+hin+eng"
OCR_DPI = 300               # render DPI for a text-less page before OCR

# --- LLM provider (the one remote piece; everything else runs locally) ---
# "groq"  — Groq-hosted Llama, fast, needs GROQ_API_KEY (demo default).
# "ollama"— a LOCAL model via Ollama's OpenAI-compatible API, so no document
#           ever leaves the machine — the NIC / on-premise deployment story.
# Switch with LLM_PROVIDER=ollama in the environment; nothing else changes
# (retrieval, embeddings and reranking are already fully local).
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

# The local model's CONTEXT WINDOW, in tokens. Ollama's own default is 4096 and
# it TRUNCATES a longer prompt SILENTLY — no error, no warning, it just drops the
# oldest tokens, which here means the system prompt with all the grounding and
# citation rules. So this must be raised in step with rag's context_token_budget,
# and it must MATCH the server: Ollama reads OLLAMA_CONTEXT_LENGTH at startup
# (set in deploy/ollama.service), while this value is what the engine budgets
# against and asserts on. Verify with `ollama ps` — the CONTEXT column is truth.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
