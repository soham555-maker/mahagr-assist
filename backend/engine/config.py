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
"""

# --- retrieval / embedding (the coupled four) ---
EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024
QUERY_PREFIX = ""            # bge-m3 needs no instruction prefix (unlike bge-*-en-v1.5)
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

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
import os as _os
LLM_PROVIDER = _os.environ.get("LLM_PROVIDER", "groq").lower()
OLLAMA_BASE_URL = _os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = _os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
