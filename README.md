# MahaGR Assist — शासन निर्णय सहाय्यक

A **grounded, multilingual** question-answering assistant over Maharashtra
Government documents — Government Resolutions (GRs), circulars, notifications
and office orders. Ask in **English or Marathi**; get answers pulled *only* from
the source documents, with a citation on every claim, in the language you asked.

> VJTI AI Hackathon 2026 · Problem Statement 3.

## Grounded · Multilingual · Explainable

- **Grounded** — answers come only from retrieved GR text; when the documents
  don't cover a question the assistant says so instead of guessing.
- **Multilingual** — one shared vector space (bge-m3) means an English query
  can retrieve a Marathi GR and vice-versa; answers come back in the question's
  language, preserving official terminology.
- **Explainable** — every answer carries `[n]` citations that resolve to the
  exact source GR (and page).

## Provenance / disclosure

This project is built on **our own open Retrieval-Augmented Generation core**
(originally written for a document-QA project). That core — hybrid retrieval
(dense + BM25), cross-encoder reranking, table/figure handling, grounded
generation with citation checking — is reused here as an authored library.

**The hackathon work is this repository's new domain layer:** multilingual
retrieval (bge-m3 + bge-reranker-v2-m3), Marathi/English **OCR** for scanned
GRs, Unicode-aware keyword search for Devanagari, and Government-document
question answering. See `git log` for what was built during the event.

## Architecture

```
INGEST (offline)   Documents → OCR (Marathi+English) → chunk (+tables) → embed (bge-m3)
KNOWLEDGE STORE    FAISS vector index + metadata
QUERY (online)     query (EN/Marathi) → hybrid search (BM25+dense) → rerank → grounded answer (cited)
```

## What changed from the English base (the multilingual swap)

All four coupled settings live in one file, [`backend/engine/config.py`](backend/engine/config.py):

| Setting | Was (English) | Now (multilingual) |
|---|---|---|
| Embedding model | `bge-small-en-v1.5` | `BAAI/bge-m3` |
| Embedding dim | 384 | 1024 |
| Query prefix | `"Represent this sentence…"` | `""` (bge-m3 needs none) |
| Reranker | `ms-marco-MiniLM` (EN) | `BAAI/bge-reranker-v2-m3` |

Plus: **OCR fallback** for scanned pages (`engine/ingest._ocr_page`); a
**Unicode-aware BM25 tokenizer** (`engine/hybrid.tokenize`) that keeps whole
Devanagari words instead of dropping or shattering them; and **GR metadata
extraction** (`engine/gr_metadata.py`) that parses each document's number,
date, department, category, language and cited/superseded GRs from its own
header text, so citations resolve to a real GR number + date, not just a
filename.

## Setup

```bash
cd backend
python3.11 -m venv .venv && source .venv/bin/activate   # 3.11–3.13 (torch/faiss wheels)
pip install -r requirements.txt
pip install -e .                 # puts the `engine` package on the path so scripts/ can import it

# OCR (only needed to read scanned GRs) — system Tesseract + language data:
#   Arch/CachyOS: sudo pacman -S tesseract tesseract-data-mar tesseract-data-hin tesseract-data-eng
#   Debian/Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-mar tesseract-ocr-hin

cp .env.example .env    # then paste your GROQ_API_KEY (free at console.groq.com)
```

> First run downloads the models (bge-m3 ≈ 2.2 GB, bge-reranker-v2-m3 ≈ 2.3 GB).

## Quickstart

```bash
# 1. Drop GR PDFs (born-digital or scanned) into backend/data/grs/
#    Good source: https://github.com/orgpedia/mahGRs  and  https://gr.maharashtra.gov.in

# 2. Build the index
python scripts/ingest_grs.py

# 3. Ask — in English or Marathi
python scripts/ask.py "What is the DTE admission fee structure?"
python scripts/ask.py "या शासन निर्णयात कोणती तारीख आहे?"
python scripts/ask.py            # interactive REPL (keeps conversational memory)
```

### No GR data yet? Smoke-test on the bundled fixtures

Two synthetic Marathi GRs ship in `backend/data/fixtures/` (a fee GR and a
2024 GR that supersedes it) so you can exercise the whole pipeline immediately:

```bash
python scripts/ingest_grs.py data/fixtures index
python scripts/ask.py "What is the OBC diploma fee in 2023?"     # → 6000, cited
python scripts/ask.py "Which GR supersedes the 2023 fee resolution?"
```

See [`backend/data/fixtures/README.md`](backend/data/fixtures/README.md) for
more queries and how to regenerate them.

### Real corpus + threshold calibration (Phase 1)

The `orgpedia/mahGRs` dataset already has every GR OCR'd into Marathi + English
text (no PDF/OCR needed for the corpus — our OCR path is for scanned-PDF demos).

```bash
# 1. Fetch a real corpus (default: 150 Higher & Technical Education GRs)
python scripts/fetch_mahgrs.py --count 200          # --dept, --recent, --translations

# 2. Build the index from the text corpus (bge-m3 embeddings)
python scripts/ingest_text.py data/grs_text index

# 3. Measure retrieval quality + calibrate thresholds against the gold set
python scripts/eval_retrieval.py                    # reports hit@k + a suggested cutoff
```

`eval_retrieval.py` retrieves with thresholds OFF to observe raw scores, prints
hit@1 / hit@5 / MRR, and shows the score gap between what should be KEPT vs
REJECTED — then suggests where to set `rerank_threshold` (or the cosine
thresholds with `--no-rerank`). Expand `data/gold/gold.json` to ~20–30 questions
for a trustworthy number.

> Heavy step: first ingest downloads bge-m3 (~2.2 GB) + bge-reranker-v2-m3
> (~2.3 GB). Ensure a few GB free (and ideally a GPU) before ingesting.

### Officer-assistance tools (Phase 2)

Built on the same engine (`engine/officer.py`, CLI `scripts/officer_tools.py`) —
all grounded and cited, in the question's language:

```bash
python scripts/officer_tools.py summarize <gr_id>            # plain summary of one GR (FR 3.5.2)
python scripts/officer_tools.py explain  "<question>"        # answer in simple language (FR 3.5.1)
python scripts/officer_tools.py compare  <gr_a> <gr_b>       # highlight differences (FR 3.5.3)
python scripts/officer_tools.py supersede <gr_id>            # which GRs it replaces / is replaced by (FR 3.5.5)
python scripts/officer_tools.py related  <gr_id>             # recommend similar GRs (FR 3.5.4)
```

`supersede` is a pure metadata-graph lookup (no LLM, no key); `related` is
vector similarity (no key); `summarize`/`explain`/`compare` need `GROQ_API_KEY`.

## Known gaps / roadmap (next steps, not yet done)

- ⚠ **Thresholds still need calibrating on YOUR corpus.** The harness is ready
  (`scripts/eval_retrieval.py` + `data/gold/gold.json`) — run it after ingesting
  a real corpus and set `text_threshold` / `table_threshold` / `rerank_threshold`
  in `engine/retrieval.py` from its suggested cutoff. They're recall-leaning
  placeholders until then. Expanding the gold set to ~20–30 questions makes the
  number trustworthy.
- **Frontend portal** — the officer-facing chat UI (not ported yet).
- **Tests** — the English base's suite was fixture-bound (arXiv gold set,
  Supabase, dim 384) and was removed rather than ported half-broken. Write a
  fresh suite against a GR gold set; the pure functions (chunking, RRF fusion,
  citation parsing, Devanagari tokenization) are the easy first targets.
- **Legacy corpus path** — `app/main.py` and `engine/documents.py` still carry
  the Supabase upload/Storage pipeline from the base. The baseline runs without
  them (`ingest_grs.py` + `ask.py`); trim or re-point them when building the portal.
- **Table/figure captions** — `_resolve_caption` matches the English words
  "Table"/"Figure"; add Marathi equivalents (तक्ता etc.) for Marathi GRs.

## Deployment note

The LLM call is isolated behind `engine/rag.py`, using Groq (fast) for the
demo. For on-premise / NIC deployment it can be swapped to a local Llama-3 via
Ollama so no document leaves the department — the rest of the pipeline
(embedding, retrieval, reranking) already runs fully locally.
