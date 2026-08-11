# MahaGR Assist — शासन निर्णय सहाय्यक

A **grounded, multilingual** question-answering assistant over Maharashtra
Government documents — Government Resolutions (GRs), circulars, notifications
and office orders. Ask in **English or Marathi**; get answers pulled *only* from
the source documents, with a citation on every claim, in the language you asked.

**Everything runs on one machine.** Embedding, retrieval, reranking *and*
generation are local — no document ever leaves the box.

> VJTI AI Hackathon 2026 · Problem Statement 3.

| | |
|---|---|
| **Corpus** | 99,410 Government Resolutions · 401,573 embedded passages · 33 departments · 1962–2029 |
| **Citation graph** | 317,250 reference edges · 8,983 supersession edges · 30,799 linked documents |
| **Models (all local)** | `BAAI/bge-m3` (embeddings) · `BAAI/bge-reranker-v2-m3` (reranker) · `qwen2.5:3b-instruct-q8_0` via Ollama (generation) |
| **Tests** | 195 passing, model-free, ~4 s |
| **Hardware it actually runs on** | one laptop, 6 GB RTX 4050 |

## Grounded · Multilingual · Explainable

- **Grounded** — answers come only from retrieved GR text; when the documents
  don't cover a question the assistant says so instead of guessing.
- **Multilingual** — one shared vector space (bge-m3) means an English query
  can retrieve a Marathi GR and vice-versa; answers come back in the language
  you choose, preserving official terminology.
- **Explainable** — every answer carries `[n]` citations that resolve to the
  exact source GR, its number, its date and its page.

---

## See it working

Every screenshot below is the real system, captured from a live run against the
full 99,410-GR index with the local model — not a mockup.

### The assistant answers, and shows its sources

An **English** question answered from a **Marathi** Government Resolution. Note
the amber banner: the cited GR has been **superseded** by a 2025 order, and the
system says so before the officer relies on it.

![Grounded answer with citation and supersession warning](docs/screenshots/13-ask-english-cited.png)

### It refuses when the corpus doesn't cover the question

This is the trust behaviour that matters most for a government tool — it does
not invent an answer.

![Abstention on an out-of-corpus question](docs/screenshots/16-ask-abstention.png)

### Cross-lingual retrieval, both directions

A **Marathi** question answered in **English**, sourced from the Marathi GR at
100% relevance. The query and the document never share a single word of script.

![Marathi question, English answer](docs/screenshots/18-ask-cross-lingual.png)

A **Marathi** question answered in **Marathi**, cited to the source GR:

![Marathi question, Marathi answer](docs/screenshots/14-ask-marathi.png)

### One query reaches across departments

Asked about Scheduled Tribe scholarships, it cites GRs from the **Tribal
Development Department** — a different department, and in real life a different
portal entirely — with relevance scores on each source.

![Cross-department retrieval](docs/screenshots/15-ask-cross-department.png)

### "Explain simply" for non-specialist readers

The same grounded pipeline, instructed to answer in plain language (SRS FR 3.5.1).

![Explain simply mode](docs/screenshots/17-ask-explain-simply.png)

### Scope the search, honestly

Department / date / language filters are pushed **into** the vector search
(FAISS `IDSelector`), not applied to the results afterwards — and each answer
carries a "Searched only: …" line so a narrowed search is never invisible.
Department counts come from the database, not a hard-coded list.

![Scope filter with live department counts](docs/screenshots/08-ask-scope-filter.png)

### Browse the corpus

Server-paginated over 99,410 documents with department chips — the browser never
downloads the corpus.

![Browse the indexed GRs](docs/screenshots/09-browse-list.png)

Open one and you get the full text, what it supersedes and relates to, and its
citation neighbourhood drawn from the graph:

![Document detail with graph and references](docs/screenshots/10-browse-document.png)

### Roles and an audit trail

Every question is audited — who asked what, when, and which GRs were cited.
The trail deliberately stores the question and the cited GR numbers but **never
the answer text or document bodies**, so it cannot become a second uncontrolled
copy of the corpus.

![Admin audit trail](docs/screenshots/12-admin-audit.png)

The same page as a Desk Officer — the IT Admin role is enforced on the server,
and the client-side check is convenience only:

![Role-based access control](docs/screenshots/19-rbac-access-denied.png)

<details>
<summary>More screenshots — landing page, login, empty states</summary>

Public landing page:

![Landing page](docs/screenshots/01-landing-hero.png)

Architecture, as presented on the landing page:

![Architecture diagram](docs/screenshots/02-landing-architecture.png)

Retrieval pipeline:

![Retrieval pipeline](docs/screenshots/03-landing-retrieval.png)

On-premise / security story:

![Security and on-prem](docs/screenshots/04-landing-security.png)

Measured results:

![Measured results](docs/screenshots/05-landing-results.png)

Officer sign-in (four SRS roles seeded):

![Login](docs/screenshots/06-login.png)

The Ask page before a question, showing the live corpus statistic read from
`/corpus/stats`:

![Ask page](docs/screenshots/07-ask-empty.png)

Document detail, scrolled to the full text and reference panels:

![Document full text](docs/screenshots/11-browse-document-detail.png)

</details>

---

## What it does, mapped to the SRS

| SRS requirement | Where it lives |
|---|---|
| FR 3.1 — repository, OCR, metadata, embeddings | `engine/ingest.py`, `engine/gr_metadata.py`, `scripts/ingest_corpus.py` |
| FR 3.2 — semantic + keyword search, ranked, with scores | `engine/retrieval.py` (`CorpusRetriever`), relevance shown on every source card |
| FR 3.3 — RAG, only from the corpus, cited, conflicts flagged, abstains | `engine/rag.py`, `officer.supersede_warnings` |
| FR 3.4 — English/Marathi in and out, cross-lingual search | `bge-m3` shared vector space, `rag.language_directive` |
| FR 3.5 — explain / summarize / compare / related / supersession | `engine/officer.py`, `POST /explain` `/summarize` `/compare`, `GET /related` `/supersede` |
| FR 3.7 — secure portal, chat, history, download, feedback | `frontend/`, `app/auth.py`, `app/db.py` |
| NFR — on-premise, explainable, scalable | fully local stack; `scripts/verify_offline.py` proves it |

## Architecture

```
INGEST (offline)   Documents → OCR (Marathi+English) → chunk (+tables) → embed (bge-m3, GPU)
KNOWLEDGE STORE    FAISS IndexHNSWFlat (vectors)  +  SQLite (document text, metadata, BM25/FTS5)
                   joined by one integer: gr_chunks.faiss_id == the vector's position
QUERY (online)     query (EN/Marathi)
                     → ANN over HNSW  ┐
                     → BM25 over FTS5 ┘ fused by RRF
                     → hydrate text from SQLite → group by GR → cross-encoder rerank
                     → grounded, cited answer (or an abstention)
```

**Scale.** The knowledge store holds **99,410 Government Resolutions across all
33 departments** of the `orgpedia/mahGRs` dataset. (`/corpus/stats` reports 34
department labels: the extra one holds the 2 synthetic fee-table sample GRs
described below.) Search is approximate-nearest-neighbour (~O(log n)) rather
than a brute-force scan, and **no chunk text is held in RAM** — FAISS returns
integer positions and SQLite turns them into text, so memory stays flat as the
corpus grows.

**Knowledge graph.** Each GR's `वाचा` (reference) block is parsed into edges, so
the system can answer "what does this replace, and has it itself been replaced?"
transitively. 317,250 edges were extracted; **46,412 (14.63%) resolve to a
document inside the corpus**. That number is a finding, not a bug: the corpus is
complete, so the remaining ~76% of references point at orders Maharashtra has
never published in this dataset. The system reports this rather than guessing.

## Measured quality — the honest numbers

Measured with `scripts/eval_retrieval.py` and `scripts/eval_answers.py` against a
23-question gold set on the full corpus. Full tables and caveats in
`CHECKLIST.md`.

| Metric | Result |
|---|---|
| Retrieval hit@1 / hit@5 / MRR | 14/20 · 18/20 · 0.787 |
| Answer carries a citation | 19–20/20 |
| Grounded (no unsupported claim) | 95% (3 phantom citations across 60 answers) |
| Factually correct | 13/20 |
| Degenerate answers | 0/20 |
| Correctly abstains on out-of-corpus questions | 2/3 |
| Latency p50 | 13.9 s — **over the SRS's 10 s target** |

Ranges are run-to-run variance at `temperature=0.2` over 20 questions; a
difference of 1 is noise. Generation is ~76% of the latency, so the remaining
levers trade against correctness — see "Known gaps" below.

## Provenance / disclosure

This project is built on **our own open Retrieval-Augmented Generation core**
(originally written for a document-QA project). That core — hybrid retrieval
(dense + BM25), cross-encoder reranking, table/figure handling, grounded
generation with citation checking — is reused here as an authored library.

**The hackathon work is this repository's new domain layer:** multilingual
retrieval (bge-m3 + bge-reranker-v2-m3), Marathi/English **OCR** for scanned
GRs, Unicode-aware keyword search for Devanagari, the scaled HNSW+SQLite
knowledge store, the GR citation/supersession graph, and Government-document
question answering. See `git log` for what was built during the event.

Two GRs in the corpus are **synthetic samples** (`backend/data/fixtures/`) —
realistic fee-table documents we generated to demonstrate table and exact-number
handling, one superseding the other. They are labelled as samples wherever they
appear; every other document is a real published GR.

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
Devanagari words instead of dropping or shattering them (`\w` splits "शासन" at
its vowel marks); and **GR metadata extraction** (`engine/gr_metadata.py`) that
parses each document's number, date, department, category, language and
cited/superseded GRs from its own header text, so citations resolve to a real GR
number + date, not just a filename.

## Setup

```bash
cd backend
python3.11 -m venv .venv && source .venv/bin/activate   # 3.11–3.13 (torch/faiss wheels)
pip install -r requirements.txt
pip install -e .                 # puts the `engine` package on the path so scripts/ can import it

# OCR (only needed to read scanned GRs) — system Tesseract + language data:
#   Arch/CachyOS: sudo pacman -S tesseract tesseract-data-mar tesseract-data-hin tesseract-data-eng
#   Debian/Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-mar tesseract-ocr-hin

cp .env.example .env
```

> First run downloads the models (bge-m3 ≈ 2.2 GB, bge-reranker-v2-m3 ≈ 2.3 GB).

## Running the full stack — order matters

`ollama serve` does **not** load the model; it loads it on the *first request*.
Starting the API first means the reranker takes ~1.2 GB of VRAM before the LLM
loads, which silently forces a partial CPU offload and a **measured 1.9× slower
decode** — with no error anywhere. So warm the model *before* starting uvicorn:

```bash
pkill -f "[u]vicorn app.api"; pkill -x ollama; sleep 4

OLLAMA_MODELS=/path/to/ollama OLLAMA_KEEP_ALIVE=-1 OLLAMA_CONTEXT_LENGTH=8192 ollama serve &
until curl -sf localhost:11434/api/tags >/dev/null; do sleep 2; done

# force the model into VRAM while the card is still free
curl -s localhost:11434/api/generate \
  -d '{"model":"qwen2.5:3b-instruct-q8_0","prompt":"hi","stream":false}' >/dev/null
ollama ps                       # MUST report 100% GPU

cd backend && uvicorn app.api:app --port 8000 &
curl -s localhost:8000/health   # llm_placement.fully_on_gpu MUST be true

cd ../frontend && npm install && npm run dev     # http://localhost:3000
```

`OLLAMA_KEEP_ALIVE=-1` matters: with a timed keep-alive the model unloads when
idle and the *next* question reloads it while uvicorn already holds the
reranker, silently restoring the slow path.

`GET /health` reports the active backend, index size, both model names and where
the LLM is actually running:

```json
{
  "status": "ok", "indexed_vectors": 401573, "vector_backend": "hnsw",
  "documents": 99410, "departments": 34,
  "embedding_model": "BAAI/bge-m3", "reranker_model": "BAAI/bge-reranker-v2-m3",
  "llm_provider": "ollama", "llm_model": "qwen2.5:3b-instruct-q8_0",
  "llm_placement": {"processor": "100% GPU", "fully_on_gpu": true, "context": 8192}
}
```

## Building a corpus from scratch

```bash
python scripts/fetch_mahgrs.py --list               # every department + its GR count
python scripts/fetch_mahgrs.py --cluster education  # 18,078 GRs, resumable, threaded
python scripts/fetch_mahgrs.py --all                # all 33 departments → 99,421

# Bulk embedding wants the WHOLE GPU, so stop anything else holding it first.
# .env pins EMBED_DEVICE=cpu for serving; a real env var overrides it.
pkill -f "[u]vicorn app.api"; pkill -x ollama
EMBED_DEVICE=cuda python scripts/ingest_corpus.py --batch-size 8

python scripts/build_graph.py                       # citation/supersession edges, seconds, no GPU
python scripts/verify_corpus.py                     # proves FAISS and SQLite still agree
python scripts/seed_users.py                        # the four SRS roles
```

`ingest_corpus.py` is **resumable and idempotent** — re-running it skips
everything already ingested, and after an interruption it reconciles SQLite
against the last saved FAISS index before continuing.

`verify_corpus.py` exists because equal row counts do **not** prove alignment:
an off-by-500 shift mid-corpus leaves both counts identical while every citation
names the wrong GR. It re-embeds a random sample and scores each chunk against
*its own* vector (correct ≈ 0.9998; misaligned ≈ 0.3–0.8).

### Evaluating it

```bash
python scripts/eval_retrieval.py     # hit@1 / hit@5 / MRR + threshold calibration
python scripts/eval_answers.py       # cited / grounded / correct / degenerate / latency (needs the API up)
python scripts/eval_ann.py           # HNSW efSearch recall vs exact brute force
python scripts/verify_offline.py     # PROVES on-prem (run with the API stopped)
```

## API surface

```
GET  /health                        liveness, index size, models, GPU placement
POST /auth/login                    JWT (bcrypt + PyJWT), four SRS roles
GET  /corpus/stats                  documents / chunks / departments / date span
GET  /documents?q&department&limit&offset      paginated GR list
GET  /documents/{id}/text           full text of one GR (view / download)
POST /ask                           grounded, cited answer (+ scope filters, conversation memory)
POST /summarize  /explain  /compare officer assistance (FR 3.5.1–3.5.3)
GET  /supersede/{id}  /related/{id} supersession + recommendations (no LLM)
GET  /graph/{id}                    citation neighbourhood + transitive supersede chain
GET  /graph/stats/summary           edge counts + resolution breakdown
GET  /conversations  /conversations/{id}       history (FR 3.7.3)
POST /feedback                      thumbs up/down on an answer (FR 3.7.5)
GET  /admin/audit-logs              IT Admin only
```

## Officer portal

Next.js 14 + shadcn/ui + Radix, navy/teal government palette, WCAG-minded.

| Route | What |
|---|---|
| `/` | public landing page — architecture, on-prem story, live corpus stat |
| `/login` | officer sign-in; JWT + role in `sessionStorage` (not `localStorage` — XSS reach) |
| `/ask` | grounded chat, EN/मराठी, scope filters, conflict banners, history, feedback |
| `/browse` | document browser, full text, summarize, supersession, related, graph, compare |
| `/admin` | audit trail, role-gated on the server |

## Tests

```bash
cd backend && pip install -r requirements-dev.txt
python -m pytest tests/ -q        # 195 tests, ~4 s, no models, no GPU, no network
```

Model-free by design: `conftest.py` stubs `sentence_transformers`, so the suite
runs fast and without torch. FAISS, numpy and SQLite are **real** in the tests —
only the models are stubbed. Covers Devanagari tokenization, GR metadata and
reference parsing, RRF fusion, citation parsing/resolution, chunking, table
detection, the HNSW id contract, corpus DB crash-recovery, the two-stage
retrieval pipeline, the supersession graph (including cycle guards), and auth.

## On-premise / NIC deployment

Embedding, retrieval, reranking **and generation** all run locally. Groq remains
behind a provider seam (`engine/rag.py`) as a hosted fallback, selected by
`LLM_PROVIDER`, but the default and the demo path is fully offline.

`scripts/verify_offline.py` **proves** it rather than asserting it: it blanks the
Groq key, makes every non-loopback socket raise, and then answers two questions.

Both models fit the 6 GB card together. Measured live with the full stack
serving: **3,698 MiB** (Ollama / qwen2.5-3b-q8_0) + **1,546 MiB** (the API
process — reranker weights plus CUDA context) = **5,382 of 6,141 MiB**. The
reranker is loaded **straight into fp16**
(`model_kwargs={"torch_dtype": "float16"}`); converting after load with `.half()`
leaves the discarded fp32 weights in torch's allocator and costs 2,766 MiB for a
1,083 MiB model — precisely the difference between fitting and OOM.

## Known gaps — stated, not hidden

- **Latency: p50 13.9 s against the SRS's 10 s target.** Generation is ~76% of
  it. The dials that shorten it (`context_token_budget`, `max_final_k`) trade
  against correctness, so this needs a measured decision, not another guess.
  *Answer streaming is the planned fix — it makes the wait perceptual rather
  than real, without trading retrieval quality.*
- **Marathi generation is weaker than English.** On the 3B local model, Marathi
  answers are terser and occasionally repetitive where the English ones are
  fluent. Retrieval is unaffected (it is the same vector space either way) — this
  is a generation-side limit of a 3-billion-parameter model on a 6 GB card.
- **Hindi is not supported yet**, though the SRS asks for it. bge-m3 and qwen2.5
  both handle Hindi, so this is a small change rather than a structural one.
- **One out-of-corpus question still answers instead of refusing** (farm loan
  waiver). It scores 0.825 — far above the abstention floor — because it
  genuinely retrieves loan-related GRs. A score gate cannot catch that one.
- **`DEV_NO_AUTH=1` is set in `backend/.env`** for the demo, which bypasses
  login, roles and rate limiting at runtime. The portal is not "secured" until
  it is turned off.
- **Graph resolution is 14.63%** — an absolute ceiling, because ~76% of
  references in Maharashtra's published GRs point outside the published dataset.
- **The 16 original GR PDFs** in `Original Maha-GR/` have a garbled text layer
  (broken font encoding) and still need an OCR re-ingest.

## Roadmap

The next block of work — a full portal redesign, semantic search replacing the
current title-filter browse, and a set of new officer tools (currency check,
policy timeline, case files, comparison matrix) — is planned phase-by-phase in
[`PLAN_V2.md`](PLAN_V2.md). Engineering history and per-phase proof live in
[`CHECKLIST.md`](CHECKLIST.md) and [`HANDOFF.md`](HANDOFF.md).
