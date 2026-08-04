# MahaGR Assist — HANDOFF

Written for someone with **zero memory of how this was built**. Read this first.

## 1. What this project is

A **multilingual, source-grounded question-answering system over Maharashtra
Government Resolutions (GRs)**, for the VJTI AI Hackathon 2026, Problem
Statement 3. An officer asks a question in **English or Marathi**; the system
retrieves the relevant GRs and answers **only from them**, with a **citation on
every claim**, and **refuses ("insufficient information")** when the corpus
doesn't cover the question. Backend = Python (FastAPI + a RAG engine); frontend
= Next.js. The RAG engine was ported from a prior project (ResearchOS) and
adapted; that reuse is disclosed in `README.md`.

There is **no SPEC.md / PLAN.md in the repo**. The "plan" is **`ROADMAP.md`** (a
phased checklist, kept ticked as phases land). The formal spec is the external
hackathon SRS (a Word doc, not in the repo). "Phase N" everywhere = the phases
in `ROADMAP.md`.

## 2. Current state at a glance

- **Index built:** `backend/index/` = **713 vectors, dim 1024** (196 real
  Higher & Technical Education GRs from orgpedia + 2 synthetic fee-table GRs).
- **Models:** `bge-m3` (embeddings, 1024-d) + `bge-reranker-v2-m3` (reranker),
  cached in `~/.cache/huggingface` (~4.5 GB). GPU = NVIDIA RTX 4050 (6 GB),
  torch 2.13.0+cu130. It also runs on CPU, slower.
- **Retrieval quality (measured):** hit@1 19/20, hit@5 20/20, MRR 0.975 with the
  reranker, on a 23-question gold set. Out-of-corpus questions abstain.
- **LLM:** Groq `llama-3.1-8b-instant`. `backend/.env` has a working key.
- **Venv:** `backend/.venv` (Python 3.11). `pip install -e .` already done.
- **Tests:** 37, all passing, model-free (no torch needed to run them).
- **Git:** local repo, remote = `github.com/soham555-maker/mahagr-assist`.
  ⚠ The remote still has OLD commits that include `Co-Authored-By: Claude`
  trailers; local history was rewritten to remove them but **not force-pushed
  yet** (owner will `git push --force origin main` later).

## 3. Repo map, file by file

### Root
- `README.md` — project overview, setup, quickstart, disclosure of engine reuse.
- `ROADMAP.md` — **the phase checklist** (source of truth for status).
- `DEMO.md` — presenter cheat-sheet: verified demo questions + click order.
- `DEPLOYMENT.md` — Phase 7 runbook (Dockerfiles/compose as text; not yet created as files).
- `HANDOFF.md` — this file.
- `Original Maha-GR/` — 16 real government GR PDFs the owner supplied. **Gitignored.**
  Their text layer is *garbled* (broken font encoding) — see decisions below. Ingested
  once then pruned; pending OCR re-ingest.
- **Presentation artifacts live OUTSIDE the repo**, in
  `/home/soham/Projects/Claude/vjti-hackathon/`: `MahaGR_PA3_Presentation.pptx`
  (the submission deck), `MahaGR-Assist-Deck.pptx` (earlier pitch deck),
  `architecture.svg`, `DEMO_VIDEO_SCRIPT.md`, `build.js`/`deck2.js` (deck generators).

### backend/engine/ — the RAG engine (no web layer)
- `config.py` — **single source of truth** for model choices: `EMBED_MODEL=BAAI/bge-m3`,
  `EMBED_DIM=1024`, `QUERY_PREFIX=""`, `RERANK_MODEL=BAAI/bge-reranker-v2-m3`,
  `OCR_LANGS="mar+hin+eng"`, `OCR_DPI=300`, and the LLM seam
  (`LLM_PROVIDER`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`).
- `ingest.py` — `IngestionPipeline`. `extract_pages_from_pdf(force_ocr=)` (text
  layer, with OCR fallback for empty pages and `force_ocr` to override a broken
  text layer); `_ocr_page` (Tesseract, graceful if not installed); `process_pdf`
  (v1: text + pdfplumber tables); `process_text` (pre-extracted `.mr.txt`, splits
  on `# Page N` markers); `chunk_pages`; `process_pdf_v2` (pymupdf4llm visual path,
  **ported, unused by the current corpus**).
- `vector_store.py` — `FaissStore` (FAISS `IndexFlatIP`, cosine via normalized
  vectors; dim from config). add/search/save/load + `hit_for_index`.
- `retrieval.py` — `RetrievalConfig` (**calibrated thresholds:** `text 0.55`,
  `table 0.50`, `rerank 0.85`, `query_prefix ""`); `KeywordIndex` (BM25);
  `Retriever` (dense + BM25 fused by RRF, optional cross-encoder rerank, threshold
  gate → abstain via floor + `low_confidence`); `load_default_retriever`.
- `hybrid.py` — `tokenize` (Devanagari-aware, `[a-z0-9ऀ-ॿ]+`) and `rrf_fuse`.
- `reranker.py` — `Reranker` wrapping `bge-reranker-v2-m3`.
- `rag.py` — `GenerationConfig` (LLM provider seam; Groq model
  `llama-3.1-8b-instant`, `context_token_budget=2500`, `max_tokens=700`);
  `SYSTEM_PROMPT` (government + multilingual + conflict-flagging rules);
  `build_prompt`, `format_block`, `parse_citations`, `resolve_citations`,
  `trim_to_budget`; `make_client` (groq or ollama), `OllamaClient` (stdlib),
  `call_llm` (provider-aware, 429 backoff on groq), `rewrite_query`, `answer`.
- `gr_metadata.py` — `extract(text)` → gr_number, ISO date (from Marathi
  digits/months), department, category, language (mr/en), references, supersedes.
  Recognizes शासन निर्णय / शासन आदेश / परिपत्रक / अधिसूचना number labels.
- `officer.py` — officer-assistance features: `document_chunks`, `list_documents`,
  `summarize`, `explain`, `compare`, `supersession` (metadata graph),
  `supersede_warnings` (conflict check used by `/ask`), `related` (vector similarity).
- `table_extract.py` — pdfplumber table extraction → `row_to_sentence` (ported;
  English "Table N" caption detection).
- `visual_ingest.py` — figure/formula layout extraction (ported; not on the text path).
- `documents.py` — **LEGACY**, Supabase upload path from ResearchOS; **not used** by `app/api.py`.

### backend/app/ — the web layer
- `api.py` — **the FastAPI app in use.** Loads index + models once (lifespan) +
  `db.init()`. Endpoints: `/health`, `/documents`, `/documents/{id}/text`,
  `/ask` (persists turn + loads DB history into rewrite + adds conflict warnings),
  `/conversations` (GET/DELETE), `/conversations/{id}`, `/feedback`, `/summarize`,
  `/explain`, `/compare`, `/supersede/{id}`, `/related/{id}`. No Supabase/auth.
- `db.py` — SQLite persistence (conversations, messages incl. sources+warnings,
  feedback). Path via `MAHAGR_DB` (default `data/db/mahagr.db`).
- `main.py` — **LEGACY** ResearchOS Supabase app; **not used** (run `api.py`).

### backend/scripts/ — CLIs (run from `backend/`, venv active)
- `fetch_mahgrs.py` — download the orgpedia `.mr.txt` corpus.
- `ingest_text.py` — build the index from `.mr.txt` files (the real corpus path).
- `ingest_grs.py` — build a **fresh** index from a folder of PDFs.
- `add_pdfs.py "<dir>" [index] [--ocr]` — **append** PDFs to an existing index; `--ocr` forces OCR.
- `add_fixtures.py` — append the 2 synthetic fee-table GRs to the index.
- `ask.py "question"` — CLI query (reranker + conversational memory).
- `officer_tools.py <summarize|explain|compare|supersede|related> ...` — officer features CLI.
- `eval_retrieval.py [--no-rerank]` — gold-set retrieval eval + threshold calibration.
- `make_fixtures.py` — regenerate the synthetic fixture PDFs (reportlab).
- `verify_demo.py` — verify the demo questions still retrieve the right GR + number.

### backend/tests/ — 37 model-free tests
`conftest.py` (stubs `sentence_transformers`, adds backend to path),
`test_hybrid`, `test_gr_metadata`, `test_chunking`, `test_rag`, `test_officer`,
`test_config`, `test_llm_provider`, `test_db`.

### backend/data/
- `grs_text/` — 196 orgpedia `.mr.txt` GRs (the corpus source; gitignored, present locally).
- `fixtures/` — `GR-2023-fees.pdf`, `GR-2024-fees-revised.pdf` (synthetic, with fee
  tables) + `README.md`. The 2024 supersedes the 2023.
- `gold/gold.json` — 23-question gold set (EN + Marathi + out-of-corpus).
- `grs/.gitkeep` — placeholder for raw PDF drops. `db/` — SQLite (gitignored, runtime).

### backend/ (root files)
`requirements.txt`, `requirements-dev.txt` (pytest), `pyproject.toml`
(`pip install -e .` exposes the `engine` package), `.env` (has a working
`GROQ_API_KEY`), `.env.example`, `index/` (built FAISS index).

### frontend/ — Next.js 14 portal (navy/teal theme)
- `app/page.tsx` — **Ask** view: grounded chat, English/Marathi toggle,
  **"Explain simply"** mode (calls `/explain`), abstention banner, conflict
  banner, **conversation-history sidebar**, **thumbs feedback**.
- `app/browse/page.tsx` — **Browse** view: search GRs, read full text,
  **Summarize** button, Supersession + Related panels, **Compare two GRs**.
- `lib/api.ts` — typed client for every backend endpoint.
- `components/ui.tsx` — shared UI (AbstentionBanner, LangToggle, SourceCard, Spinner, EmptyHint, isAbstention).
- `app/layout.tsx`, `app/globals.css`, `tailwind.config.ts`, `.env.local`
  (`NEXT_PUBLIC_API_URL`, default `http://localhost:8000`), `.env.local.example`.

## 4. Status by phase (see ROADMAP.md)

- **Phase 1 — Corpus & retrieval:** ✅ DONE **except** OCR re-ingest of the 16
  `Original Maha-GR` PDFs (blocked on installing Tesseract Marathi data). Gold
  set expanded to 23; thresholds calibrated on real data.
- **Phase 2 — Officer features:** ✅ DONE. Conflict/supersede warnings on `/ask`;
  summarize/explain/compare/supersede/related all have API routes **and** UI actions.
- **Phase 3 — Persistence:** ✅ DONE. SQLite conversations/messages/feedback;
  history loaded from DB into query-rewrite; sidebar + feedback thumbs in the portal.
- **Phase 4 — Auth, roles & security:** ❌ NOT STARTED (needs a decision: real
  officer login/JWT/roles, or skip for the hackathon).
- **Phase 5 — Frontend polish:** 🟡 PARTIAL. Portal works with all features; NOT
  done: filter by department/date/language, download referenced GR, mobile
  sidebar, richer empty/error states.
- **Phase 6 — Tests/observability:** 🟡 PARTIAL. 37 tests ✅; NOT done: a <10s
  latency check and request logging.
- **Phase 7 — Deployment:** 🟡 PARTIAL. `DEPLOYMENT.md` runbook written; the
  Ollama on-prem seam exists in code; Dockerfiles/compose files NOT created yet.
- **Phase 8 — Presentation:** ✅ DONE (deck + diagram + demo/video scripts, in
  `/home/soham/Projects/Claude/vjti-hackathon/`).

## 5. Non-obvious decisions (why, tried-and-rejected, gotchas)

1. **Fresh repo, engine ported from ResearchOS, disclosed.** To satisfy the
   hackathon's "original work / don't reuse a whole prior project" concern:
   new domain (Marathi GRs), new git history, README discloses the reused core.
2. **Translate at the edges, never at ingest.** We index the **original Marathi**
   (so citations point at the real GR, not a machine translation). Cross-lingual
   search works because `bge-m3` maps Marathi and English into **one shared vector
   space** — no translation step. Rejected: translating the corpus to English.
3. **bge-m3 swap was not "just change one line."** It forced: dim 384→1024 (index
   rebuilt), **remove the query prefix** (bge-*-en needed one; bge-m3 does not —
   leaving it silently degrades retrieval), a **multilingual reranker**, a
   **Devanagari tokenizer fix**, and **threshold recalibration**.
4. **Devanagari tokenizer bug (real gotcha).** Python's `\w` excludes Devanagari
   vowel-marks (matras), so it **split Marathi words** ("शासन" → "श","सन"); the old
   ASCII `[a-z0-9]` dropped Marathi entirely. Fix: `[a-z0-9ऀ-ॿ]+` keeps whole words.
5. **Government PDFs have BROKEN font encodings, not just scans.** The 16
   `Original Maha-GR` PDFs have a text layer, but it extracts as **garbled
   Devanagari** ("निर्णय"→"चनणचय"). So OCR is needed even when a text layer exists
   → added `force_ocr`. Those 16 were ingested from the garbled text then **pruned**;
   re-ingest with `--ocr` after installing Tesseract Marathi. This is why the
   OCR feature matters beyond scanned images.
6. **Groq model = 8b-instant, not 70B.** 70B hit Groq's **per-day token limit**
   (100k). Switched to `llama-3.1-8b-instant` (separate, larger daily bucket) and
   cut `context_token_budget` 6000→2500 and `max_tokens` 1024→700 to conserve
   tokens. Gotcha: **Groq limits are per-account, not per-key** — a new key on the
   same account shares the same exhausted quota; only a different account / waiting
   for reset / paid tier helps.
7. **Slim new `app/api.py`, not the ported `main.py`.** The ResearchOS `main.py`
   is Supabase/multi-tenant; we wrote a dependency-light local API instead
   (on-prem friendly). `main.py` and `documents.py` remain but are unused.
8. **Fixtures are SYNTHETIC and generated with reportlab (not LibreOffice).**
   Tried LibreOffice HTML→PDF first; its subset font produced `(cid:NN)` garbage in
   pdfplumber and wrapped numbers across lines. reportlab gives a clean text layer +
   ruled tables; it lacks complex-script shaping (visually imperfect Devanagari) but
   the **extracted text is correct**, which is what matters. Per-script font
   fallback is needed because Noto Devanagari has no Latin glyphs. The fixtures
   exist to demonstrate **table/number retrieval** (the real orgpedia text has no
   structured tables). They are disclosed as synthetic.
9. **Old ResearchOS tests were deleted, not ported** (they were bound to arXiv
   fixtures/Supabase/dim-384). New suite is model-free via a `sentence_transformers`
   stub in `conftest.py`.
10. **`eval_retrieval.py` measures with thresholds OFF** to see the raw
    keep-vs-reject score gap, then suggests a cutoff. `rerank_threshold` governs
    abstention on the deployed (reranker) path; the cosine `text_threshold` only
    applies on the `--no-rerank` path.
11. **Supersede graph = metadata, not an LLM call.** Built from each GR's parsed
    `references` + `supersedes` flag. `officer.supersede_warnings` uses it to flag,
    on `/ask`, when a cited GR is superseded by a newer one (deterministic).
12. **`pip install -e .` is REQUIRED** — scripts do `from engine import ...`, and
    the editable install is what puts `engine` on the path. Missing it → `ModuleNotFoundError: engine`.
13. **Python 3.11 venv on purpose** — 3.14 was too new for torch/faiss wheels.
14. **Claude attribution removed from git history** at the owner's request; remote
    still needs a force-push to reflect that.

## 6. Commands (run from `backend/` with the venv active unless noted)

```bash
# --- one-time setup ---
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pip install -r requirements-dev.txt          # for tests
# OCR system packages (CachyOS/Arch), needed to ingest scanned/broken-font PDFs:
sudo pacman -S tesseract-data-mar tesseract-data-hin tesseract-data-eng

# --- build / refresh the index (the current index is already built) ---
python scripts/fetch_mahgrs.py --count 200        # -> data/grs_text/
python scripts/ingest_text.py data/grs_text index # -> index/ (downloads models on first run)
python scripts/add_fixtures.py                    # + the 2 fee-table sample GRs
# after installing tesseract-mar, add the owner's 16 real PDFs via OCR:
python scripts/add_pdfs.py "../Original Maha-GR" index --ocr

# --- evaluate / verify (no GROQ key needed) ---
python scripts/eval_retrieval.py                  # hit@k + rerank calibration
python scripts/eval_retrieval.py --no-rerank      # cosine calibration
python scripts/verify_demo.py                     # confirm demo questions retrieve correctly
python -m pytest tests/ -q                        # 37 model-free tests

# --- query from the CLI (needs GROQ_API_KEY in .env) ---
python scripts/ask.py "What is the OBC diploma fee for 2023-24?"
python scripts/officer_tools.py supersede संकीर्ण-२०२३        # no key needed
python scripts/officer_tools.py summarize 201710121514029708 # needs key

# --- run the app (two terminals) ---
uvicorn app.api:app --port 8000                   # backend (loads models ~40s; needs key for LLM endpoints)
# in frontend/:  npm install && npm run dev       # http://localhost:3000 ; NEXT_PUBLIC_API_URL -> :8000
```

Notes: model load needs ~4.5 GB GPU/RAM; kill stray servers before re-running
(a lingering `uvicorn` holds the GPU). Health check: `curl localhost:8000/health`.

## 7. The precise next step + standing agreement

**Next step:** pick one and tell the assistant to do it —
- **Phase 4 (auth/roles)** — the only phase needing a decision first: do you want
  real officer login (JWT + the SRS roles), or skip it for the hackathon?
- **Phase 5 (frontend polish)** — filters, document download, mobile, states.
- **Finish Phase 1's OCR ingest** — just run the `sudo pacman` line above, then the
  assistant runs `add_pdfs.py ... --ocr` and verifies the 16 real GRs read cleanly.

**Two small external to-dos (owner):** install `tesseract-data-mar` (to finish
Phase 1 OCR); `git push --force origin main` (to scrub Claude from the GitHub history).

**Standing agreement:** after implementing each phase, the assistant **explains
what was done in that phase in plain language**, and (when asked) gives
**interview-style questions** for it and **what the owner must do to prep the next
phase**. Keep honoring this. Do not start a phase until told which one.
