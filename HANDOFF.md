# MahaGR Assist — HANDOFF

Written for a reader with **zero memory of the build**. Read this first, then
`PLAN.md` (the phase-by-phase plan being implemented now), `CHECKLIST.md`
(what is actually done, with proof) and `ROADMAP.md` (the earlier initial-build
checklist).

---

## 0. NEXT SESSION — START HERE

> **RESYNC 2026-08-09.** This file was written on 2026-08-06 and a full day of
> work (2026-08-07) landed after it. What follows in §0 was re-verified against
> the tree on 2026-08-09; §2–§5 have been corrected in place. If a number here
> disagrees with an older paragraph further down, **this section wins**.

### Where the project actually is (end of 2026-08-10)

- **THE CORPUS IS COMPLETE: 99,410 GRs / 401,573 vectors, all 33 departments**
  of the orgpedia dataset, at `/mnt/win/mahagr/index/`. PLAN Phase 6 finished
  2026-08-10. Every older figure (18,078 / 41,474 / 64,744 / 156,795) is dead.
- **Graph: 317,250 edges — 46,412 resolved / 29,466 ambiguous / 241,372
  dangling**, **30,799 documents** with at least one resolved edge, 8,983
  supersede edges.
- **Tests: 157, all passing.**
- **Measured, current:** hit@1 14/20 · hit@5 18/20 · MRR 0.787 · CITED 19/20 ·
  **GROUNDED 95% (3 phantoms in 60 answers, measured over 3 repeats)** ·
  CORRECT 13/20 · DEGENERATE 0/20 · ABSTAINS 2/3 · latency **p50 13.9 s**
  (still over the SRS's 10 s — see "open" below).
- **`efSearch` is 4096** (recall@60 0.997). It has now decayed TWICE with corpus
  growth; re-run `scripts/eval_ann.py` after any large ingest.
- **Disk is no longer the emergency it was.** `/mnt/win` has **137 GB free**
  (older notes say 47 GB). Root `/` is still tight: **2.1 GB free of 93 GB**.
- **PLAN Phase 4 (auth/RBAC/audit) is IMPLEMENTED but untested**, and
  **Phase 5 (portal rebuild on shadcn/ui) is largely done**.

Full write-up with every before/after table: **`CHECKLIST.md` → PLAN Phase 6**.

### THE CURRENT JOB — answer quality vs the 10 s NFR

PLAN Phase 6 is DONE; there is no more corpus to ingest. The open work is the
tension between latency and answer quality, which `context_token_budget` dials
between and neither end of which meets the SRS:

| budget | latency p50 | CORRECT | blocks reaching the model |
|---|---|---|---|
| 6000 | 34.1 s | 14/20 | ~3 |
| **3000 (current)** | **13.9 s** | **13/20** | often **1** |

At 3000, one long Marathi chunk costs ~2,280 tokens, so `trim_to_budget` keeps
ONE block and drops 11 — HANDOFF §5.5's starvation failure, returning at the
budget chosen to fix latency. That starvation is also what turns the model's
transcription of a document's own numbering (*"अ.क्र.४, ५ व ६"*) into phantom
citations. **Fixing this needs a measurement designed for it — repeats, and more
than 23 gold questions — not another single-run tweak.**

**Read the ops notes in §5.29-§5.36 before touching any of it; all cost real
time and none are visible in the code.**

### START THE STACK IN THIS ORDER — the runbook was subtly wrong

`ollama serve` does **not** load the model; it loads it on the **first request**.
So "start Ollama before the API" (what this file used to say) still ended with
the LLM loading *after* uvicorn had taken 1.2 GB for the reranker, leaving it too
little VRAM and forcing a **partial CPU offload — measured 1.9x slower decode**
(generation p50 18.9 s → 10.1 s once fixed). Waiting on `/api/tags` proves the
daemon is up, not that the model is resident. The missing step is a warmup:

```bash
pkill -f "[u]vicorn app.api"; pkill -x ollama; sleep 4
OLLAMA_MODELS=/mnt/win/mahagr/ollama OLLAMA_KEEP_ALIVE=60m \
  OLLAMA_CONTEXT_LENGTH=8192 ollama serve &
until curl -sf localhost:11434/api/tags >/dev/null; do sleep 2; done
# THE STEP THAT WAS MISSING — force the model into VRAM while the card is free
curl -s localhost:11434/api/generate \
  -d '{"model":"qwen2.5:3b-instruct-q8_0","prompt":"hi","stream":false}' >/dev/null
ollama ps            # MUST say 100% GPU
uvicorn app.api:app --port 8000 &
curl -s localhost:8000/health   # llm_placement.fully_on_gpu MUST be true
```

Both models fit — **3,734 MiB (LLM) + 1,179 MiB (reranker) = 4,913 of 6,141**.
Capacity was never the problem; load ORDER was. `/health` reports this, so check
it before quoting any latency number.

**AND THE ORDERING FIX EXPIRES.** `OLLAMA_KEEP_ALIVE=30m`/`60m` unloads the model
when idle; the next question reloads it *while uvicorn already holds the
reranker*, silently restoring the CPU offload and the 1.9x slowdown. Observed
exactly that an hour after a successful run — `ollama ps` empty,
`llm_placement: null`. **For a demo use `OLLAMA_KEEP_ALIVE=-1` (never unload).**
Otherwise: idle an hour, ask the first question, and it runs slow with no error.

**Also note `hnsw_meta.json` keeps the efSearch value from BUILD time (512).**
That is not stale config — `HnswStore.load()` deliberately re-applies
`config.HNSW_EF_SEARCH` (verified: the loaded index reports 1024). efSearch is a
query-time dial; do not "fix" the meta file.

### Fixed on 2026-08-09 (was "known breakage")

1. ✅ **`SYSTEM_PROMPT_COMPACT` lost the conflict rule (FR 3.3.3) and
   "cite nothing"** in an undocumented 2026-08-07 rewrite. Restored; the
   regression test passes again.
2. ✅ **`JWT_SECRET` no longer has a shipped default.** `app/auth.py` now raises
   at import if it is unset (unless `DEV_NO_AUTH=1`), and `.env` holds a
   generated 64-char secret. A signing key with a public default in the repo
   lets anyone mint an "IT Admin" token.
3. ⚠ **`DEV_NO_AUTH=1` is still set in `backend/.env`**, so Phase 4 auth is
   bypassed at runtime. Intentional for the demo — but do not claim the portal
   is secured without turning it off.
4. ⚠ **Phase 4 still has no tests** — the only module in the repo without them.

### Open, and honest

- **The SRS's 10 s NFR is not met: p50 13.3 s, 16/23 over.** Generation is 76%
  of it. Completion is ~517 tokens against a 200-350 expectation, and 7/23 hit
  the 768 cap. The remaining levers trade against CORRECT — this needs a
  measured decision, not another guess.
- **One out-of-corpus question still answers instead of refusing** (farm loan
  waiver). It scores **0.825**, far above the abstention floor, because it
  genuinely retrieves loan-related GRs. The hard gate cannot catch this one.
- **Retrieval's improvement is not attributable to wave A alone** — it is the
  first `eval_retrieval` since 2026-08-06, across four changes. Say "improved
  across these four", not "adding Finance improved retrieval".

### What changed on 2026-08-07 that no document recorded

- **`engine/text_table_detect.py` (new)** — table detection over OCR'd text, and
  a **full re-ingest** to apply it. Chunks 64,744 → 74,004, of which **17,736 are
  `content_type='table'`**. See PLAN.md "Phase 2.5". Tests:
  `test_text_table_detect.py`, `test_text_table_ingest.py`.
- **The 2 synthetic fee-table fixtures were added to the hnsw corpus** (18,078 →
  18,080 documents, department `उच्च व तंत्र शिक्षण विभाग`). They used to exist only
  in the flat index, which is why older notes call those gold questions
  "structurally unanswerable" on hnsw. That caveat no longer applies.
- **`RetrievalConfig.rerank_pool` 15 → 40** — the only dial that moved retrieval
  quality on the big corpus (hit@1 12→13/20, hit@5 14→15/20, MRR 0.642→0.688,
  for ~+0.5 s). Widening the ANN/BM25 candidate lists changed nothing, i.e. the
  right chunk was already in the pool and the cross-encoder never read it.
- **Phase 4 auth** — `app/auth.py`, `users`/`audit_log` in `app/db.py`,
  `POST /auth/login`, `GET /admin/audit-logs`, `current_user` on every route,
  `/ask` writing an audit row, `scripts/seed_users.py`.
- **Phase 5 portal rebuild** — shadcn/ui + Radix, a public landing page at
  `app/page.tsx`, and the officer portal moved to a route group
  `app/(portal)/{ask,browse,admin}`. `npx tsc --noEmit` clean.
- **`/health` now reports `llm_placement`** — whether Ollama put the model fully
  on the GPU. A partial CPU offload is a ~4x slowdown that raises no error.
- **`rag.merge_extra_chunks`** and `OLLAMA_FREQUENCY_PENALTY` were added;
  `LLM_MAX_TOKENS=2048` is now set in `.env`.

### JOB 1 — Fix the graph's 2% reference-resolution rate  ✅ **DONE (2026-08-06)**

Resolution **2.0% → 9.5%**; documents with at least one resolved edge
**1,123 → 3,904**; median canonical length of an extracted reference **19 → 28**
(the same as a document's own number, which is what "no longer truncated"
looks like). Full write-up, including the two traps and why 9.5% is the honest
ceiling, is in `CHECKLIST.md` under PLAN Phase 3. What shipped:

- `gr_metadata`: reference lines are now parsed as items → comma-separated
  segments → trimmed at the date, instead of as whitespace-bounded tokens;
  `_parse_date` learned the dotted form (`दि. ३०.१०.२०१०`), which took cited-date
  coverage 34% → 72%; `extract()` returns `reference_details` ({number, date})
  alongside the old `references` list.
- `graph.build_edges` uses the cited **date** to disambiguate a number shared by
  several documents (**1,771** edges resolved that way); it never guesses —
  no unique date match means the edge stays `ambiguous`.
- `scripts/reparse_refs.py` (**new**) re-parses from `gr_documents.text`, so no
  re-ingest: ~3 min on CPU vs ~25 min of GPU.
- Tests **108 → 122**.

This closes PLAN Phase 3. **JOB 2 below is the next thing to do.**

### JOB 2 — Get latency under the SRS's 10 s, and accuracy back up  *(~1 hour, no GPU)*

Currently **p50 11.4 s, 12/23 over 10 s** (was 4.7 s at 196 GRs).

**Where the time goes (inferred, not yet instrumented):** out-of-corpus
questions returned in **2.6-3.2 s** while full answers took **12-16 s**. Both do
the same retrieval; only the generation length differs. So retrieval is roughly
1-2 s and **generation dominates**. The lever is therefore PROMPT SIZE, not the
vector index — `efSearch=512` costs 0.53 ms and is irrelevant here.

**Try in this order, re-measuring with `scripts/eval_answers.py` each time
(needs the API running):**

    1. rerank_threshold 0.85 -> ~0.99   (engine/retrieval.py RetrievalConfig)
       THE BIG ONE, and it is the SAME change that should fix accuracy.
       It is still calibrated for 196 GRs; eval_retrieval.py on the big corpus
       suggests ~0.993. Fewer weak chunks clear the gate -> shorter prompt ->
       faster generation AND fewer wrong citations. One line. Do this first.
    2. max_final_k 12 -> 5-6            (fewer context blocks)
    3. context_token_budget 6000 -> ~3500 for ollama (engine/rag.py).
       CAUTION: 3500 was measured WORSE on the 196-GR corpus (CORRECT 17/20 vs
       19-20/20) because the answer was often in block 2-3. Re-test on the big
       corpus rather than assuming that still holds.
    4. Cap generation length (num_predict / max_tokens) — bounds the worst case
       (max was 29.5 s) without touching retrieval.
    5. Only if still short: instrument /ask with per-stage timings to confirm
       the retrieval-vs-generation split rather than inferring it.

**Do NOT reach for a bigger/faster GPU model** — a 6 GB card already holds
qwen2.5:3b-q8_0 (~3.6 GB) + the reranker fp16 (~1.2 GB). There is no headroom.

### Then, and only then
Re-run the full `scripts/eval_answers.py` and update the measured table in
`CHECKLIST.md`. Phase 4 (auth/roles/audit) is PLANNED in `PLAN.md` but not
started — do not begin it until the owner says so (standing agreement, §7).

**Note for JOB 2:** the corpus SQLite changed under Phase 3 (`gr_number` was
re-trimmed on 1,069 documents, `refs` on 15,741, and `gr_edges` was rebuilt).
Vectors and chunks were NOT touched, so retrieval numbers are still comparable
with the ones in `CHECKLIST.md`.

---

## 1. What this is

A **multilingual, source-grounded question-answering system over Maharashtra
Government Resolutions (GRs)** — VJTI AI Hackathon 2026, problem statement PS-3.

Ask in **English or Marathi**; it retrieves relevant GRs and answers **only from
them**, puts a **citation on every claim**, **abstains** when the corpus doesn't
cover the question, and **runs entirely on this machine** (no document ever
leaves it — that is the on-premise/NIC story the SRS asks for).

Stack: Python **FastAPI** + a RAG engine ported from a prior project
(**ResearchOS**) and adapted — disclosed in `README.md` — plus a **Next.js**
portal. The formal spec is the hackathon SRS, a Word document kept OUTSIDE the
repo at `/home/soham/Downloads/PDFS/SRS_AI_Powered_Question_Answering_System_final.docx`.

"Phase N" always means a phase in `PLAN.md` unless it says "ROADMAP Phase N".

---

## 2. Current state at a glance

- **Index — TWO of them, selected by `VECTOR_BACKEND`:**
  - `VECTOR_BACKEND=hnsw` (**the corpus, PLAN Phase 2 — the one to demo**):
    `/mnt/win/mahagr/index/` = `corpus.hnsw` (FAISS `IndexHNSWFlat`) +
    `corpus.db` (SQLite: documents, chunk text, FTS5 BM25). **99,410 GRs /
    401,573 chunks across ALL 33 departments** — the complete orgpedia dataset.
    Source text at `/mnt/win/mahagr/corpus/` (1.13 GB, 99,421 files).
    Re-ingested 2026-08-07 for table extraction; the pre-table index is kept at
    `/mnt/win/mahagr/index_v1/` and can be deleted once the new one is trusted.
    Ignore `corpus.index` / `corpus_meta.json` in that folder — they are an
    unused flat-store leftover, not part of the hnsw corpus.
  - `VECTOR_BACKEND=flat` (**default**): `backend/index/` = 713 vectors
    (196 HTE GRs + 2 synthetic fee-table GRs). The original demo index; still
    used by the model-free tests and the fixture/supersession demo.
- **Models:** `BAAI/bge-m3` (embeddings, 1024-d) + `BAAI/bge-reranker-v2-m3`
  (cross-encoder reranker), cached in `~/.cache/huggingface` (6.7 GB, on root).
- **LLM: fully local** via Ollama — `qwen2.5:3b-instruct-q8_0`, 8192-token
  context, models stored on `/mnt/win`. Groq (`llama-3.1-8b-instant`) remains a
  hosted fallback, selected by `LLM_PROVIDER`. Offline operation is **proven**,
  not asserted, by `backend/scripts/verify_offline.py`.
- **Hardware:** RTX 4050 laptop GPU, **6141 MiB VRAM** (the binding constraint on
  every model decision), torch 2.13.0+cu130.
- **Retrieval quality — TWO DIFFERENT NUMBERS, do not mix them up:**
  - *196-GR flat index:* hit@1 19/20, hit@5 20/20, MRR 0.975.
  - *18,078-GR HNSW corpus (2026-08-06):* **hit@1 12/20, hit@5 14/20, MRR 0.642**,
    answer CORRECT 11/20, **GROUNDED 20/20**, latency **p50 11.4 s, 12/23 over
    the SRS 10 s target**. Scaling degraded accuracy and latency; groundedness
    and abstention held. See `CHECKLIST.md` for the full table and the likely fix.
  - Thresholds are STILL `text 0.55` / `rerank 0.85`. `eval_retrieval.py` now
    suggests ~0.994 — **deliberately NOT applied**: the KEEP and REJECT score
    distributions genuinely OVERLAP at 41k documents (relevant p10 0.946 vs
    reject median 0.980), so 0.994 sits above the 25th percentile of RELEVANT
    hits and would abstain on ~a quarter of answerable questions. At this
    corpus size relevance is no longer bimodal, which is the finding.
  - `HNSW_EF_SEARCH` is **4096** (recall@60 0.997, recall@10 1.000 at 401,573
    vectors). It has decayed TWICE: 512 measured 0.986 at 74k and 0.962 at 157k;
    1024 measured 0.980 at 157k and 0.967 at 402k. **An ANN parameter is a
    function of index size, not a constant, and nothing reports when it goes
    stale.** At this scale exact brute force is 35.7 ms/query vs 1.1 ms for
    HNSW — the first corpus where ANN is a real speed win, not just scaling.
- **Answer quality (`scripts/eval_answers.py`, local qwen2.5:3b-q8_0) — again,
  TWO SETS OF NUMBERS. Quote the one matching the index you are running:**

  | | 196-GR **flat** | 18,078-GR hnsw | **41,474-GR hnsw (CURRENT)** |
  |---|---|---|---|
  | CITED | 19–20/20 | 15/20 | **19–20/20** |
  | GROUNDED | 20/20 | 20/20 | **20/20** |
  | CORRECT | 19–20/20 | 11/20 | **14–15/20** |
  | DEGENERATE | 0/20 | 0/20 | **0/20** |
  | ABSTAINS | 2/3 | 2/3 | **2/3** |
  | latency p50 | 4.7 s | 11.4 s | **13.3 s** |
  | over 10 s | 2/23 | 12/23 | **16/23** |

  The 41k column is Phase 6 wave A **plus** four config corrections found while
  measuring it (`LLM_MAX_TOKENS` 2048→768, `LLM_CONTEXT_BUDGET`→3000,
  `abstain_floor`, and loading the LLM fully onto the GPU). See `CHECKLIST.md`.

  Ranges are run-to-run variance at `temperature=0.2` over only 20 questions — a
  difference of 1 is noise. On the hnsw corpus, 3 of the CORRECT failures are the
  synthetic fee-table fixtures, which live only in the flat index and are
  structurally unanswerable there, so the fair figure is ~11/17. **Fixing both
  the accuracy and the latency gap is §0 JOB 2.**
- **Tests:** **157, all passing.** Model-free, ~1 s, no GPU, no network. Was 46
  before Phase 2, 122 after Phase 3, +18 for table extraction, +17 for Phase 6
  (`reference_block`, the `EMBED_MAX_SEQ` OOM guard, the hard abstention gate,
  date validation, the filtered-efSearch boost). **Phase 4 (auth) still has
  none.**
- **Venv:** `backend/.venv` (Python 3.11), `pip install -e .` already done.
- **Disk (critical ops fact):** root `/` is 93 G and **98% full (~2.1 G free)**.
  A 376 G NTFS **Windows** partition is mounted at **`/mnt/win`** with
  **~137 G free** (older notes say 47 G — that was freed up since) and holds the
  corpus, the index and the Ollama models. **Do NOT put the HF cache on NTFS** —
  it uses hard/symlinks that NTFS breaks.
- **Git:** remote `github.com/soham555-maker/mahagr-assist`. Last commit is
  `f814fc3`. **ALL Phase-1 AND Phase-2 work is UNCOMMITTED in the working tree**
  (the owner commits explicitly; see §7 for the full file list). Local history
  was scrubbed of `Co-Authored-By: Claude`; the remote still has the old commits
  and needs a `git push --force origin main` from the owner.
- **Where the big files live (none of them on root `/`):**
  | Path | What |
  |---|---|
  | `/mnt/win/mahagr/corpus/<Department>/*.mr.txt` | 18,078 raw GRs, 244 MB |
  | `/mnt/win/mahagr/index/corpus.hnsw` | the FAISS HNSW graph |
  | `/mnt/win/mahagr/index/corpus.db` | SQLite: documents, chunk text, FTS5 |
  | `/mnt/win/mahagr/ollama/` | Ollama models |
  | `~/.cache/huggingface` (6.7 GB, **on root**) | bge-m3 + reranker |
  | `backend/index/` | the old 713-vector flat index |
  | `backend/data/db/mahagr.db` | portal state (conversations/feedback) |

---

## 3. Repo map, file by file

Paths are relative to the repo root `/home/soham/Projects/mahagr-assist/`.

### Root documents
| Path | What it is |
|---|---|
| `README.md` | Overview, setup, ResearchOS-port disclosure |
| `PLAN.md` | **The plan being implemented now** — 5 phases, with interview Qs |
| `CHECKLIST.md` | **NEW.** Per-phase done/not-done with the proof for each claim |
| `ROADMAP.md` | The earlier initial-build checklist (7 phases, mostly done) |
| `DEMO.md` | Presenter cheat-sheet |
| `DEPLOYMENT.md` | ROADMAP Phase-7 deployment runbook (Docker, on-prem + cloud) |
| `HANDOFF.md` | This file |
| `deploy/ollama.service` | **NEW.** systemd *user* unit for the Ollama server. Not installed yet — see §7 |
| `Original Maha-GR/` | 16 real government GR PDFs (gitignored). **Their text layer is garbled** (broken font encoding) → need OCR re-ingest after `tesseract-data-mar` is installed |

Presentation artifacts live **outside the repo**, in
`/home/soham/Projects/Claude/vjti-hackathon/` (`MahaGR_PA3_Presentation.pptx`,
`architecture.svg`, `DEMO_VIDEO_SCRIPT.md`, deck generators).

### `backend/engine/` — the RAG engine
- **`corpus_db.py`** — **NEW (Phase 2).** The document/metadata half of the
  scaled corpus: SQLite `gr_documents` + `gr_chunks` (+ an FTS5 index for BM25).
  `gr_chunks.faiss_id` **is** the vector's position in the FAISS index — that
  integer is the entire join between the two files. Kept separate from
  `app/db.py` for a layering reason, not a stylistic one: `retrieval.py` reads
  it on every query, and `engine/` must never import `app/`.
- **`config.py`** — single source of truth for model choices, **and the one place
  `backend/.env` is loaded**. The load must stay here and stay first: the
  settings below are read from `os.environ` at import time, so loading `.env`
  later (as `rag.py` used to) silently ignored them. Holds `EMBED_MODEL=BAAI/bge-m3`,
  `EMBED_DIM=1024`, `QUERY_PREFIX=""`, `RERANK_MODEL=BAAI/bge-reranker-v2-m3`,
  `OCR_LANGS`, `EMBED_DEVICE` (embedder), `RERANK_DEVICE` + `RERANK_FP16`
  (cross-encoder — deliberately separate, see §5.7), and the LLM seam
  (`LLM_PROVIDER`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_NUM_CTX`).
- **`ingest.py`** — `IngestionPipeline`: `extract_pages_from_pdf(force_ocr=)`
  (text layer, OCR fallback, `force_ocr` for the broken-font PDFs), `_ocr_page`
  (Tesseract), `process_pdf` (text + pdfplumber tables), `process_text` (reads
  orgpedia `.mr.txt` with `# Page N` markers), `chunk_pages`. Honours `EMBED_DEVICE`.
- **`vector_store.py`** — TWO stores, chosen by `config.VECTOR_BACKEND`:
  - `FaissStore` (`IndexFlatIP`) — exact, brute force, O(n) per query, all chunk
    text in a RAM JSON sidecar. Still backs the 713-vector demo index and the
    model-free tests.
  - **`HnswStore` (`IndexHNSWFlat`) — NEW (Phase 2).** ~O(log n) graph search and
    **no text at all**: `search()` returns positions + scores, and
    `corpus_db.py` turns positions into text. That is what keeps RAM flat as the
    corpus grows. Supports **native filtered search** via
    `SearchParametersHNSW` + `IDSelectorBatch` (see §5.21).
- **`retrieval.py`** — `RetrievalConfig` (calibrated thresholds, `rerank_pool=15`),
  BM25 `KeywordIndex`, `Retriever` (dense + BM25 fused by RRF, optional
  cross-encoder rerank, abstention via a score floor), **`CorpusRetriever` (NEW
  — the two-stage scale path: ANN → BM25 → RRF → hydrate from SQLite → group by
  GR → rerank → gate)**, `load_default_retriever` (dispatches on
  `VECTOR_BACKEND`). Both retrievers expose the SAME `retrieve()` contract, so
  `rag.py` / `officer.py` / `api.py` are backend-agnostic.
- **`hybrid.py`** — Devanagari-aware `tokenize` (`[a-z0-9ऀ-ॿ]+`) + `rrf_fuse`.
- **`reranker.py`** — `Reranker` (bge-reranker-v2-m3). Honours `RERANK_DEVICE`
  and loads **straight into fp16** on CUDA (never `.half()` afterwards — §5.7).
- **`rag.py`** — the generation half. `GenerationConfig` carries the provider seam
  plus **three values resolved per-provider in `__post_init__`**:
  `context_token_budget` (groq 1300 = rate-limit bound / ollama 6000 =
  context-window bound), `tokens_per_devanagari` (2.0 Llama / 1.2 qwen) and
  `compact_prompt`. Also: `estimate_tokens` (Devanagari-aware **and
  tokenizer-specific**), `trim_to_budget`, `SYSTEM_PROMPT` (long, for large
  models) and **`SYSTEM_PROMPT_COMPACT`** (short, for small local models),
  `final_reminder` / `final_reminder_compact`, `language_directive` (maps portal
  ISO codes to real language names), `warn_if_over_context` (guards Ollama's
  silent truncation), `build_prompt`, `parse_citations`, `resolve_citations`,
  `make_client` (groq|ollama), `OllamaClient` (stdlib only), `call_llm`
  (provider-aware, 429 backoff), `rewrite_query`, `answer`.
- **`graph.py`** — **NEW (Phase 3).** The supersede/citation knowledge graph:
  `build_edges` (one SQL pass over the `refs` Phase 2 already stored; no model,
  no GPU), `neighbourhood`, `supersede_chain` (transitive, newest-first),
  `stats`. All traversals are cycle-guarded. **Not Graph RAG** — edges are
  parsed deterministically and used for provenance/conflict warnings; they do
  not drive retrieval.
- **`gr_metadata.py`** — `extract()` → gr_number (शासन निर्णय / आदेश / परिपत्रक /
  अधिसूचना), ISO date (Marathi digits + month names, and the DOTTED numeric form
  GR reference lines actually use), department, category, language, references,
  `reference_details` ({number, date}), supersedes. **The reference parser is the
  part with the sharp edges — see §5.28.**
- **`officer.py`** — `summarize` / `explain` / `compare`, `supersession` (metadata
  graph), `supersede_warnings` (conflict check surfaced on `/ask`), `related`,
  `document_chunks`.
- **`text_table_detect.py`** — **NEW (2026-08-07, PLAN "Phase 2.5").** Finds
  tables in **already-OCR'd text**, where pdfplumber has no structure left to
  read, and rewrites each row through `table_extract.row_to_sentence` so a fee
  question matches prose instead of a jumbled word stream. Pipe tables appear in
  **7,611 GRs (42%)**, dash-separated in 1,805. Thresholds are deliberately
  strict (≥3 pipes/line, ≥2 rows, consistent pipe count) because a false
  positive is worse than a false negative here. Wired in via
  `ingest.chunk_text`.
- **`table_extract.py`, `visual_ingest.py`** — ported (tables / figure layout).
- **`documents.py`** — **LEGACY** (ResearchOS Supabase path). Unused. Do not extend.

### `backend/app/` — the HTTP layer
- **`api.py`** — **the FastAPI app actually in use.** Lifespan loads the index +
  bge-m3 + reranker once, and calls `db.init()`. Routes: `/health` `/documents`
  `/documents/{id}/text` `/ask` (persists the turn, loads DB history into the
  query rewrite, adds conflict warnings) `/conversations` (GET/DELETE)
  `/conversations/{id}` `/feedback` `/summarize` `/explain` `/compare`
  `/supersede/{id}` `/related/{id}`. No auth (that is PLAN Phase 4).
  **Phase-2 changes:** new **`GET /corpus/stats`** (documents, chunks,
  per-department counts, date span — powers the portal's corpus line and the
  filter options; computed once at startup and cached, it is a COUNT over 18k
  rows). `/health` now also reports `vector_backend`, `documents`,
  `departments`. **`/documents` is paginated** (`q`, `department` (repeatable),
  `limit`, `offset`). **`/ask` accepts scope filters** (`departments`,
  `date_from`, `date_to`, `doc_language`) via `AskReq.filters()`.
  `/documents/{id}/text` returns the stored full text on the corpus backend
  rather than re-stitching overlapping chunks.
- **`auth.py`** — **NEW (Phase 4, 2026-08-07).** bcrypt password hashing, JWT
  issue/verify (PyJWT, HS256, 1-day expiry), the `get_current_user` dependency,
  `require_role([...])`, and an **in-process** token-bucket rate limiter
  (20 req/min/user). `DEV_NO_AUTH=1` short-circuits all three — it is **ON in
  `.env` today**, so nothing is actually gated at runtime.
- **`db.py`** — SQLite: conversations, messages (incl. sources + warnings),
  feedback, and (Phase 4) `users` + `audit_log`. Path from `MAHAGR_DB`, default
  `data/db/mahagr.db`. Note the deliberate split: portal state lives here,
  corpus state lives in `engine/corpus_db.py`, and `engine/` never imports
  `app/` — so a corpus rebuild can never touch the audit trail.
- **`main.py`** — **LEGACY**, unused.

### `backend/scripts/`
| Script | Purpose |
|---|---|
| `fetch_mahgrs.py` | **REWRITTEN (Phase 2).** Multi-department, resumable, threaded corpus download. Uses the **git trees API** (1 request/department) — the contents API pages at 100 and would burn GitHub's 60/hour limit just listing. `--list`, `--cluster education`, `--dry-run` |
| **`ingest_corpus.py`** | **NEW (Phase 2).** Builds the scaled index: chunk → GPU batch-embed → FAISS + SQLite. **Resumable, idempotent, crash-consistent** (see §5.22) |
| `ingest_text.py` | Build the index from `.mr.txt` |
| `ingest_grs.py` | Build a fresh index from PDFs |
| `add_pdfs.py "<dir>" [index] [--ocr]` | Append PDFs; `--ocr` forces OCR |
| `add_fixtures.py` | Append the 2 synthetic fee-table GRs |
| `ask.py` | One-off CLI query |
| `officer_tools.py` | CLI for the officer features |
| `eval_retrieval.py [--no-rerank]` | Gold-set retrieval eval + threshold calibration |
| **`eval_answers.py`** | **NEW.** Scores the *generated answer*: cited / grounded / correct / degenerate / abstains + latency vs the 10 s NFR. **Needs the API running.** |
| **`verify_offline.py`** | **NEW.** *Proves* on-prem: blanks the Groq key and makes every non-loopback socket raise, then answers 2 questions. **Run with the API stopped** (it holds the GPU). |
| `make_fixtures.py` | Generate the reportlab fixtures |
| `verify_demo.py` | Verify the scripted demo questions |
| **`build_graph.py`** | **NEW (Phase 3).** Builds `gr_edges` from the corpus metadata. Seconds, no GPU, idempotent. `--show <gr>` prints one GR's neighbourhood + supersede chain. **Run; 60,420 edges** |
| **`reparse_refs.py`** | **NEW (Phase 3).** Re-parses `gr_number` / `refs` / `supersedes` from `gr_documents.text` — a re-PARSE, not a re-ingest (~3 min CPU vs ~25 min GPU). `--dry-run` reports the diff without writing. Run `build_graph.py` after it |
| **`seed_users.py`** | **NEW (Phase 4).** Seeds the four SRS roles — `admin1` (IT Admin), `reviewer1`, `translator1`, `officer1`, all with password `password123`. Idempotent |
| **`verify_corpus.py`** | **NEW (Phase 6).** Proves the FAISS index and SQLite still AGREE. `ingest_corpus.py` only checks that the counts match, and equal counts do not prove equal *alignment* — an off-by-500 shift mid-corpus leaves both numbers identical while every citation names the wrong GR. This re-embeds a random sample and scores each chunk against **its own** vector via `score_for_index` (a search could return a near-duplicate neighbour and mask the shift). Correct ≈ 0.9998; misaligned ≈ 0.3–0.8. CPU, ~1 min, safe to run with the API up |
| **`cited_departments.py`** | **NEW (Phase 6).** Counts which department is named in each GR's `वाचा` block, to pick the ingestion order by evidence. Measured: **Finance is cited by 32.4% of the corpus and is one of the smallest departments in the dataset** — while Revenue & Forest (0.5%) and Law & Judiciary (0.3%), which this file used to name as the main gap, are near-irrelevant. Read-only, no GPU, ~1 min |
| **`eval_ann.py`** | **NEW (Phase 2).** Calibrates HNSW `efSearch` — recall vs exact brute force, using REAL gold-set query embeddings. Reports recall@10 *and* recall@60 (60 = the real candidate-pool size). See §5.26 for why this script exists |

### `backend/tests/` — 122 tests, model-free
`conftest.py` (stubs `sentence_transformers` so the suite needs no torch/GPU, and
puts `backend/` on the path), `test_hybrid`, `test_gr_metadata`, `test_chunking`,
`test_rag`, `test_officer`, `test_config`, `test_llm_provider`, `test_db`,
plus the Phase-2 additions:
- **`test_corpus_db.py`** (14) — schema, chunk hydration, FTS5 BM25 on Devanagari,
  FTS5 operator-character safety, facet filters, idempotent re-ingest, and the
  **crash-recovery** contract.
- **`test_hnsw_store.py`** (13) — id contract (`add_vectors` returns the start
  position), filtered search via `IDSelector`, empty-filter-abstains, exact
  `score_for_index`, save/load, resume, and a recall-vs-exact floor.
- **`test_corpus_retrieval.py`** (9) — the two-stage pipeline on a real tmp
  HNSW+SQLite corpus with a stubbed model: hydration, the per-GR cap, filter
  push-down, the rerank threshold gate and floor, and DB/index drift tolerance.

- **`test_gr_metadata.py`** — Phase 3 added 10 cases for REFERENCE extraction,
  every one of them a verbatim line from the corpus: a number with internal
  spaces, a number that must not swallow the date after it, the department that
  sits between two `क्र.` labels, the inner `प्र.क्र.` that must not be mistaken
  for a label, dotted dates, and prose-with-a-slash.
- **`test_graph.py`** (30) — Phase 3: GR-number canonicalisation (OCR variants
  agree, distinct numbers stay distinct), edge resolution incl. dangling and
  ambiguous, idempotent rebuild, transitive supersede chains, **cycle guards**,
  neighbourhood node caps, the schema backfill, date-based disambiguation of a
  shared GR number (and the refusal to guess when the date fits nobody), and
  backwards compatibility with a corpus whose `refs` are plain strings.

FAISS, numpy and SQLite are real in the tests; only the *models* are stubbed.

### `backend/data/`
`grs_text/` (196 orgpedia `.mr.txt`, gitignored, present) · `fixtures/`
(`GR-2023-fees.pdf`, `GR-2024-fees-revised.pdf`, synthetic, + README; the 2024
one supersedes the 2023 one, which is what demos conflict detection) ·
`gold/gold.json` (**23-question gold set**; entries are `{q, lang, expected[],
in_corpus}` — note the key is `q`, not `question`) · `db/` (runtime SQLite) ·
`grs/.gitkeep`.

### `frontend/` — Next.js 14

> **RESTRUCTURED 2026-08-07.** The portal was rebuilt on **shadcn/ui + Radix +
> lucide** (`components/ui/*` — 19 primitives, `components.json`, `lib/utils.ts`,
> `tailwindcss-animate`). Current layout:
>
> | Route | File | What |
> |---|---|---|
> | `/` | `app/page.tsx` (883 lines) | **Public landing page** — architecture diagrams, the on-prem story, a live corpus stat. *This used to be the Ask page.* |
> | `/login` | `app/login/page.tsx` | Phase-4 login; stores the JWT + role in `sessionStorage` (not `localStorage` — XSS reach) |
> | `/ask` | `app/(portal)/ask/page.tsx` | the grounded chat |
> | `/browse` | `app/(portal)/browse/page.tsx` | document browser + graph panel |
> | `/admin` | `app/(portal)/admin/page.tsx` | audit log; role-gated **on the server**, the client check is convenience only |
>
> Shared portal chrome is `app/(portal)/layout.tsx`; new components are
> `UserMenu.tsx`, `nav-link.tsx` and `components/landing/{diagrams,
> live-corpus-stat,site-chrome}.tsx`. `npx tsc --noEmit` clean (2026-08-09).

The descriptions below are still accurate about *behaviour*; only the file
paths moved.

- `app/(portal)/ask/page.tsx` — **Ask**: grounded chat, EN/मराठी toggle, "Explain simply" mode
  (`/explain`), abstention + conflict banners, conversation-history sidebar,
  thumbs feedback.
- `app/(portal)/browse/page.tsx` — **Browse**: search, read full text, Summarize button,
  Supersession + Related panels, Compare two GRs.
- `lib/api.ts` (typed client; `Scope`/`CorpusStats` types, `qs()` query-string
  helper, paginated `documents()`) · `components/ui.tsx` (shared UI: `LangToggle`,
  `SourceCard`, plus the Phase-2 **`useCorpusStats`**, **`CorpusStat`**
  ("Searching N GRs across M departments"), **`ScopeFilter`** (department/date/
  language, collapsed behind one button) and **`ScopeNotice`** (per-answer
  "Searched only: …" so a narrowed search is never invisible)) ·
  `app/layout.tsx` · `.env.local` (`NEXT_PUBLIC_API_URL`, default
  `http://localhost:8000`).
- **Browse is now server-paginated** (50/page + "Load more") with department
  chips. It used to fetch every document and filter in the browser — a
  multi-MB page load at 18k GRs.
- Verified with `npx tsc --noEmit` **and** `npx next build` (both clean).
  `components/ui.tsx` is now `"use client"` because it holds hooks.

---

## 4. Status mapped to PLAN.md

**Initial build (ROADMAP.md): ✅ done** — multilingual engine, officer features
(summarize/explain/compare/supersede/related + conflict warnings), SQLite
persistence, the portal, and calibrated retrieval. *Still outstanding from
ROADMAP:* OCR-ingest the 16 original PDFs (blocked on `tesseract-data-mar`).

**PLAN Phase 1 — Local LLM (Ollama): ✅ DONE** (one known gap, below).
Model `qwen2.5:3b-instruct-q8_0` on `/mnt/win`, 8192 context, `.env` configured,
backend serving locally, offline **proven**. Three silent bugs and two quality
bugs were found and fixed on the way — all detailed in §5. Full write-up with
before/after numbers is in `CHECKLIST.md`.
*Known gap:* one out-of-corpus question (farm loan waiver) is still answered
instead of refused, even though retrieval correctly flags it
(`low_confidence=True`, scores ~0.001). Groq abstains correctly there. A 3B model
occasionally stretches weak context despite a decisive instruction.
*Not done:* the Ollama systemd unit is written but **not installed** (§7).

**PLAN Phase 2 — Multi-department corpus, FAISS-HNSW + SQLite: ✅ DONE.**
The headline scale work. `IndexFlatIP` + RAM JSON → `IndexHNSWFlat` + chunk
text/metadata in SQLite; the six education-cluster departments (**18,078 GRs**)
ingested; retrieval rewired to two-stage (ANN → BM25 → RRF → hydrate → group by
GR → rerank); department/date/language filters pushed natively into the FAISS
search; BM25 moved from in-RAM `rank_bm25` to SQLite FTS5. Tests 46 → 82.
Full write-up with measured numbers in `CHECKLIST.md`; the non-obvious traps are
§5.20–§5.26 below.

**PLAN Phase 3 — Supersede knowledge graph + visualization: ✅ DONE (built,
run and verified).** Written while the Phase-2 ingest still held the GPU, because
none of it needs the corpus or a model — Phase 2 already persisted every
document's parsed `references` and `supersedes` flag, so the graph is one SQL pass.
Running it on the real corpus exposed that the *extraction* feeding it was
broken (2% of references resolved); that is fixed and re-parsed — see §0 JOB 1
and `CHECKLIST.md`. Graph now: **60,420 edges, 5,753 resolved (9.5%), 3,904
documents linked**.
*Done:* `gr_edges` table + a real schema migration (`corpus_db._migrate`, so an
existing 18k corpus is NOT re-embedded for a metadata-only change),
`gr_metadata.canonical_number()`, `engine/graph.py` (edge building with
three-valued resolution, cycle-safe neighbourhood, transitive supersede chain),
`scripts/build_graph.py`, `GET /graph/{doc_id}`, `GET /graph/stats/summary`.
`officer.supersession` now reads the graph (falling back to the old scan when it
has not been built), `/ask` conflict warnings use the TRANSITIVE chain, and
`frontend/components/graph.tsx` draws the neighbourhood as hand-rolled SVG
(+1.65 kB, vs 100-400 kB for react-flow). **26 tests.** See §5.27.
*Then run for real:* `scripts/build_graph.py` on the 18k corpus, which is what
exposed the broken reference extraction behind it (§0 JOB 1, now fixed and
re-parsed). Graph today: **60,420 edges / 5,753 resolved (9.5%) / 3,904 linked
documents**, and `scripts/reparse_refs.py` exists to re-derive `refs` from the
stored text without a re-ingest. **Tests for Phase 3: 40.**

**PLAN Phase 2.5 — Table extraction from OCR'd text: ✅ DONE (2026-08-07,
unplanned).** `engine/text_table_detect.py` turns pipe/dash tables in the OCR'd
`.mr.txt` into `row_to_sentence` prose, because pdfplumber has nothing to work
with once a PDF has been flattened to text. Forced a full re-ingest: chunks
64,744 → **74,004**, of which **17,736 are `content_type='table'`**. See
`PLAN.md` "Phase 2.5".

**PLAN Phase 4 — Admin dashboard (auth, roles, audit): 🟡 IMPLEMENTED
2026-08-07, NOT TESTED.** `app/auth.py` (bcrypt + PyJWT + `require_role` +
in-process token bucket + `DEV_NO_AUTH`), `users`/`audit_log` in `app/db.py`,
`POST /auth/login`, `GET /admin/audit-logs`, `current_user` on every route,
`/ask` writing an audit row (question + cited GR numbers + ip, deliberately not
the answer text), `scripts/seed_users.py` seeding the four SRS roles
(`admin1`/`reviewer1`/`translator1`/`officer1`, all `password123`).
**Caveats that must be stated, not glossed:** zero tests · `DEV_NO_AUTH=1` is
currently ON so nothing is gated at runtime · `JWT_SECRET` is the literal
`maha-secret` · the Reviewer role is not wired · the rate limiter is in-process
and resets on restart. See `PLAN.md` Phase 4.

**PLAN Phase 5 — UX polish + presentation refresh: 🟡 LARGELY DONE 2026-08-07.**
Portal rebuilt on shadcn/ui + Radix + lucide; `app/page.tsx` is now a public
landing page and the officer portal moved into a route group
`app/(portal)/{ask,browse,admin}`; `app/login/page.tsx`, `components/UserMenu.tsx`,
`components/nav-link.tsx`, `components/landing/*` are new. `npx tsc --noEmit`
clean (2026-08-09). Outstanding: mobile sidebar, uneven loading/error states, and
`DEMO.md` + the deck still describe the OLD single-page portal.

**PLAN Phase 6 — Full corpus + complete graph: ✅ COMPLETE (2026-08-10).**
All 33 departments / **99,410 GRs / 401,573 vectors**; graph **317,250 edges,
46,412 resolved, 30,799 documents linked**. The headline finding is a NEGATIVE
one and it is provable: on the unchanged education documents, wave A's 23,394
new documents bought **+2,886** resolved edges while wave B/C's 57,939 bought
**+20** — ~144 per 1,000 documents vs 0.35, a 400x difference, exactly as
`scripts/cited_departments.py` predicted. And because the corpus is now
complete, the 14.63% resolution rate is an ABSOLUTE ceiling: **76% of references
in Maharashtra's published GRs point outside the published dataset.**
Scaling cost almost nothing — 2.4x the documents, hit@5 unchanged at 18/20.

*(wave A, 2026-08-09, kept for the reasoning:)*
The 6 most-cited missing departments ingested: **18,080 → 41,474 GRs**,
74,004 → **156,795 vectors**, graph **60,421 → 120,224 edges** with **resolved
6,267 → 17,935**. On the *unchanged original documents alone*, resolution went
**10.37% → 15.15% (+46%)** with no parser change — the clean proof that the
graph was limited by corpus coverage. Ingestion order was measured, not guessed
(`scripts/cited_departments.py`): **Finance is cited by 32.4% of the corpus and
cost only 1,046 GRs**, and it became the #1 citation target overall.
**Waves B/C — the remaining 21 departments to 99,421 GRs — not started.**
Full tables and the four bugs this surfaced: `CHECKLIST.md`.

---

## 5. Non-obvious decisions, rejected alternatives, and gotchas

Everything here cost real debugging time and is **not** recoverable from reading
the code.

1. **Fresh repo, engine ported from ResearchOS, openly disclosed** — to satisfy
   the hackathon's originality concern (new domain, new git history).

2. **Translate at the edges, not at ingest.** The index stores the *original
   Marathi*, so citations point at the real GR text. Cross-lingual retrieval
   works through bge-m3's shared vector space, not a translation step.

3. **The bge-m3 swap was not a one-line change.** It forced dim 384→1024,
   **removal of the query prefix** (bge-*-en-v1.5 was trained to expect an
   instruction prefix; bge-m3 must NOT have one — keeping it degrades every query
   silently, with no error), a multilingual reranker, the Devanagari tokenizer
   fix, and full threshold recalibration.

4. **Devanagari BM25 tokenizer.** `\w` splits Marathi at vowel marks
   ("शासन" → "श","सन") and an ASCII-only pattern dropped it entirely. Correct
   pattern: `[a-z0-9ऀ-ॿ]+`.

5. **Devanagari token density, and why it is per-TOKENIZER.** Groq's free tier
   caps 6000 tokens/min; Marathi is ~2 tokens/char under Llama's byte-BPE, so an
   English `words×1.33` estimate under-counted ~10× and requests hit 6629 tokens
   → 413 errors. That is why `estimate_tokens` counts Devanagari separately.
   **But the rate is a property of the tokenizer, not of Marathi.** Measured on
   qwen2.5 via its reported `prompt_tokens`: **1.09 tokens/char**, not 2.0.
   Applying Llama's 2.0 to qwen over-counted by 1.7×, so a single ~1900-char GR
   chunk "cost" 3264 of a 2200 budget and `build_prompt` **sent 1 block and
   dropped 10** — which is what made the local model answer with a bare `[1]`.
   Now `tokens_per_devanagari` is per-provider. Also: **Groq limits are
   per-account, not per-key** — a new key on the same account shares the quota.

6. **Groq model 70B → `llama-3.1-8b-instant`** — the 70B hit the per-day token
   limit; the 8B has a much larger bucket and is fine for grounded QA.

7. **`EMBED_DEVICE` vs `RERANK_DEVICE` — one switch for both models was wrong.**
   On a 6 GB GPU, Ollama + bge-m3 + reranker do not all fit, so `EMBED_DEVICE=cpu`
   moved *both* retrieval models to CPU. Measured, that was a mistake: the two
   have opposite cost profiles.
   - **Bi-encoder (embedder):** one short query per request, corpus side
     precomputed → **0.1 s on CPU**. CPU is fine.
   - **Cross-encoder (reranker):** 15 (query, chunk) pairs, nothing precomputable
     → **27.6 s on CPU vs 0.33 s on GPU (~80×)**.

   Retrieval without rerank was 0.1 s; with rerank, 37.5 s. This single issue was
   the entire latency problem (end-to-end p50 **38 s → 2.6 s** once fixed).
   The `~157 ms on CPU` note previously in `reranker.py` was measured on the
   *old* 22M English ms-marco MiniLM, not the 568M bge-reranker-v2-m3 — a stale
   comment that hid the regression.

   **fp16 must be LOADED, not converted.** `CrossEncoder(...)` then `.half()`
   leaves the discarded fp32 weights in torch's caching allocator: the process
   held **2766 MiB for a 1083 MiB model**. Passing
   `model_kwargs={"torch_dtype": "float16"}` costs **1174 MiB**. That recovered
   1.6 GB — precisely the difference between the 8-bit LLM fitting beside the
   reranker on a 6 GB card (3686 + 1174 = 4860 of 6141) or OOM-ing.

8. **Ollama on the Windows NTFS partition.** Root has no space, so models go to
   `/mnt/win/mahagr/ollama`. Ollama's blobs are plain files → fine on NTFS; the
   HF cache must NOT go there (hard/symlinks). Gotchas:
   (a) Windows **Fast Startup** leaves NTFS dirty → the partition mounts
   read-only; fully shut down Windows (`powercfg /h off`).
   (b) The installer registers a **system** systemd service running as the
   `ollama` user with default (root) paths — keep it disabled and run the server
   yourself. **`OLLAMA_MODELS` must be set on the SERVER, not on `ollama pull`**
   (the CLI is only a client; the daemon decides where blobs land).

9. **Ollama's context window silently truncates.** The default `num_ctx` is
   **4096**, and a longer prompt is not rejected — Ollama drops the oldest tokens,
   i.e. the system prompt carrying every grounding and citation rule, and returns
   a plausible-looking answer. Invisible from the outside. Set
   `OLLAMA_CONTEXT_LENGTH=8192` **on the server** and keep `config.OLLAMA_NUM_CTX`
   in step; `rag.warn_if_over_context()` shouts if a prompt would exceed it.
   Verify with `ollama ps` — its CONTEXT column is the truth.

10. **Prompt length is not free on a small model — this was the big one.**
    The bare-`[1]` answers looked like 3B incapacity. Isolating model from prompt
    on *identical* retrieved context:

    | prompt | completion tokens | output |
    |---|---|---|
    | full production prompt | **4** | `[1]` |
    | minimal two-line prompt | **99** | correct, cited, 3 sentences |
    | system prompt, no reminder | 25 | wrongly abstained |

    ~350 words of mostly-prohibitive rules collapse a 3B model into emitting only
    the token it is sure of — the citation. Hence `SYSTEM_PROMPT_COMPACT` plus a
    one-line reminder placed **after** the question, auto-selected when
    `provider == "ollama"`; Groq keeps the long, already-verified prompt.
    Result: DEGENERATE **8/20 → 0/20**.

    **Six prompt variants were measured before finding this, and each "fix"
    traded one failure for another** — do not repeat them:
    - Demanding citations harder ("an answer with no `[n]` is invalid") → the
      model **stopped abstaining** and hallucinated on out-of-corpus questions.
    - Adding a realistic worked example → the model emitted **the example itself**
      as a cited answer on an unrelated question (confidently wrong — the worst
      possible failure for a government tool).
    - Any literal the prompt shows gets copied: a `<topic of the blocks>`
      placeholder and a `... your sentence ... [1]` shape token were both echoed
      verbatim. **Show no literal you would not accept as output.**
    - Turning the citation rule up further pushed DEGENERATE to 16/20.

11. **Quantization was NOT the cause of the degenerate answers.** The upgrade
    `qwen2.5:3b` (Q4, 1.9 GB) → `qwen2.5:3b-instruct-q8_0` (3.3 GB) was tried as
    the fix and did **not** work on its own (DEGENERATE went 6 → 9). It is kept
    because it is better and now fits, but the real causes were §5.5 and §5.10.

12. **`context_token_budget = 6000` for ollama, tested against 3500.** 3500
    measured worse (CORRECT 17/20 vs 19–20/20) for ~1 s saved: the 2nd and 3rd
    block is often where the answer actually is. Groq stays at 1300, which is the
    *same real token volume* as its old 2200-at-rate-2.0 — so its rate-limit
    behaviour is unchanged.

13. **`/ask` silently dropped its `language` argument.** The portal's EN/मराठी
    toggle was a no-op on the main path (FR 3.4.2 / 3.4.5) — the field existed on
    the request model and was simply never passed to `rag.answer`. Also, the
    frontend sends ISO codes, so the instruction read *"Write the answer in mr."*;
    `language_directive()` now maps to real names ("mr" → "Marathi (मराठी)"),
    which fixed the same latent flaw in `/summarize`, `/explain`, `/compare`.

14. **Slim `app/api.py`, not the ported Supabase `main.py`** — on-prem friendly,
    no cloud dependencies.

15. **Fixtures are SYNTHETIC, built with reportlab** (LibreOffice HTML→PDF
    produced CID glyphs that broke pdfplumber). They exist to demo table/number
    retrieval and supersession, because the real orgpedia text has no structured
    tables. Disclosed as synthetic.

16. **Old ResearchOS tests deleted; a fresh model-free suite written.**
    `conftest.py` stubs `sentence_transformers` so tests need no torch or GPU.
    **Tests must not depend on this machine's `.env`** — two did, and broke when
    `LLM_PROVIDER` changed; they now pass an explicit `GenerationConfig`.

17. **`pip install -e .` is REQUIRED** — scripts do `from engine import ...`.

18. **Python 3.11 on purpose** (3.14 is too new for torch/faiss wheels).

19. **Scale decision for PLAN Phase 2 — stay fully local, no extra server.**
    Move FAISS-Flat + RAM JSON → **FAISS `IndexHNSWFlat`** (in-process HNSW,
    persisted to `/mnt/win`) + chunk text/metadata in **SQLite**. Postgres+pgvector
    and Qdrant were rejected *not* because they are cloud (they run locally too)
    but to avoid operating another server and to reuse the SQLite already in use.
    They remain the upgrade path if native metadata filtering at scale becomes the
    bottleneck. The earlier worry about "sending embeddings to a cloud server" was
    a misconception — all of these run on-machine.

20. **One GPU, two jobs, never at once.** Bulk ingestion embeds on the **GPU**
    with `EMBED_DEVICE` unset and **Ollama stopped**; serving runs the embedder on
    **CPU** and gives the GPU to the LLM + reranker. Stray `uvicorn`/`ollama`
    processes hold VRAM and will OOM the next thing you start — kill them first.
    **`.env` sets `EMBED_DEVICE=cpu`, and `load_dotenv` does not override a real
    environment variable — so `EMBED_DEVICE=cuda python scripts/ingest_corpus.py`
    is what actually wins.** Forgetting this is a ~10x slowdown that looks like
    normal slowness; `ingest_corpus.py` therefore PRINTS the device it got and
    warns on CPU. (This fired on the first smoke run — 40 documents took 4.3 min.)

21. **FAISS-HNSW *can* pre-filter — PLAN.md was out of date.** The plan assumed
    it couldn't and proposed over-fetch-then-filter. faiss 1.14 has
    `SearchParametersHNSW` + `IDSelectorBatch`; verified here that a filtered
    search returns only allowed ids and still fills top-10 at 1% selectivity. So
    the department/date/language filter is resolved to an id set in SQLite and
    pushed **into** the search — a post-filter would silently lose recall
    whenever the whole top-k belonged to an excluded department. Two real traps:
    (a) the SWIG wrapper does **not** own the selector or its id array, so a
    Python-side reference must be kept alive or FAISS reads freed memory;
    (b) the graph is still built over all vectors, so a narrow filter makes the
    walk traverse mostly-rejected nodes — `efSearch` is raised when a filter is
    active to compensate.

22. **The FAISS index and SQLite are two files that must agree, and cannot be
    written in one transaction.** `gr_chunks.faiss_id` IS the vector's position,
    so a drift of even a few rows makes every later citation name the wrong GR
    while looking perfectly plausible — the worst possible failure for a
    government tool. Resolution: **the saved FAISS index is the authority.**
    SQLite commits every batch, FAISS is checkpointed every N documents, so a
    crash leaves SQLite AHEAD; on the next run `corpus_db.delete_chunks_from()`
    drops every chunk row past the index's `ntotal` and re-ingests those
    documents. Getting this backwards is silent.

23. **BM25 had to move from `rank_bm25` to SQLite FTS5.** `rank_bm25` holds a
    tokenized copy of the whole corpus in RAM (a dict per document) — fine at
    713 chunks, several GB at 65k. FTS5 does the same BM25 ranking off disk in
    the database we already have. Verified FTS5's `unicode61` tokenizer handles
    Devanagari correctly (whole words, not split at vowel marks — HANDOFF §5.4's
    bug in its FTS form). **Trap: FTS5's MATCH syntax treats `-`, `/`, `.`, `*`
    and `OR` as operators**, so a raw question — or a GR number like
    `संकीर्ण-२०२३/प्र.क्र.४५` — raises `fts5: syntax error`. Every query is
    tokenized with `hybrid.tokenize` and quoted first.

24. **fp16 embedding is a 3.7x speedup for free — the single biggest ingestion
    win, and it was measured, not assumed.** The first full run was heading for
    ~2 hours. Benchmarked on 512 real Marathi GR chunks (avg 1570 chars):

    | dtype | batch | throughput | peak VRAM |
    |---|---|---|---|
    | float32 | 8 | 9.8 chunks/s | 2.62 GB |
    | float16 | 8 | 34.6 chunks/s | 1.32 GB |
    | **float16** | **16** | **36.6 chunks/s** | **1.49 GB** |

    Accuracy cost: cosine agreement between the fp32 and fp16 embedding of the
    same chunk is **min 0.99975, mean 1.00008** — orders of magnitude below the
    ~0.02–0.05 score gaps retrieval discriminates on, and the cross-encoder does
    the precise ranking afterwards regardless. Now `config.EMBED_FP16`
    (default on, CUDA only — fp16 on CPU is slower). Note this also means
    **`--batch-size 32` is not the answer to slow ingestion; fp16 is.** Raising
    the batch on fp32 just OOMs (§5.20); the card was compute-bound at ~100%.

25. **HNSW recall on REAL 1024-d embeddings is much worse than a synthetic
    benchmark suggests — `efSearch` is NOT yet calibrated.** A 64-dimension
    random-vector test reported recall@10 = 0.999 at `efSearch=128`. The same
    setting on 13,913 real GR chunks measured **0.735**:

    | efSearch | recall@10 | ms/query |
    |---|---|---|
    | 32 | 0.476 | 0.087 |
    | 128 (current default) | 0.735 | 0.140 |
    | 256 | 0.870 | 0.238 |
    | exact brute force | 1.000 | 0.951 |

    Two caveats, both real: those queries were RANDOM vectors, which in 1024-d
    are nearly orthogonal to everything and are ANN's worst case; and
    recall@10 is the wrong metric here — the pipeline takes
    `candidate_k_ann=60` chunks and BM25 feeds the pool independently, so what
    matters is recall@60. **`scripts/eval_ann.py` exists to settle this with
    real gold-set query embeddings and has NOT been run yet** (it needs the
    finished index). The economics are clear though: ANN costs a fraction of a
    millisecond against a ~5 s answer, so **buy recall — raise `efSearch`**.
    It is a query-time dial; changing it needs no re-ingest.

26. **PHASE 3: the hard part of the knowledge graph is REFERENCE RESOLUTION,
    not traversal.** Traversing a few thousand edges is trivial. Deciding
    whether `संकीर्ण-२०२३/प्र.क्र.४५/तांशि-४` printed in one OCR'd GR is the same
    order as a `gr_number` parsed from another is not: they differ by spacing,
    by Devanagari vs ASCII digits (`२०२३` vs `2023`), and by punctuation the
    scanner dropped. Hence `gr_metadata.canonical_number()`, and three things
    that are easy to get wrong:
    - **Keep `/`.** It is structural in a GR number (subject code / proposal
      number / desk code). Stripping it would merge unrelated orders and
      **fabricate** supersessions — far worse than missing one.
    - **Store the resolution three-valued** (`resolved` / `dangling` /
      `ambiguous`), and **keep the danglers**. "This GR builds on an order we
      do not hold" is information an officer needs, and the resolution rate is
      the honest measure of corpus completeness.
    - **A self-reference is neither an edge nor a dangler.** OCR variants of a
      GR's own number slip past `gr_metadata._references` (which only excludes
      by raw substring) and were being counted as dangling, inflating the
      "missing order" count. Caught by a test.

    **Cycles are real** — two GRs amending each other, or an ambiguous match —
    and every traversal carries a `seen` set plus a depth cap. Without them
    `/graph` is an infinite loop behind an HTTP request.

    **Schema evolution:** `CREATE TABLE IF NOT EXISTS` does NOT add columns to
    an existing table. `corpus_db._migrate()` + `backfill_canonical_numbers()`
    exist so a metadata-only change does not force a ~25 min re-embed of 18k
    documents.

27. **An external-content FTS5 table cannot be deleted from with `DELETE`.**
    It stores only the inverted index, so removing a row requires handing the
    ORIGINAL text back via `INSERT INTO t(t, rowid, text) VALUES('delete',…)` —
    and therefore must happen BEFORE the content row is deleted. A plain
    `DELETE FROM gr_chunks_fts` leaves terms pointing at rowids that no longer
    exist, and BM25 then returns ghost ids. Caught by a test, not by inspection.

28. **A GR number cannot be matched as a whitespace-bounded token — that single
    assumption is what made the knowledge graph nearly empty (2% of references
    resolved).** Real numbers contain spaces (`एनजीसी-२०१०/(१९३/१०) /मशि-४`), so
    the old pattern cut every one into a fragment. What actually delimits a
    number in a reference line is *punctuation and the date that follows it*, so
    the block is now parsed as items → comma-separated segments → each trimmed
    at its date. Four things that cost real time and are invisible in the code:
    - **`क्र` occurs INSIDE most GR numbers** (`प्र.क्र.४५`), so stripping the
      "क्रमांक :" label at the first match truncates the number to `४५`. The
      discriminator is the slash: a number's internal `क्र` always sits after
      one, an introducing label never does — so take the LAST label whose
      preamble contains no `/`. An earlier rule ("the label must be within 3
      words of the start") measured **7.9% vs 9.5%** resolution.
    - **`_parse_date` had no dotted form.** GR reference lines write
      `दि. ३०.१०.२०१०`, not `30/10/2010`; adding it took cited-date coverage
      **34% → 72%**. Guard the pattern with lookarounds, or a GR number like
      `१४०२५/११/२०२३` parses as a date.
    - **The cited date is a resolution tool, not decoration.** 2,138 canonical
      numbers in this corpus are held by more than one document; the date
      disambiguates **1,771** edges. When it matches none of the candidates the
      edge stays `ambiguous` — measured, the nearest candidate is usually more
      than a month away, i.e. genuinely a different order in the same series.
    - **Do not re-ingest for a metadata change.** `gr_documents.text` holds the
      full text, so `scripts/reparse_refs.py` re-derives number/refs in ~3 min
      on CPU. It deliberately does NOT rewrite `date` or `department`: those
      have better sources (the order id, the corpus folder) than an OCR'd
      header line — and `date` is exactly what the graph disambiguates on.

    `refs` now stores `{number, date}` dicts; `corpus_db.reference_entries()`
    reads that AND the old plain-string shape, so an older corpus keeps working
    without a re-embed.

29. **THE EMBEDDER'S SEQUENCE LENGTH IS AN OOM SURFACE, AND BATCH SIZE IS A
    DECOY.** A wave-A ingest died with CUDA OOM **15,000 documents in**, at
    `--batch-size 16` — the measured-safe value that had already embedded
    125,000 chunks. The cause was that `max_seq_length` was never set, so it
    inherited bge-m3's advertised **8192**. Attention memory is QUADRATIC in
    sequence length and **a batch pads to its longest member**, so peak VRAM is
    a function of the single worst chunk in the corpus, not of the batch size.
    One ~1,900-token chunk asked for a **1.76 GiB** tensor on a 6 GB card.
    It never fired on the education cluster; Public Health / Public Works
    annexures contain chunks that cluster simply did not have — i.e. **a new
    department can carry a new failure mode**.
    Fixed by `config.EMBED_MAX_SEQ=1024`. Measured: p50 494 / p95 662 / p99 768
    tokens, and only **92 of 129,320 chunks (0.071%)** exceed 1024. Those 92
    keep their full text in SQLite, in citations and in the cross-encoder —
    only the embedding sees the first 1024 tokens.

30. **`ollama serve` does not load the model. The first REQUEST does.**
    So "start Ollama before the API" is necessary but not sufficient: the model
    was still loading *after* uvicorn claimed 1.2 GB for the reranker, and got a
    **partial CPU offload — 1.9x slower decode** (generation p50 18.9 s → 10.1 s
    once fixed). `curl /api/tags` proves the daemon is up, NOT that the model is
    resident. Send one warmup `generate` between the two (§0 has the sequence)
    and confirm with `ollama ps` → `100% GPU` and `/health` →
    `llm_placement.fully_on_gpu: true`. **Never quote a latency number without
    checking that field.** Both fit: 3,734 + 1,179 = 4,913 MiB of 6,141 — the
    problem was ORDER, never capacity.

31. **Prompt SIZE is set by `context_token_budget`, not by `max_final_k`.**
    Halving `max_final_k` 12 → 6 left prompt tokens at **5,823, unchanged**,
    because `trim_to_budget` was already discarding those blocks — a clean
    negative result worth not repeating. The real lever is the budget:
    6000 → 3000 took prompt 5,823 → 2,398 and latency p50 34.1 → 17.2 s.
    §5.12's warning that 3500 measured WORSE did **not** reproduce here; that
    was the 196-GR era with `rerank_pool=15`, when the answer was often in block
    2-3. With `rerank_pool=40` and hit@1 15/20 the top blocks are usually right,
    which is exactly the condition that makes trimming safe. **Re-test
    calibrations after the corpus changes; do not inherit them.**

32. **ABSTENTION MUST NOT DEPEND ON THE MODEL — it is now a gate in code.**
    Measured with **5 repeats per question** (n=3 on one run is pure noise; the
    "2/3 → 1/3 → 0/3" that looked like a trend was one question flipping):

    | out-of-corpus question | top score | abstained |
    |---|---|---|
    | PhD aerospace fee (EN) | 0.028 | 4/5 |
    | Farm loan waiver (MR) | 0.825 | 0/5 |
    | Passport application (EN) | 0.006 | **0/5** |

    The passport question was answered with **fabricated Marathi prose carrying
    citations** while retrieval reported 0.006 and `low_confidence=True`. The
    signal was correct and available; the model ignored it. Since §5.10 records
    six prompt variants that each traded one failure for another, the fix is not
    a seventh prompt: `rag._hard_abstention` refuses below
    `config.abstain_floor` (**0.10**) *without calling the model*.
    - 0.10 is deliberately NOT near `rerank_threshold` (0.85): that gate decides
      which chunks are good enough to SHOW, this one decides whether to speak at
      all. Relevant top hits measure p10 **0.946**.
    - The refusal **cites nothing** — the premise of firing is that nothing
      retrieved is relevant.
    - It refuses in the language of the **question**, not of the retrieved text
      (an English question had been refused in Marathi).
    - Refusing got **faster** (~2 s vs ~13 s): the gate runs before generation.

33. **URL-ENCODE THE CORPUS PATHS — one real filename contains a SPACE.**
    `Revenue_and_Forest_Department/202510171720567619 .pdf.mr.txt` (space before
    `.pdf`). `urllib` rejects a URL with an unencoded space *before it makes any
    network call* — "URL can't contain control characters" — so `_get`'s retry
    loop could never help. It is a **deterministic** failure, which is the
    dangerous kind: the run reports `failed 1` of 99,421 and a 99,420-file
    corpus reads as a rounding error rather than a defect. `fetch_mahgrs.py` now
    quotes both the department and the filename (several department names also
    contain `,` and `-`).

34. **A CHECK YOU COMPUTE BUT NEVER ACT ON IS WORSE THAN NO CHECK.** The wave-B
    chain script computed the download failure count into `$FAILED` and then
    gated only on the exit code, so it proceeded to ingest an incomplete corpus.
    The wave-A version DID gate on `$FAILED` and would have stopped. The
    variable being present made the script *look* guarded. Grep for the variable
    you compute, not just the one you print.

35. **AN ANN PARAMETER IS A FUNCTION OF INDEX SIZE, NOT A CONSTANT.**
    `HNSW_EF_SEARCH` has now gone stale twice, silently, with nothing in the
    system reporting it:

    | corpus | setting | recall@60 |
    |---|---|---|
    | 74k vectors | ef=512 | 0.986 |
    | 157k vectors | ef=512 | **0.962** |
    | 402k vectors | ef=1024 | **0.967** |
    | 402k vectors | **ef=4096** | **0.997** (recall@10 1.000) |

    A greedy walk over a bigger small-world graph visits a smaller FRACTION of
    it for a fixed candidate budget. Re-run `scripts/eval_ann.py` after any
    large ingest. It is a query-time dial — no re-ingest. Also note exact brute
    force is **35.7 ms/query** at 402k vs 7.7 ms at 157k, so this is the first
    corpus where HNSW is a genuine speed win rather than only a scaling story.

36. **THE FILTERED-SEARCH efSearch BOOST SILENTLY BECAME A NO-OP.**
    `params.efSearch = max(ef, min(4 * ef, 1024))` was written when `ef` was
    128, so the 4x boost (512) sat comfortably under the 1024 ceiling. When `ef`
    was later raised to 1024 at corpus scale the expression collapsed to
    `max(1024, 1024)` — filtered searches lost the compensation §5.21 documents
    them as REQUIRING, and nothing failed. The ceiling now scales
    (`config.HNSW_EF_SEARCH_MAX`) and a test asserts the boosted value is
    strictly greater than the base. **A constant that bounds another constant
    must scale with it, or it becomes a silent equality.**

37. **PHANTOM CITATIONS CAN BE THE DOCUMENT'S OWN NUMBERING.** Measured 3 in 60
    answers (5%), one question reproducibly. The GR text reads
    *"संदर्भाधिन अ.क्र.४, ५ व ६"* ("referenced items no. 4, 5 and 6") and the
    model transcribes those into `[4] [5] [6]`. It is NOT inventing sources — it
    is echoing enumeration that exists in the source, and with only one block
    sent those numbers fall outside the valid range.
    The enabler is starvation: `sources=1, dropped=11` — one long Marathi chunk
    costs ~2,280 of the 3,000-token budget, so `trim_to_budget` keeps ONE block.
    That is §5.5's failure returning at the budget chosen to fix LATENCY.
    **Phantoms are still never silently stripped** — the detection IS the
    groundedness property, and here it correctly surfaced a real miscitation.

38. **A metric with n=3 cannot be tuned against.** ABSTAINS is scored over three
    questions; a single run distinguishes nothing. Two full config experiments
    were run against what turned out to be one question's sampling noise before
    this was measured properly with repeats. The repo's standing "a difference
    of 1 is noise" applies to the 20-question metrics; for the 3-question one,
    the ENTIRE range is noise. Repeat before concluding.

---

## 6. Commands

All backend commands are run **from `backend/`**. Either activate the venv
(`source .venv/bin/activate`) or call `.venv/bin/python` directly.

```bash
# ---------- one-time setup ----------
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e . && pip install -r requirements-dev.txt
sudo pacman -S tesseract-data-mar tesseract-data-hin tesseract-data-eng   # OCR (not installed yet)

# ---------- tests (fast, no GPU, no network, no API key) ----------
python -m pytest tests/ -q                    # 140 collected: 139 pass, 1 known fail

# ---------- build / refresh the small index (already built) ----------
python scripts/fetch_mahgrs.py --dept Higher_and_Technical_Education_Department \
    --count 200 --out data/grs_text
python scripts/ingest_text.py data/grs_text index
python scripts/add_fixtures.py
python scripts/add_pdfs.py "../Original Maha-GR" index --ocr   # after tesseract-data-mar

# ---------- build the SCALED corpus (PLAN Phase 2) — already built ----------
python scripts/fetch_mahgrs.py --list                 # departments + GR counts
python scripts/fetch_mahgrs.py --cluster education    # 18,078 GRs -> /mnt/win, resumable
# Bulk embedding needs the WHOLE GPU. .env pins EMBED_DEVICE=cpu for serving and
# load_dotenv does NOT override a real env var, so EMBED_DEVICE=cuda is what wins.
pkill -f "[u]vicorn app.api"; pkill -x ollama          # free the GPU first
EMBED_DEVICE=cuda python scripts/ingest_corpus.py --checkpoint 1000 --batch-size 32
# Interrupted? Just run it again — it is resumable and idempotent.

# ---------- serve the scaled corpus ----------
# in backend/.env (or the environment):
#   VECTOR_BACKEND=hnsw
#   MAHAGR_INDEX_DIR=/mnt/win/mahagr/index
curl -s localhost:8000/corpus/stats     # documents, chunks, per-department counts

# ---------- start the local LLM server ----------
# Models MUST live on /mnt/win (root has ~2.9 G free). OLLAMA_MODELS and
# OLLAMA_CONTEXT_LENGTH must be set on the SERVER, not on `ollama pull`.
sudo systemctl disable --now ollama            # keep the packaged system service OFF
OLLAMA_MODELS=/mnt/win/mahagr/ollama \
OLLAMA_KEEP_ALIVE=30m \
OLLAMA_CONTEXT_LENGTH=8192 ollama serve        # leave running
ollama list                                    # qwen2.5:3b-instruct-q8_0 (3.3 GB)
ollama ps                                      # after a query: expect 100% GPU, CONTEXT 8192

# ---------- run the backend ----------
uvicorn app.api:app --port 8000
curl -s localhost:8000/health                  # shows provider + model + vector count
# frontend/:  npm install && npm run dev       → http://localhost:3000

# ---------- COMPLETE the corpus: all 33 departments (PLAN Phase 6) ----------
python scripts/fetch_mahgrs.py --list                  # 33 departments, 99,421 GRs
python scripts/fetch_mahgrs.py --all --dry-run         # what it would download
python scripts/fetch_mahgrs.py --all                   # ~81k new files, resumable
pkill -f "[u]vicorn app.api"; pkill -x ollama          # the ingest needs the WHOLE GPU
EMBED_DEVICE=cuda python scripts/ingest_corpus.py --checkpoint 1000 --batch-size 16
python scripts/build_graph.py                          # rebuild gr_edges afterwards

# ---------- Phase 4 auth: seed the four SRS roles ----------
python scripts/seed_users.py     # admin1 / reviewer1 / translator1 / officer1
                                 # all password123. DEV_NO_AUTH=1 in .env bypasses auth.

# ---------- knowledge graph (PLAN Phase 3) — no GPU, minutes ----------
# Only needed after an ingest, or after changing gr_metadata's parsing.
python scripts/reparse_refs.py --dry-run       # what would change, writes nothing
python scripts/reparse_refs.py                 # re-parse number/refs from stored text
python scripts/build_graph.py                  # rebuild gr_edges + print resolution
python scripts/build_graph.py --show 201805301529314021   # one GR's neighbourhood

# ---------- evaluate ----------
python scripts/eval_retrieval.py [--no-rerank] # retrieval only, no LLM needed
python scripts/eval_answers.py                 # ANSWER quality — needs the API running
python scripts/verify_demo.py

# ---------- prove it runs offline (stop the API first — it holds the GPU) ----------
pkill -f "[u]vicorn app.api"
python scripts/verify_offline.py
```

**Gotchas when running any of this**
- `pkill -f "uvicorn app.api"` will **kill its own shell** if the pattern appears
  in that shell's command line. Use a bracket pattern: `pkill -f "[u]vicorn app.api"`.
- Only one process can hold the GPU comfortably. `verify_offline.py` and the
  diagnostic scripts OOM if `uvicorn` is still running — stop it first.
- VRAM budget: LLM ~3.7 GB + reranker ~1.2 GB ≈ 4.9 GB of 6.1 GB.
- First query after an idle period is slow (model reload); `OLLAMA_KEEP_ALIVE=30m`
  keeps it resident during a demo.
- `curl localhost:8000/health` is the fastest confirmation that `.env` took effect.

---

## 7. Precise next step, owner to-dos, standing agreement

### Next step
**PLAN Phase 6 — ingest all 33 departments (99,421 GRs), then rebuild the
graph.** See `PLAN.md` Phase 6 for the numbers, the ordering and the risks.
(The old "next step: Phase 3" text that used to sit here is obsolete — Phases 3,
4 and 5 have all landed since. §0 is the authoritative status.)

### ⚠ Phase 2 has TWO unfinished verification steps — do these FIRST
Both need the finished index and the API; neither needs a re-ingest. Until they
are done, **do not quote retrieval quality numbers for the big corpus** — the
only measured numbers in this repo (hit@1 19/20, MRR 0.975) are from the
**196-GR flat index**, not from 18,078 GRs.

1. **Calibrate `efSearch`** — `python scripts/eval_ann.py` (see §5.25). The
   current default of 128 measured only 0.735 recall@10 on real embeddings.
   Expect to raise it; ANN latency is sub-millisecond so recall is cheap.
2. **Recalibrate the retrieval thresholds on the bigger corpus** —
   `python scripts/eval_retrieval.py`. `text 0.55` / `rerank 0.85` were
   calibrated against 196 GRs. With 18,078, there are far more
   plausible-but-wrong chunks, so the score distributions shift and the
   abstention boundary almost certainly moves. Then re-run
   `python scripts/eval_answers.py` (needs the API up) for end-to-end
   CITED/GROUNDED/CORRECT/ABSTAINS + latency vs the SRS's 10 s target.

Note the gold set (`data/gold/gold.json`, 23 questions) is still valid and is
now a **much harder** test — the same questions must find the right GR among
18,078 instead of 196. Two questions target the synthetic fee-table fixtures,
which exist **only in the flat index**, so expect those two to fail on `hnsw`.

**Phase-2 ops facts worth carrying forward:**
- Bulk embedding is **compute-bound at ~100% GPU**: `--batch-size 32` OOMs on a
  6 GB card with ~1,800-char Marathi chunks (torch retries silently and
  throughput collapses to 8 chunks/s); **`--batch-size 8` is the setting that
  works**, ~12–13 chunks/s, 3.4 GB VRAM.
- The ingest is genuinely resumable — this was proven the hard way when the
  OOM run was killed mid-flight and the restart reported
  `recovered from an interrupted run: dropped 3096 orphan chunk rows`.

### Owner to-dos (not the assistant's)
1. **Install the Ollama user service** (one time, no sudo needed):
   ```bash
   mkdir -p ~/.config/systemd/user
   cp deploy/ollama.service ~/.config/systemd/user/
   systemctl --user daemon-reload && systemctl --user enable --now ollama
   ```
2. ~~**Add `/mnt/win` to `/etc/fstab`**~~ ✅ **DONE** (verified 2026-08-09):
   `UUID=145213A852138E1C /mnt/win ntfs3 rw,uid=1000,gid=1000,windows_names,iocharset=utf8,nofail 0 0`.
   Note `nofail` — if Windows leaves NTFS dirty (Fast Startup, §5.8) the boot
   still succeeds and the partition is simply **absent**, which looks like a
   missing corpus rather than a mount error. Check `findmnt /mnt/win` first when
   something is mysteriously empty.
3. ~~`sudo pacman -S tesseract-data-mar`~~ ✅ **INSTALLED** (`tesseract-data-mar
   2:4.1.0-5`, verified 2026-08-09). The 16-GR OCR ingest is now **unblocked**
   and still to be run: `python scripts/add_pdfs.py "../Original Maha-GR" index --ocr`.
4. `git push --force origin main` to scrub `Co-Authored-By: Claude` from GitHub.
5. **Decide how far to take PLAN Phase 6** — the full 33 departments is
   ~81,000 more files to download and ~3 h of GPU to embed. The machine must
   stay awake and `/mnt/win` must stay mounted for the duration (see to-do 2).
6. **Commit the Phase-1 through Phase-5 work when ready** — it is ALL
   uncommitted (last commit `f814fc3`, which predates every phase in `PLAN.md`).
   The file list below was accurate on 2026-08-06 and is now INCOMPLETE; run
   `git status` for the truth. Phase 4/5 additionally added `backend/app/auth.py`,
   `backend/scripts/seed_users.py`, `backend/engine/text_table_detect.py`,
   `backend/tests/test_text_table_{detect,ingest}.py`, `frontend/app/login/`,
   `frontend/app/(portal)/`, `frontend/components/{ui/,landing/,UserMenu.tsx,
   nav-link.tsx}`, `frontend/components.json` and `frontend/lib/utils.ts`, and
   DELETED `frontend/app/browse/page.tsx` (moved into the route group).
   **`backend/.next/` is build junk in the wrong directory — delete it, don't
   commit it.** The 2026-08-06 snapshot:

   *Modified:* `DEMO.md` · `HANDOFF.md` · `README.md` ·
   `backend/app/api.py` ·
   `backend/engine/{config,ingest,officer,rag,reranker,retrieval,vector_store}.py` ·
   `backend/scripts/{fetch_mahgrs,officer_tools}.py` ·
   `backend/tests/{test_llm_provider,test_rag}.py` ·
   `frontend/app/page.tsx` · `frontend/app/browse/page.tsx` ·
   `frontend/components/ui.tsx` · `frontend/lib/api.ts`

   *New (untracked):* `CHECKLIST.md` · `PLAN.md` · `deploy/` ·
   `backend/engine/{corpus_db,graph}.py` ·
   `backend/scripts/{ingest_corpus,eval_ann,eval_answers,verify_offline,build_graph,reparse_refs}.py` ·
   `backend/tests/{test_corpus_db,test_hnsw_store,test_corpus_retrieval,test_graph}.py` ·
   `frontend/components/graph.tsx`

   Phase 3 additionally MODIFIED `backend/engine/{gr_metadata,graph,officer}.py`,
   `backend/tests/test_gr_metadata.py`, `frontend/lib/api.ts` and
   `frontend/components/graph.tsx`.

   Note `backend/.env` is gitignored but was changed too — it now sets
   `VECTOR_BACKEND=hnsw` and `MAHAGR_INDEX_DIR=/mnt/win/mahagr/index`.

### If you are a NEW session with zero memory — read this first
1. **Read §0 first** — it is the resynced status; the rest of this file is older.
2. The thing to demo is the **`hnsw` corpus**: 18,080 GRs / 74,004 chunks,
   6 departments, on `/mnt/win`. `.env` already points at it.
3. **Kill stray `uvicorn`/`ollama` before starting anything** — one 6 GB GPU,
   and they will OOM each other (`pkill -f "[u]vicorn app.api"`, `pkill -x ollama`;
   note the bracket, or `pkill` kills its own shell).
4. `python -m pytest tests/ -q` → **139 pass, 1 fails** (the known
   `SYSTEM_PROMPT_COMPACT` regression, §0). Needs no GPU/network. Run it first to
   confirm the tree is sane.
5. `curl -s localhost:8000/health` is the fastest confirmation of which index and
   which LLM are actually loaded — and whether Ollama put the model fully on the
   GPU (`llm_placement`).
6. Retrieval quality on the big corpus is measured but NOT re-tuned: `efSearch`
   is calibrated (512), the **thresholds are not** — `rerank_threshold` is still
   the 196-GR value of 0.85. Do not quote retrieval numbers without saying which
   index they came from.

### Standing agreement (how the assistant works on this repo)
- Implement **one PLAN phase at a time, in order.** Do not start the next phase
  until told to.
- **Before each phase:** state plainly what the owner must do manually
  (installs, keys, decisions, disk). If something is needed, stop and wait.
- **After implementing:** verify it actually works and show the proof (tests,
  curl output, a script run). Never claim done without evidence, and say clearly
  what could not be verified.
- **Then explain the phase in simple plain language** — the owner presents this
  work and needs to understand it, not re-derive it.
- **Then give a short list of simple but critical viva/interview questions** on
  that phase, each with a one-line hint.
- Keep `HANDOFF.md`, `PLAN.md` and `CHECKLIST.md` updated as phases complete.
- **Everything runs locally.** No cloud services, no data leaving the machine.
- **Git:** commit only when asked; author as
  `Soham Margaj <sohammargaj55555@gmail.com>`; **no `Co-Authored-By` / Claude
  attribution**; never push (the owner handles GitHub).
