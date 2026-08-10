# MahaGR Assist — phase checklist

Running log of what is actually DONE vs claimed, per `PLAN.md`. Updated as each
phase closes. `ROADMAP.md` tracks the earlier initial-build phases separately.

Legend: ✅ done & verified · 🟡 partly done · ⏳ blocked on the owner · ❌ not started

> **RESYNC 2026-08-09.** Phases 1–3 below were written on 2026-08-06. A day of
> work landed on 2026-08-07 that no document recorded: **table extraction +
> a full re-ingest** (new "Phase 2.5"), **Phase 4 auth**, and the **Phase 5
> portal rebuild**. Their sections are at the bottom of this file.
> Consequences for the numbers in Phases 1–3: chunks are now **74,004**
> (not 64,744), documents **18,080** (not 18,078), resolved graph edges
> **6,267** (not 5,753), and `RetrievalConfig.rerank_pool` is **40** (not 15).
> Answer-quality numbers below have **not** been re-measured since the
> re-ingest — treat them as pre-table-extraction.

---

## PLAN Phase 1 — Local LLM (Ollama)  🟡
Verified 2026-08-06.

| # | Item | State | Proof |
|---|------|-------|-------|
| 1 | `qwen2.5:3b` pulled to `/mnt/win/mahagr/ollama` (root untouched) | ✅ | `ollama list` → 1.9 GB; `df /` unchanged |
| 2 | `backend/.env` set (`LLM_PROVIDER`/`OLLAMA_MODEL`/`EMBED_DEVICE`) | ✅ | `/health` → `ollama` + `qwen2.5:3b` |
| 3 | Backend answers with the local model | ✅ | `/ask` returns cited EN + Marathi answers |
| 4 | Runs **offline** — nothing leaves the machine | ✅ | `scripts/verify_offline.py` (no Groq key, non-loopback sockets blocked) |
| 5 | Ollama as a persistent server | ⏳ | unit written to `deploy/ollama.service`; owner installs (see HANDOFF §7) |
| 6 | Marathi answer quality good enough to demo | ✅ | bare-`[1]` replies **8/20 → 0/20**; see "the prompt was the bug" below |
| 7 | Model upgraded to 8-bit (`qwen2.5:3b-instruct-q8_0`) | ✅ | 3.3 GB on `/mnt/win`; `ollama ps` → 100% GPU, ctx 8192 |

**Bugs found and fixed while verifying** (all silent, none previously known):
- `.env` was read *after* `config.py` had already snapshotted `os.environ`, so
  `EMBED_DEVICE=cpu` was ignored → `.env` now loads in `config.py`.
- Cross-encoder pinned to CPU by `EMBED_DEVICE` → **27.6 s/query**. Split out
  `RERANK_DEVICE` (auto→GPU) + fp16 → **0.33 s**; end-to-end p50 **38 s → 2.6 s**.
- `/ask` dropped its `language` argument → the portal's EN/मराठी toggle did
  nothing (FR 3.4.2 / 3.4.5). Threaded through; ISO codes now map to language
  names, which fixed the same latent flaw in summarize/explain/compare.

### "The model is too weak" was wrong — the PROMPT was the bug
The bare-`[1]` answers looked like 3B incapacity. Two upstream causes, both found
by instrumenting rather than guessing:

1. **The model was being starved of context.** `estimate_tokens` priced Devanagari
   at Llama's 2.0 tokens/char, but qwen2.5 has real Devanagari vocabulary and
   measures **1.09** — a 1.7× over-count. One ~1900-char Marathi chunk therefore
   "cost" 3264 of a 2200 budget, so `build_prompt` sent **1 block and dropped 10**.
   The rate is now per-tokenizer and the budget per-provider (groq is rate-limit
   bound, ollama is context-window bound). Blocks sent: **1 → 3**; the estimate is
   now within 2% of Ollama's reported `prompt_tokens`.
2. **The prompt itself suppressed the answer.** Isolating model from prompt on
   identical context: full prompt → **4 completion tokens** (`"[1]"`); a two-line
   prompt → **99 tokens**, a correct cited answer. ~350 words of mostly-prohibitive
   rules collapse a 3B model into emitting only the token it is sure of. Added
   `SYSTEM_PROMPT_COMPACT` + a one-line reminder placed AFTER the question,
   selected automatically for ollama; groq keeps the long, verified prompt.

Also raised Ollama's context window to 8192 (`OLLAMA_CONTEXT_LENGTH`) — its 4096
default **silently truncates**, dropping the system prompt with no error — and
added `rag.warn_if_over_context` so that can never happen unnoticed again.

**Added:** `scripts/eval_answers.py` (answer-quality harness),
`scripts/verify_offline.py` (on-prem proof), `deploy/ollama.service`.
**Tests:** 37 → 46, all passing.

**Measured** (23-Q gold set, `scripts/eval_answers.py`, qwen2.5:3b-instruct-q8_0):

| | before | after |
|---|---|---|
| CITED | 19/20 | 19–20/20 |
| GROUNDED | 20/20 | 20/20 |
| CORRECT | 18/20 | **19–20/20** |
| DEGENERATE (bare `[1]`) | **8/20** | **0/20** |
| ABSTAINS | 2/3 | 2/3 |
| latency p50 | 38 s → 2.6 s | 4.7 s |

Ranges are run-to-run variance at `temperature=0.2` over only 20 questions —
treat single-point differences of 1 as noise. Latency rose from 2.6 s because the
prompt now carries 3 blocks instead of 1; `context_token_budget=6000` beat 3500
(CORRECT 19–20/20 vs 17/20) so the seconds were worth it. **2/23 exceed 10 s.**

**Known remaining gap:** one out-of-corpus question (farm loan waiver) is still
answered instead of refused, even though retrieval correctly flags it
(`low_confidence=True`, scores ~0.001). The abstention instruction is now
decisive, but a 3B model still occasionally stretches. Groq abstains correctly.

---

## PLAN Phase 2 — Multi-department corpus, FAISS-HNSW + SQLite  ✅
Implemented 2026-08-06.

| # | Item | State | Proof |
|---|------|-------|-------|
| 1 | Multi-department corpus fetched | ✅ | **18,078 GRs / 244 MB across 6 departments, 0 failures** (`/tmp/fetch.log`) |
| 2 | SQLite schema `gr_documents` + `gr_chunks` | ✅ | `engine/corpus_db.py`; 14 tests |
| 3 | `HnswStore` (FAISS `IndexHNSWFlat`) beside `FaissStore` | ✅ | `engine/vector_store.py`; 13 tests |
| 4 | `scripts/ingest_corpus.py` — resumable, idempotent, GPU batch-embed | ✅ | see the run below |
| 5 | Two-stage retrieval (ANN → group by GR → rerank) | ✅ | `retrieval.CorpusRetriever` |
| 6 | Department / date / language filters | ✅ | native FAISS `IDSelector`, not a post-filter |
| 7 | Frontend filters + corpus-size stat | ✅ | `ScopeFilter`, `CorpusStat`; `tsc --noEmit` clean |

**Tests: 46 → 82**, all passing, still model-free (<1 s, no torch, no GPU, no network).

### What the numbers were before this phase
713 vectors · `IndexFlatIP` (exact, O(n)) · every chunk's text in a RAM JSON
sidecar · BM25 via `rank_bm25` (whole corpus tokenized in RAM).

### Design corrections found while implementing (both were in PLAN.md)
1. **FAISS-HNSW *can* pre-filter.** PLAN said it couldn't and proposed
   over-fetch-then-filter; faiss 1.14 has `SearchParametersHNSW` + `IDSelector`,
   verified here. Filters are now pushed INTO the search — a post-filter loses
   recall silently whenever the top-k is entirely from an excluded department.
2. **BM25 had to move to SQLite FTS5.** Not anticipated in the plan.
   `rank_bm25` keeps the tokenized corpus in RAM (several GB at 65k chunks).

### Bugs caught by the new tests, not by inspection
- **An external-content FTS5 table cannot be `DELETE`d from.** Removing a row
  needs the ORIGINAL text handed back via the `'delete'` command, before the
  content row goes. A plain DELETE left terms pointing at dead rowids, so BM25
  returned ghost ids. (`test_delete_chunks_from_is_crash_recovery`)
- **`EMBED_DEVICE=cpu` in `.env` silently applied to bulk ingestion** — the
  first smoke run took 4.3 min for 40 documents. The script now prints its
  device and warns on CPU; the real run is `EMBED_DEVICE=cuda …`.

### MEASURED ON THE FULL 18,078-GR CORPUS (2026-08-06) — READ THIS
Scaling the corpus 90x **degraded answer quality and doubled latency**. Reported
as measured, not as hoped.

| | 196-GR flat index | 18,078-GR HNSW corpus |
|---|---|---|
| hit@1 | 19/20 | **12/20** |
| hit@5 | 20/20 | **14/20** |
| MRR | 0.975 | **0.642** |
| CITED | 19-20/20 | 15/20 |
| **GROUNDED** | 20/20 | **20/20** (zero phantom citations) |
| CORRECT | 19-20/20 | **11/20** |
| DEGENERATE | 0/20 | **0/20** |
| ABSTAINS | 2/3 | 2/3 |
| latency p50 | 4.7 s | **11.4 s** |
| over the SRS 10 s | 2/23 | **12/23** |

**What survived scaling — and matters most:** GROUNDED stayed 20/20. Not one
answer cited a document that wasn't in its context. The abstention machinery
also held (2/3, unchanged). The system got *less accurate*, never *less honest*.

**What regressed:** ranking. 3 of the CORRECT failures are the synthetic
fee-table fixtures, which exist only in the flat index and are structurally
unanswerable here, so the fair figure is ~11/17. The rest is real: at 18k
documents there are far more plausible-but-wrong chunks than at 196.

**The most likely single cause, and the first thing to try next:**
`rerank_threshold` is still **0.85**, calibrated against 196 GRs.
`eval_retrieval.py` on the new corpus suggests **~0.993** (relevant p25 0.990 vs
reject p75 0.997 — the distributions now OVERLAP, where before there was a clean
gap). A too-low threshold admits weak chunks, which plausibly hurts BOTH metrics
at once: wrong GR cited, and a longer prompt to generate from. **Not yet tried.**

**Latency caveat — not yet isolated.** Abstentions returned in 2.6-3.2 s while
full answers took 12-16 s, so generation dominates. Retrieval could not be timed
separately in-process because uvicorn + Ollama already hold the 6 GB GPU and a
second reranker will not fit (the "one GPU, two jobs" rule, HANDOFF §5.20).

**Verified working end-to-end:** cross-department retrieval (a Marathi
scholarship question returned **Tribal Development** GRs), the department filter,
out-of-corpus abstention, zero phantom citations, and `doc` ids on every citation.

### Three ingestion runs, and what each one taught
1. **`--batch-size 32`, fp32** — CUDA OOM. torch's allocator retried silently
   and throughput collapsed to **8 chunks/s**; the only visible symptom was a
   `CUDACachingAllocator … OOM` line buried in the log. Killed.
2. **`--batch-size 8`, fp32** — no OOM, 3.4 GB VRAM, but **~10 chunks/s at 100%
   GPU**: compute-bound, so a bigger batch could not have helped. ETA ~2 hours.
   Killed. Its restart is what *proved* the crash-recovery path, reporting
   `recovered from an interrupted run: dropped 3096 orphan chunk rows`.
3. **`--batch-size 16`, fp16** — **36.6 chunks/s, 1.5 GB**, ~25 min. Shipped.

The lesson worth presenting: the instinct was "raise the batch size"; the
measurement said the card was already saturated and the real lever was
**precision**, not batch size.

## PLAN Phase 3 — Supersede knowledge graph + visualization  ✅ BUILT AND RUN
Written and unit-tested 2026-08-06 **while the Phase-2 ingest was still
running** — none of it needs the corpus or the GPU, because Phase 2 already
persisted each document's parsed `references` and `supersedes` flag.

| # | Item | State | Proof |
|---|------|-------|-------|
| 1 | `gr_edges` table + schema migration | ✅ | `corpus_db._MIGRATIONS`, `_migrate()` |
| 2 | GR-number canonicalisation | ✅ | `gr_metadata.canonical_number` + 3 tests |
| 3 | Edge builder (resolved/dangling/ambiguous) | ✅ | `graph.build_edges` + 7 tests |
| 4 | Traversal: neighbourhood + transitive supersede chain | ✅ | `graph.py` + 9 tests, incl. **cycle guards** |
| 5 | `scripts/build_graph.py` | ✅ | run on the 18,078-GR corpus — **60,420 edges**, see the resolution section below |
| 6 | `GET /graph/{doc_id}` + `/graph/stats/summary` | ✅ | routes registered, api imports clean |
| 7 | Rewire `officer.supersession` to read `gr_edges` | ✅ | uses the graph when built, **falls back to the scan when not** |
| 8 | `/ask` conflict warnings use the transitive chain | ✅ | "replaced by B, which was itself replaced — latest is C" |
| 9 | Frontend graph visualisation | ✅ | `components/graph.tsx`; `next build` clean |
| 10 | **SRS FR 3.7.4** — view/download the referenced GR | ✅ | citations now carry `doc`; open + download on every source card and the Browse detail |

**Tests: 82 → 122.**

### Running it exposed the real bug: 2% of references resolved
The graph built fine and was almost empty of edges. The cause was upstream, in
extraction, not in the graph: `gr_metadata._references()` matched a
slash-bearing token bounded by **whitespace**, but a real GR number **contains
spaces** — `एनजीसी-२०१०/(१९३/१०) /मशि-४`. Every reference came out truncated
(`२०११/प्रक्र`, `१३६/विशि-३`, `अनौस-२०२०/प्र.क्र.१०२/`) and matched nothing.
The measurable symptom: a document's own canonical number is a median of **28**
characters, an extracted reference only **19**.

Fixed by parsing a reference line the way it is actually written — cut the
block into numbered items, each item into comma-separated segments, trim each
segment at the date that follows the number:

    वाचा : १) शासन निर्णय, उच्च व तंत्र शिक्षण विभाग, क्र. एनजीसी-२०१०/(१९३/१०) /मशि-४, दि. ३०.१०.२०१०.
           └item┘└ doc type ┘└──── department ────┘└label┘└──── the number ────┘  └── the date ──┘

Two traps found while doing it, both of which quietly corrupt the output:
- **`क्र` also occurs INSIDE a GR number** (`प्र.क्र.४५`), so stripping at the
  first label match truncates the number to `४५`. The discriminator is the
  slash: a number's internal `क्र` always sits after one, an introducing label
  never does. (An earlier attempt capped the label to "within 3 words of the
  start" instead — measurably worse: 7.9% vs 9.5%.)
- **`_parse_date` had no dotted-date form.** GR reference lines write
  `दि. ३०.१०.२०१०` far more often than `30/10/2010`, so 2 of every 3 cited
  dates were being dropped. Adding it took date coverage 34% → 72%.

**The fix needed no re-ingest.** `gr_documents.text` holds the full text of all
18,078 documents, so `scripts/reparse_refs.py` re-PARSES in ~3 minutes on CPU
instead of ~25 minutes of GPU re-embedding. It updates only what is derived from
text (number, canonical number, refs, supersedes) and deliberately leaves `date`
and `department` alone — those have better sources (the order id and the corpus
folder) than an OCR'd header line.

### Measured, before and after (18,078-GR corpus)

| | before | after |
|---|---|---|
| edges | 67,681 | 60,420 |
| **resolved** | **1,348 (2.0%)** | **5,753 (9.5%)** |
| ambiguous | 1,908 | 6,623 |
| dangling | 64,425 | 48,044 |
| documents with ≥1 resolved edge | 1,123 | **3,904** |
| median canonical length of a reference | 19 | 28 (= own numbers) |
| references carrying a parsable date | 0 | 43,738 (72%) |

Fewer *edges* and more *resolved* ones is the point: the old count was inflated
by fragments and by dates mis-read as GR numbers.

**1,771 of the resolved edges were disambiguated by the cited date** — 2,138
canonical numbers in this corpus are shared by more than one document, and the
date is the only thing that separates them. Where the date singles out no
candidate, the edge stays `ambiguous`; it is never guessed.

### Why 9.5% is the honest ceiling, not a failure
> **Re-measured 2026-08-09 (`scripts/cited_departments.py`).** The "Finance /
> GAD / Revenue" attribution below was half right and is corrected in `PLAN.md`
> Phase 6: counting the department named in every GR's `वाचा` block gives
> **Finance 32.4%** and **General Administration 9.6%** — but **Revenue & Forest
> 0.5%** and **Law & Judiciary 0.3%**. Acting on the guess would have cost
> ~10,000 GRs of download and GPU for almost nothing.

- We hold **6 of Maharashtra's ~33 departments**. Measured: only ~21% of
  unresolved references even name a department we ingested. The rest cite
  Finance / GAD / Revenue orders, or name no department at all.
- Of the edges that stay `ambiguous`, the nearest candidate by date is more
  than a month away in the large majority of cases — i.e. they are genuinely
  different orders in the same number series, not the same order.
- Prefix-matching truncated numbers was tried earlier and **rejected**: on a
  20k sample it recovered 362 matches and produced 864 ambiguities.

**Verified end to end on the running API:** `/graph/stats/summary` reports the
breakdown; `/graph/{id}` returns a 17-node neighbourhood with a supersede chain
and three dated ghost references; `/supersede/{id}` reads the graph; and `/ask`
on a Marathi question about non-aided-school evaluation criteria returned a
cited answer **plus** the conflict warning *"GR प्रामाअ-२०१८/प्र.क्र.७३/एसएम-४
may be superseded by GR उमाशा-२०२१/प्र.क्र १८/एसएम-४ (2021-02-24) — verify
before relying on it."*

### Frontend: hand-drawn SVG, not react-flow/cytoscape
(One layout change came out of running the graph for real: with 3.5× more
resolved edges a well-cited GR now draws 16+ neighbours, and the labels
collided. Each arc is capped at 7 nodes, newest first, with "+N more linked GRs
not drawn" stated in the legend, and labels are dropped — hover keeps them —
once the diagram is crowded. Browse route 5.26 → 5.55 kB.)

Those libraries are 100–400 kB of JS and bring their own visual language that
would then need restyling to match the portal. What is actually needed is small
and fixed — one focal GR, its direct neighbours, and a linear chain — so the
layout is two lines of trigonometry. **Measured: the whole panel added 1.65 kB
to the Browse route** (3.61 → 5.26 kB). The layout is deliberately deterministic
(no force simulation) so a presenter can point at a node and it stays put. If
this ever needs pan/zoom/drag over hundreds of nodes, that is when to reach for
a real graph library.

The panel leads with the finding **in words** ("the order likely in force is X")
before the diagram — a presenter shouldn't have to interpret a picture to
deliver the point — and renders **dangling references as dashed ghost chips**
rather than hiding them.

### The design decision worth defending in a viva
The hard part is **not** the graph — traversing a few thousand edges is trivial.
It is **reference resolution**: deciding whether `संकीर्ण-२०२३/प्र.क्र.४५/तांशि-४`
printed in one OCR'd document is the same order as a `gr_number` parsed from
another. They differ by spacing, by Devanagari vs ASCII digits, and by
punctuation the scanner dropped. So matching is on a canonical form, and every
edge is stored **three-valued**:

- `resolved` — matched exactly one document
- `dangling` — a real reference to an order the corpus does not hold
- `ambiguous` — matched several (a canonical-form collision)

Dangling edges are **kept, not dropped**. "This GR builds on an order we don't
have" is information an officer needs, and the resolution rate is the honest
measure of corpus completeness. Note `/` is preserved during canonicalisation
because it is structural in a GR number — over-normalising would *fabricate*
supersessions, which is worse than missing one.

**Cycles are real** (two GRs amending each other, or an OCR collision) and every
traversal carries a `seen` set plus a depth cap — without them `/graph` is an
infinite loop behind an HTTP request. Two tests pin this.

### Bug the tests caught
A GR whose reference list contains its own number (an OCR variant that
`gr_metadata._references` misses because it only excludes by raw substring) was
being recorded as a **dangling** edge — inflating the "we don't hold this order"
count with self-references. Now skipped as neither an edge nor a dangler.

## PLAN Phase 2.5 — Table extraction from OCR'd text  ✅
Unplanned; built 2026-08-07, **verified by re-ingest 2026-08-09**.

| # | Item | State | Proof |
|---|------|-------|-------|
| 1 | `engine/text_table_detect.py` | ✅ | pipe + dash table detection, `row_to_sentence` output |
| 2 | Wired into `ingest.chunk_text` | ✅ | `ingest.py:425` |
| 3 | Corpus re-ingested with tables | ✅ | chunks **64,744 → 74,004**; **17,736** rows now `content_type='table'` |
| 4 | Tests | ✅ | `test_text_table_detect.py`, `test_text_table_ingest.py` |

**The problem it solves.** The corpus is orgpedia's pre-OCR'd `.mr.txt`, so
there is no PDF structure left and **pdfplumber cannot help**. A fee schedule
arrives as a raw word stream, which `bge-m3` — trained on prose — embeds badly,
so a question about fee amounts matched nothing. Each detected row is rewritten
as the same sentence form the born-digital PDF path already produced:
`"In this row: अ.क्र. is 1, प्रवर्ग is खुला, शुल्क is 5,000"`.

**Measured across all 18,078 GRs before building it:** pipe-delimited tables in
**7,611 GRs (42% of the corpus)**, dash-separated in 1,805. That measurement is
what justified the work — 42% is not an edge case.

**Deliberately conservative**, and this is the design point worth defending: a
false positive (prose parsed as a table and mangled into broken row sentences)
is strictly worse than a false negative (a table left as prose, which is no
worse than the previous behaviour). Hence ≥3 pipes per line, ≥2 rows, and a
consistent pipe count ±1 for OCR noise. Numbered lists (`१.`, `२.`) are
explicitly **not** treated as tables — they are usually just numbered prose and
the false-positive rate was unacceptable.

**Side effects of the re-ingest**, both undocumented until 2026-08-09:
- The 2 synthetic fee-table fixtures were added to the hnsw corpus
  (18,078 → **18,080** documents). Older notes calling those gold questions
  "structurally unanswerable on hnsw" no longer apply.
- The graph was rebuilt on the new corpus: resolved edges **5,753 → 6,267**.
- The pre-table index survives at `/mnt/win/mahagr/index_v1/` (887 MB). Delete
  it once the current index is trusted.

**Not yet measured:** whether table chunks actually improved answer quality on
fee/quota questions. `eval_answers.py` has not been re-run since the re-ingest,
so there is no before/after number for this phase. Say "42% of GRs contain
tables and they are now indexed as prose", not "it improved accuracy by X".

## PLAN Phase 4 — Admin dashboard (auth, roles, audit)  🟡
Implemented 2026-08-07. **Not tested, not hardened.**

| # | Item | State | Proof / caveat |
|---|------|-------|----------------|
| 1 | `users` + `audit_log` in `app/db.py` | ✅ | schema present; kept out of `engine/corpus_db.py` so a corpus rebuild can never touch the audit trail |
| 2 | bcrypt password hashing | ✅ | `auth.get_password_hash` / `verify_password` — real bcrypt, not a bare SHA |
| 3 | JWT login | ✅ | `POST /auth/login`, HS256, 1-day expiry; token in `sessionStorage`, not `localStorage` |
| 4 | `current_user` dependency on every route | ✅ | `app/api.py` |
| 5 | `require_role` + role scoping | 🟡 | only `/admin/audit-logs` is role-gated, plus a language restriction on `/ask` for non-translators. **Reviewer role is not wired** — `db.list_conversations(all_users=True)` exists but no route calls it |
| 6 | Audit every `/ask` | ✅ | logs user, ip, question, cited GR numbers — **deliberately not the answer text or document bodies** |
| 7 | Per-user rate limiting | 🟡 | token bucket, 20/min, returns 429 with a clear message. **In-process** — resets on restart, does not survive multiple workers |
| 8 | `/admin` page | ✅ | `app/(portal)/admin/page.tsx`; server-gated, the client-side role check is convenience only |
| 9 | `DEV_NO_AUTH` escape hatch | ✅ | exists — **and is currently `=1` in `.env`, so auth is entirely bypassed at runtime** |
| 10 | Seed script for the four SRS roles | ✅ | `scripts/seed_users.py` — `admin1`/`reviewer1`/`translator1`/`officer1`, all `password123` |
| 11 | **Tests** | ❌ | **none.** The only module in the repo without a test file |

**Two things to fix before this can be called done:**
1. `JWT_SECRET` defaults to the literal `"maha-secret"` in `app/auth.py` **and**
   is set to exactly that in `.env`. A signing key with a shipped default is the
   kind of thing a reviewer finds in thirty seconds.
2. There are no tests, so none of the above is *proven* — it is only *present*.

## PLAN Phase 5 — UX polish + presentation refresh  🟡
Rebuilt 2026-08-07.

| # | Item | State | Proof |
|---|------|-------|-------|
| 1 | Portal rebuilt on shadcn/ui + Radix + lucide | ✅ | 19 primitives under `components/ui/`, `components.json`, `lib/utils.ts` |
| 2 | Public landing page | ✅ | `app/page.tsx` (883 lines) — architecture diagrams, on-prem story, live corpus stat |
| 3 | Officer portal moved to a route group | ✅ | `app/(portal)/{ask,browse,admin}` + shared `layout.tsx` |
| 4 | Login page + user menu | ✅ | `app/login/page.tsx`, `components/UserMenu.tsx` |
| 5 | Typecheck | ✅ | `npx tsc --noEmit` clean (2026-08-09) |
| 6 | Mobile / responsive sidebar on Ask | ❌ | still open |
| 7 | Loading / empty / error states | 🟡 | uneven across pages |
| 8 | Deck + `DEMO.md` refreshed | ❌ | both still describe the OLD single-page portal and the old corpus numbers |

**Not verified:** `npx next build` has not been run since the restructure — only
`tsc --noEmit`. A route-group layout can typecheck and still fail to build.

## PLAN Phase 6 — Full corpus + complete graph — ✅ **COMPLETE (2026-08-10)**

> **WAVE B/C (2026-08-10): all 33 departments ingested. THE CORPUS IS COMPLETE.**
>
> | | wave A (41k) | **wave B/C (99k)** |
> |---|---|---|
> | departments | 12 | **33 (all)** |
> | documents | 41,474 | **99,410** |
> | vectors | 156,795 | **401,573** |
> | graph edges | 120,224 | **317,250** |
> | resolved | 17,935 | **46,412** |
> | docs with ≥1 resolved edge | 11,753 | **30,799** |
> | supersede edges | 3,321 | **8,983** |
> | hit@1 / hit@5 / MRR | 15/20 · 18/20 · 0.812 | 14/20 · **18/20** · 0.787 |
> | latency p50 | 13.3 s | 13.9 s |
> | tests | 152 | **157** |
>
> Fetch 1.6 h (57,948 files) · ingest **117 min** (244,778 new chunks) ·
> alignment re-verified at full scale (400 samples, worst self-similarity
> **0.9997**, 0 orphans).
>
> ### THE HEADLINE FINDING: wave B/C was the wrong investment, and it is provable
> Controlled comparison — the same 18,080 education documents, same text, same
> parser; only the set of documents they can MATCH AGAINST grew:
>
> | corpus | resolved edges (education docs only) | rate |
> |---|---|---|
> | 18k (6 depts) | 6,267 | 10.37% |
> | **41k (+6 depts, wave A)** | **9,153** | **15.15%** |
> | **99k (+21 depts, wave B/C)** | **9,173** | **15.17%** |
>
> **Wave A: +23,394 documents → +2,886 resolved edges. Wave B/C: +57,939
> documents → +20.** That is ~144 edges per 1,000 documents versus **0.35** — a
> ~400x difference in marginal value, and a direct vindication of
> `scripts/cited_departments.py`: it identified Finance and GAD as where
> essentially all the value was, in advance, and it was right.
>
> ### The resolution rate is now an absolute ceiling, not a moving target
> 14.63% corpus-wide. Because we now hold **the entire published dataset**,
> every remaining `dangling` edge points at a document orgpedia does not have at
> all — pre-digitisation orders, central government, Acts and Rules.
> **76% of references in Maharashtra's published GRs point outside the published
> corpus.** That is a finding about the DATASET, not about this pipeline, and it
> could not be stated before the corpus was complete.
>
> ### Scaling barely cost accuracy — the architecture's main claim
> 2.4x more documents; **hit@5 unchanged at 18/20**, hit@1 15→14 (the repo's
> standing definition of noise). Set against the earlier 196 → 18,078 jump,
> which crashed hit@1 from 19/20 to 12/20: **196 → 99,410 is a 500x increase and
> hit@5 went 20/20 → 18/20.** What changed in between is `rerank_pool=40`,
> table extraction, and a calibrated `efSearch`. Retrieval cost rose as expected
> — p50 1.9 s → **2.78 s** (20% of latency) — since BM25 and reranking now work
> over 401,573 chunks.
>
> ### efSearch: an ANN parameter is a FUNCTION OF INDEX SIZE, not a constant
> | corpus | setting | recall@60 |
> |---|---|---|
> | 74k vectors | ef=512 | 0.986 |
> | 157k vectors | ef=512 | **0.962** (same setting, quietly worse) |
> | 402k vectors | ef=1024 | **0.967** (raised once, decayed again) |
> | 402k vectors | **ef=4096** | **0.997** (recall@10 1.000) |
>
> Now **4096**. Exact brute force is **35.7 ms/query** at this scale vs **1.1 ms**
> for HNSW — at 157k vectors exact was only 7.7 ms, so this is the first corpus
> where ANN is a real speed win rather than only a scaling argument.
>
> ### Three more bugs this surfaced
> 1. **A filename with a SPACE broke one download, deterministically.**
>    `Revenue_and_Forest/'202510171720567619 .pdf.mr.txt'`. urllib rejects an
>    unencoded space *before any network call*, so the retry loop could never
>    help, and `failed 1` of 99,421 reads as a rounding error rather than a bug.
>    `fetch_mahgrs.py` now URL-quotes department and filename. (HANDOFF §5.33)
> 2. **300 invalid dates, and my first count of "9" was wrong.** A range check on
>    a date STRING catches a bad year and sails past a bad DAY: rows like
>    `2025-06-94` and `2026-02-96` have plausible years and impossible days.
>    `_parse_date` formatted without ever validating. Now two checks — calendar
>    validity AND a plausible year — because neither catches the other's failure
>    (`0201-01-21` is a *valid* date and a nonsense GR). **299 repaired from the
>    order id, 1 genuinely unknown (left NULL).** The corpus span the portal
>    advertises went from `0201-01-21 .. 9202-01-19` to **1962-02-28 ..
>    2029-09-01**. 7 tests.
> 3. **The filtered-search efSearch boost had silently become a no-op.**
>    `min(4 * ef, 1024)` was written when ef was 128; once ef reached 1024 the
>    expression collapsed to `max(1024,1024)` and filtered searches lost the
>    compensation §5.21 documents them as needing. Ceiling now scales
>    (`HNSW_EF_SEARCH_MAX`), with a test asserting the boost is strictly above
>    the base. (HANDOFF §5.36)
>
> ### Method note worth carrying
> `eval_answers.py` counted GROUNDED failures but never LISTED them — a
> groundedness regression appeared as a number with no way to see which question
> caused it. It now reports `PHANTOM` failures first, since citing a document
> that was never retrieved outranks citing the wrong one that was.
>
> ### GROUNDED is 95%, not 100% — measured properly, and the cause is known
> The single eval run reported 18/20 and I did NOT accept that on one sample.
> Repeated 3x over the 20 in-corpus questions (60 answers):
>
> | run | GROUNDED |
> |---|---|
> | 1 | 20/20 |
> | 2 | 18/20 |
> | 3 | 19/20 |
> | **total** | **3 phantoms in 60 answers (5.0%)** |
>
> Not noise: **one question phantoms reproducibly**, citing `[2,3,4,5,6]` in two
> separate runs. Reproduced deliberately — `sources=1, dropped=11,
> prompt_tokens=2448`:
>
> 1. **The prompt is STARVED.** Retrieval returns 12 chunks and
>    `trim_to_budget` keeps **one**: a single long Marathi chunk costs ~2,280 of
>    the 3,000-token budget, so nothing else fits. This is HANDOFF §5.5's
>    starvation failure returning at the budget set to fix LATENCY.
> 2. **The phantom is the DOCUMENT'S OWN NUMBERING.** The GR text reads
>    *"संदर्भाधिन अ.क्र.४, ५ व ६"* — "referenced items no. 4, 5 and 6" — and the
>    model transcribes those into `[4] [5] [6]`. It is not inventing sources; it
>    is echoing the document's internal enumeration in citation brackets, and
>    with one block sent those numbers are out of range.
>
> The groundedness monitor therefore did its job — it surfaced a real
> miscitation rather than hiding it. **Phantoms are still never silently
> stripped**; that detection IS the safety property.
>
> ### THE OPEN TENSION — do not pretend this is solved
> The 10 s NFR and answer quality are in DIRECT conflict on a 3B model with a
> 6 GB card, and `context_token_budget` is the dial between them:
>
> | budget | latency p50 | CORRECT | blocks reaching the model |
> |---|---|---|---|
> | 6000 | 34.1 s | 14/20 | ~3 |
> | **3000 (current)** | **13.9 s** | **13/20** | often **1** |
>
> Neither meets 10 s. Picking the final point needs a measurement designed for
> it — repeats, and ideally more than 23 gold questions — not another single-run
> tweak, which is how a gold set gets overfitted.

### Wave A (2026-08-09) — the 6 most-cited missing departments

### What shipped

| # | Item | State | Proof |
|---|------|-------|-------|
| 1 | Ingestion order chosen by MEASUREMENT | ✅ | `scripts/cited_departments.py` (new) |
| 2 | 6 most-cited missing departments fetched | ✅ | 23,394 files, **0 failures** |
| 3 | Ingested incrementally into the same index | ✅ | 18,080 → **41,474** GRs, 74,004 → **156,795** vectors |
| 4 | Index/DB alignment proven, not assumed | ✅ | `scripts/verify_corpus.py` (new) |
| 5 | `gr_edges` rebuilt | ✅ | 60,421 → **120,224** edges |
| 6 | `efSearch` re-calibrated | ✅ | 512 → **1024** |
| 7 | Retrieval + answer quality re-measured | ✅ | tables below |
| 8 | OOM guard for bulk embedding | ✅ | `config.EMBED_MAX_SEQ` + 3 tests |
| 9 | Deterministic abstention (FR 3.3.5) | ✅ | `rag._hard_abstention` + 6 tests |

**Tests: 143 → 152.**

### Corpus

| | before | after |
|---|---|---|
| departments | 6 | **12** (of 33) |
| GRs | 18,080 | **41,474** |
| chunks = vectors | 74,004 | **156,795** |

### The knowledge graph — the headline result

| | before | after |
|---|---|---|
| edges | 60,421 | **120,224** |
| **resolved** | 6,267 | **17,935** |
| ambiguous | 6,110 | 11,562 |
| dangling | 48,044 | 90,727 |
| docs with ≥1 resolved edge | 4,197 | **11,753** |
| `supersedes` edges | 1,459 | **3,321** |

**The controlled comparison** — edges from the *original* 6 departments only.
Their text did not change; only the set of documents they can match against did:

| | before | after |
|---|---|---|
| edges | 60,421 | 60,420 |
| **resolved** | **6,267 (10.37%)** | **9,153 (15.15%)** |

**+46% more resolved edges with zero change to the parser.** This cleanly
separates the two causes of movement in the resolution rate: the 2.0% → 9.5%
earlier gain was parse quality, this one is pure corpus coverage.

### Ingestion order was measured, and the measurement paid for itself

`scripts/cited_departments.py` counted the department named in every GR's `वाचा`
block across all 18,080 documents. Where the newly-resolved edges actually landed:

| destination | resolved edges | GRs ingested | edges per GR |
|---|---|---|---|
| **Finance** | **2,579** | 1,046 | **2.47** |
| General Administration | 433 | 6,345 | 0.068 |
| Industries, Energy & Labour | 190 | 2,885 | 0.066 |
| Planning | 149 | 2,233 | 0.067 |
| Public Health | 122 | 7,076 | 0.017 |
| Public Works | 7 | 3,809 | 0.002 |

**Finance is the #1 citation target in the entire corpus** — ahead of every
education department — bought for 1,046 GRs and 12.4 MB. ~36x more productive
per GR than the next best.

**Two honest notes on the method:**
- It corrected the docs. `HANDOFF` had asserted the gap was "Finance / GAD /
  Revenue". Finance and GAD were right; **Revenue & Forest measured 0.5% and
  Law & Judiciary 0.3%** — acting on that guess would have cost ~10,000 GRs of
  download and GPU for almost nothing.
- **The proxy over-predicts.** Public Works ranked 5th (682 documents named it)
  and delivered **7 edges from 3,809 GRs**. A department NAME in a reference
  block is evidence of a citation, not proof its number will resolve — exactly
  the caveat in the script's docstring, now confirmed against real data. Use it
  as a ranking, not a forecast.

### Retrieval

| | 18k corpus | **41k corpus** |
|---|---|---|
| hit@1 | 13/20 | **15/20** |
| hit@5 | 15/20 | **18/20** |
| MRR | 0.688 | **0.812** |

⚠ **Not attributable to wave A alone.** This is the first `eval_retrieval` run
since 2026-08-06, and FOUR things changed in between: table extraction + full
re-ingest, the fee-table fixtures entering the hnsw corpus, `rerank_pool` 15→40,
and wave A. The honest claim is "quality improved across these four changes".

⚠ **These numbers were measured at `efSearch=512`, before it was raised to
1024** — the eval ran earlier in the same chain. Since 1024 measures higher
recall@60 (0.962 → 0.980), the current configuration should be equal or
slightly better, but that has NOT been re-measured. Re-run `eval_retrieval.py`
before quoting hit@k as the number for the shipped config.

### `efSearch` — recalibrated, and the reason it had to be

| efSearch | recall@10 | recall@60 | ms/query |
|---|---|---|---|
| 512 (old) | 0.961 | 0.962 | 0.74 |
| **1024 (now)** | 0.965 | **0.980** | 1.11 |
| exact | 1.000 | 1.000 | 7.72 |

**The same setting measured recall@60 0.986 at 74k vectors and 0.962 at 157k.**
HNSW recall degrades as the graph grows — a fixed candidate budget visits a
smaller fraction of a bigger graph — and nothing in the system reports it. It
must be re-measured after any large ingest. Expect to raise it again at ~400k.

### Answer quality

| | 18k baseline | **41k final** |
|---|---|---|
| CITED | 15/20 | **19–20/20** |
| **GROUNDED** | 20/20 | **20/20** |
| CORRECT | 11/20 | **14–15/20** |
| DEGENERATE | 0/20 | **0/20** |
| ABSTAINS | 2/3 | **2/3** |
| latency p50 | 11.4 s | **13.3 s** |
| over 10 s | 12/23 | 16/23 |

### Four bugs found by running this, none of which were visible in the code

**1. A CUDA OOM 15,000 documents into the ingest — and batch size was not the cause.**
`max_seq_length` was never set, so it inherited bge-m3's advertised **8192**.
Attention is quadratic in sequence length and a batch pads to its longest
member, so peak VRAM was a function of the single worst chunk in the corpus. One
~1,900-token chunk asked for a **1.76 GiB** tensor. Wave A's Public Health /
Public Works annexures contain chunks the education cluster never had.
Fixed with `config.EMBED_MAX_SEQ=1024`: p50 494 / p95 662 / p99 768 tokens, only
**92 of 129,320 chunks (0.071%)** exceed it, and those keep their full text
everywhere except the embedding.
*The crash-consistency contract held exactly as designed* — SQLite ended up
AHEAD of FAISS by 3,607 rows and `reconcile()` dropped them on resume.

**2. `LLM_MAX_TOKENS=2048`, set undocumented on 2026-08-07.** Measured cost:
latency **p50 37.4 s / max 101.9 s**, 9/23 answers hitting the cap — and
**GROUNDED fell to 19/20**, because answers truncated mid-sentence emitted
citation numbers outside the valid range. `rag.py`'s own comment had predicted
"2048 would be ~50-80 s" before anyone measured it. Reverted to the calibrated
768; groundedness returned to 20/20.

**3. `max_final_k` was never the constraint — a clean negative result.**
12 → 6 left prompt tokens at **5,823, unchanged**, because `trim_to_budget` was
already discarding those blocks. The real lever was `context_token_budget`
(6000 → 3000): prompt 5,823 → 2,398, latency p50 34.1 → 17.2 s. `HANDOFF`'s
warning that 3500 measured worse did **not** reproduce — CORRECT held and CITED
improved — because that caution came from the 196-GR era with `rerank_pool=15`,
when the answer was often in block 2-3.

**4. The local model was running 10% on CPU for every measurement.**
`/health`'s `llm_placement` reported `fully_on_gpu: false`. The runbook says
"restart Ollama BEFORE the API", which is necessary but **not sufficient**:
`ollama serve` loads the model **lazily, on first request**, so it was still
loading *after* uvicorn had taken 1.2 GB for the reranker. Waiting on
`/api/tags` proves the daemon is up, not that the model is resident. Adding one
warmup `generate` call between the two fixed it:

| | 10% CPU offload | 100% GPU |
|---|---|---|
| generation p50 | 18.9 s | **10.1 s** |
| total p50 | 20.7 s | **13.3 s** |

Both models fit — 3,734 MiB (LLM) + 1,179 MiB (reranker) = **4,913 of 6,141**.
Capacity was never the problem; load ORDER was.

### Deterministic abstention (FR 3.3.5) — a design change, not a prompt tweak

Measured with **5 repeats per question** (n=3 with one sample is pure noise —
the "2/3 → 1/3 → 0/3" I first saw was one question flipping):

| out-of-corpus question | top score | abstained |
|---|---|---|
| PhD aerospace fee (EN) | 0.028 | 4/5 |
| Farm loan waiver (MR) | 0.825 | 0/5 |
| Passport application (EN) | 0.006 | **0/5** |

The passport question was answered with **fabricated Marathi prose carrying
citations**, while retrieval was already reporting 0.006 and
`low_confidence=True`. The signal was correct and available; the model ignored
it. Since §5.10 records six prompt variants that each traded one failure for
another, the fix is not a seventh prompt — `rag._hard_abstention` refuses in
code, below `config.abstain_floor` (0.10), **without calling the model at all**.

Three properties worth defending:
- **0.10, not near `rerank_threshold` (0.85).** Different questions: 0.85 decides
  which chunks are good enough to SHOW, 0.10 decides whether to speak at all.
  Relevant top hits measure p10 **0.946**, so the floor sits two orders of
  magnitude below the weakest real hit.
- **The refusal cites nothing** — the premise of firing is that nothing retrieved
  is relevant, so a citation would be the phantom provenance we exist to prevent.
- **Refusing got FASTER** (~2 s vs ~13 s): the gate runs before generation, which
  is 76% of latency.
- It refuses in the language of the **question**, not of the retrieved text —
  an English passport question had been answered in Marathi.

### Open, and honest

- **The SRS's 10 s NFR is still not met: p50 13.3 s, 16/23 over.** Generation is
  76% of it and completion is still ~517 tokens against a 200-350 expectation.
  Remaining levers trade against CORRECT, so this needs a measured decision, not
  another guess.
- **The farm-loan question still answers instead of refusing.** It scores 0.825 —
  far above the floor — because it genuinely retrieves loan-related GRs. A
  hard-gate cannot catch this; it needs the model or a much higher threshold.
- The 4 abstention/latency configs were each measured on ONE run of 23 questions.
  Single-point differences of 1 are noise (this is the repo's standing caveat).
