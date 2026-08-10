# MahaGR Assist — PLAN.md (next-stage, system-design plan)

Day-2 plan: make the corpus **complete and searchable at scale, fully locally**,
add the knowledge graph and admin dashboard, and finish local inference. The
initial build is in `ROADMAP.md`; this file is the plan for the four post-build
goals. Implement **one phase at a time, top to bottom.** After each phase the
assistant explains it + gives interview questions (standing agreement).

**Everything here runs on your machine — no cloud, no embeddings leaving the box.**

**Honest one-day reality:** all of this is codeable in a day EXCEPT embedding the
*entire* ~100k-GR corpus (GPU-hours). So we ingest a **few education-cluster
departments (~13–18k GRs, ~65–90k chunks)** to *prove* the scalable architecture;
the pipeline is identical for the full set and can keep embedding in the background.

---

## Phase 1 — Finish local LLM (Ollama)  *(pipeline DONE; model choice open)*
**State:** ✅ `qwen2.5:3b` pulled to `/mnt/win/mahagr/ollama` (1.9 GB, root not
touched) · ✅ `.env` set (`LLM_PROVIDER=ollama`, `OLLAMA_MODEL=qwen2.5:3b`,
`EMBED_DEVICE=cpu`) · ✅ backend serves fully locally · ✅ offline **proven**
(`scripts/verify_offline.py`) · ⏳ persistent server = `deploy/ollama.service`,
awaiting a one-time install by the owner.

**Three real bugs found and fixed while verifying (each was silent):**
1. **`.env` was loaded too late.** `engine/config.py` read `os.environ` at import
   time but `load_dotenv()` only ran later in `rag.py`, so `EMBED_DEVICE=cpu` was
   silently ignored. `.env` is now loaded in `config.py` (explicit path, works
   from any cwd).
2. **The cross-encoder was on CPU: 27.6 s/query.** `EMBED_DEVICE` controlled the
   embedder *and* the reranker, but their costs are opposite — the embedder does
   one short query (0.1 s on CPU), the reranker does 15 long (query, chunk) pairs.
   New `RERANK_DEVICE` (auto→GPU) + fp16 → **0.33 s**. End-to-end p50 went
   **38 s → 2.6 s**, which is what actually meets the SRS's <10 s.
3. **`/ask` silently dropped `language`.** The portal's EN/मराठी toggle was a
   no-op on the main path (FR 3.4.2 / 3.4.5). Threaded through; ISO codes are now
   mapped to real language names ("mr" → "Marathi (मराठी)"), which also fixed the
   same latent flaw in `/summarize`, `/explain`, `/compare`.

**Measured on the 23-Q gold set** (`scripts/eval_answers.py`, new): CITED 19/20 ·
GROUNDED 20/20 (no phantoms) · CORRECT 18/20 · ABSTAINS 2/3 · latency p50 2.6 s,
max 8.3 s, **0/23 over 10 s**.

**The open item — `qwen2.5:3b` is too weak to write the answer.** Retrieval and
citation resolution are solid, but **6–8 of 20** replies are a bare `[1]` with no
sentence. Six prompt variants were measured; each fixed one failure and caused
another (demanding citations harder → it stopped abstaining; adding a worked
example → it emitted the *example* as a cited answer, i.e. confidently wrong).
The prompt is now at its best measured setting. **This is a model-capacity limit,
not a prompt bug** — see the decision in `HANDOFF.md` §7.

**Interview Qs:** quantization (GGUF/Q4) and why Q4 damage shows up as repetition
on a 3B; why Ollama *is* the inference server (no vLLM/TGI needed); VRAM math for
a 6 GB card (3B 2.2 GB + reranker fp16 1.1 GB); bi-encoder vs cross-encoder cost
and why only one of them needs the GPU; on-prem privacy vs Groq's throughput.

---

## Phase 2 — Full-corpus, searchable at scale — LOCAL (FAISS-HNSW + SQLite)  ✅ **IMPLEMENTED**

**Delivered:** 18,078 GRs across 6 departments on `IndexHNSWFlat` + SQLite
(`engine/corpus_db.py`), two-stage `CorpusRetriever`, BM25 moved to SQLite FTS5,
native FAISS metadata filtering, paginated `/documents` + new `/corpus/stats`,
frontend scope filters + corpus stat. Tests 46 → 82. Details and measured
numbers in `CHECKLIST.md`; the traps are `HANDOFF.md` §5.20–§5.26.

**⚠ Still to verify (needs the index + API, not a re-ingest):** calibrate
`efSearch` (`scripts/eval_ann.py`) and recalibrate the retrieval thresholds on
the bigger corpus (`scripts/eval_retrieval.py` → `scripts/eval_answers.py`).
The only measured retrieval numbers in the repo are from the 196-GR flat index.

The original plan for this phase follows, kept for the reasoning.

**Today's limit.** The index is FAISS `IndexFlatIP` — *brute force*, O(n)/query —
with all chunk texts in a RAM JSON sidecar. Fine for 713 vectors; it dies at the
real corpus. The SRS wants *"vector embedding for each chunk … in a vector
database for efficient indexing"* and *"scalable across departments"*, and the
presentation feedback was **"the whole corpus must be searchable — that's the
point of RAG."**

**Local vector-store options (all run on your machine, no cloud):**
- **FAISS `IndexHNSWFlat`** — in-process HNSW, **no server**, save/load from disk.
  ← *our choice* (simplest, fully local, reuses our stack).
- **Qdrant / pgvector-in-local-Docker** — also 100% local; add metadata filtering
  + a real DB, but introduce a server. Keep as a later upgrade if filtering at
  scale becomes the bottleneck.
- (FYI the "cloud" worry only applies to *hosted* Postgres like Supabase — not to
  Postgres/Qdrant you run yourself.)

**Target architecture:**
- **Vectors → FAISS `IndexHNSWFlat`** (dim 1024, cosine on normalized vectors).
  HNSW = a layered "small-world" graph; search hops greedily → ~**O(log n)**
  instead of O(n). Tunables: `M`≈32 (edges/node), `efConstruction`≈200 (build
  quality), `efSearch` (query recall↔latency). Index persisted to `/mnt/win`.
- **Text + metadata → SQLite** (we already have `app/db.py`; extend it):
  `gr_documents(id, gr_number, department, date, category, language, title, text)`
  and `gr_chunks(faiss_id, gr_id, chunk_index, page_start, page_end, content_type, text)`.
  FAISS returns integer positions → look up `gr_chunks` by `faiss_id` in SQLite →
  **no chunk text in RAM** (that's the scale fix).
- **Two-stage retrieval:** HNSW ANN top-k chunks → **group by `gr_id`** (dedupe to
  whole GRs — your presentation point) → fetch GR rows → **cross-encoder rerank**
  → grounded answer. Chunks find the passage; the GR is the citable unit.
- **Metadata filtering — CORRECTED DURING IMPLEMENTATION.** This plan said
  "FAISS-HNSW can't pre-filter natively" and proposed over-fetch-then-filter.
  That is out of date: **faiss 1.14 supports `SearchParametersHNSW` + an
  `IDSelector`**, verified on this machine (a filtered search returned only
  allowed ids, and still filled top-10 with a 1%-selectivity filter). So the
  allowed-id set is computed in SQLite and pushed **into** the search, which
  beats a post-filter — a post-filter silently loses recall when the whole
  top-k belongs to an excluded department. The honest remaining limit: the
  graph is still built over ALL vectors, so a very narrow filter makes the walk
  traverse mostly-rejected nodes (we raise `efSearch` when a filter is active to
  compensate). Qdrant/pgvector's native filtered indexes remain the upgrade path.

**Which departments (the demo corpus).** PS targets **Higher & Technical
Education**; ingest the **education & skilling cluster**:
- Higher & Technical Education (~4,725) — core
- School Education & Sports (~5,234)
- Skill Development & Entrepreneurship (~1,396)
- Medical Education & Drugs (~1,923)
- *stretch:* Social Justice & Special Assistance (~2,235) + Tribal Development (~2,538) — scholarships
→ ~13k GRs (core 4) / ~18k (with stretch) → ~65–90k chunks. **Ingest HTE first,
then add departments** as disk/time allow — "as much as possible."

**Device allocation (GPU vs CPU — the confusion, resolved):**
- **Ingestion (one-time bulk embed): use the GPU.** Run the batch job with
  `EMBED_DEVICE` unset (→ cuda) and **Ollama not running**, so the full 6 GB is for
  bge-m3. ~hundreds of chunks/sec → 65k chunks in ~30–60 min (hours on CPU).
- **Serving (per query): embedder+reranker on CPU (`EMBED_DEVICE=cpu`), LLM on
  GPU (Ollama).** One query embedding is tiny; CPU adds ~0.1–0.3 s. The GPU is
  worth more to the LLM at serving time.
- So the *same* GPU is used for embeddings during ingestion and for the LLM during
  serving — never both at once.

**Steps (backend):** extend `fetch_mahgrs.py` for multiple departments (resumable)
→ SQLite schema above → `HnswStore` (FAISS `IndexHNSWFlat`) alongside `FaissStore`,
selected by `VECTOR_BACKEND` → `scripts/ingest_corpus.py` (resumable, idempotent,
GPU batch-embed, writes SQLite + FAISS) → rewire `retrieval.py` to the two-stage
flow + recalibrate thresholds on the bigger corpus → persist/load the HNSW index
from `/mnt/win`.
**Steps (frontend):** department / date / language **filters**; a **corpus-size
stat** ("searching N GRs across M departments").

**One more thing that had to change, not anticipated above: BM25.** The sparse
half of hybrid retrieval used `rank_bm25`, which holds a tokenized copy of the
whole corpus in RAM (a dict per document) — fine for 713 chunks, several GB at
65k. It is now **SQLite FTS5**, which does the same BM25 ranking off disk in the
database we already have. Verified that FTS5's `unicode61` tokenizer handles
Devanagari correctly (whole words, not split at vowel marks — the HANDOFF §5.4
bug in its FTS form). FTS5's MATCH syntax treats `-`, `/`, `.`, `*` and `OR` as
operators, so a raw question or GR number must be tokenized and quoted first or
it raises a syntax error.

**Done when:** a query searches the whole ingested multi-department corpus in
<1–2 s, returns the right GR, the HNSW index persists across restarts, and filters
work.

**Interview Qs:** exact vs approximate NN (recall/latency); explain HNSW +
`M`/`efConstruction`/`efSearch`; why two-stage (chunk ANN → group to GR → rerank);
FAISS's filtering limitation and your work-arounds; **GPU-ingest vs CPU-serve**
device split and why; storage math (1M×1024×4 ≈ 4 GB) and how you'd sharded past
laptop scale; local vector-store options and why FAISS-HNSW here.

---

## Phase 2.5 — Table extraction from OCR'd text (UNPLANNED, done 2026-08-07)  ✅

Not in the original plan; discovered while testing fee/quota questions. The
corpus is orgpedia's pre-OCR'd `.mr.txt`, so **pdfplumber cannot help** — there
is no PDF structure left. Tables arrive as a raw word stream, which `bge-m3`
(trained on prose) embeds badly, so a fee-schedule question matched nothing.

`engine/text_table_detect.py` detects table-like text directly and converts each
row into the same `row_to_sentence` prose the PDF pipeline already produced
("In this row: अ.क्र. is 1, प्रवर्ग is खुला, शुल्क is 5,000"). Measured across the
corpus: **pipe-delimited tables in 7,611 GRs (42%)**, dash-separated in 1,805.
Deliberately conservative — ≥3 pipes per line, ≥2 rows, consistent pipe count —
because a false positive (prose mangled into broken row sentences) is worse than
a false negative (a table left as prose, i.e. no worse than before). Numbered
lists are explicitly NOT treated as tables.

**This forced a full re-ingest** (2026-08-07): chunks **64,744 → 74,004**, of
which **17,736 are now `content_type='table'`**. The previous index was kept at
`/mnt/win/mahagr/index_v1/` and can be deleted once the new one is trusted.
The 2 synthetic fee-table fixtures were also added to the hnsw corpus this time
(18,078 → **18,080** documents), so the fixture gold questions are no longer
structurally unanswerable on the scaled index.

**Tests:** `test_text_table_detect.py` + `test_text_table_ingest.py`.

---

## Phase 3 — Supersede knowledge graph + visualization  ✅ **DONE**

A **domain knowledge graph** (nodes = GRs, edges = *supersedes/amends/cites* from
the `references`/`supersedes` fields `gr_metadata` already parses). **Not Graph
RAG** (which has an LLM extract entities/relations from all text and *retrieves by
traversing* — a different, bigger thing).

### Why this is much easier now than when this plan was written
Phase 2 already stores, for every one of the 18,078 GRs, its parsed
`refs` (a JSON list of cited GR numbers) and a `supersedes` flag, in
`gr_documents`. So building the edge table is **one SQL pass over a table that
is already populated** — no re-parsing, no re-embedding, no GPU. That also means
**all of the Phase-3 code and its tests can be written and unit-tested without
the corpus**, against synthetic rows in a tmp SQLite file.

### The hard part is NOT the graph, it is resolving a reference to a document
`refs` holds GR *numbers as printed in the text* (`संकीर्ण-२०२३/प्र.क्र.४५/तांशि-४`),
which are OCR output and therefore noisy: spacing differs, Devanagari vs ASCII
digits differ, and a reference may point at a GR that is not in our corpus at all.
So resolution must be **normalised and explicitly three-valued**:
`resolved` (matched a document) / `dangling` (a real reference we don't hold) /
`ambiguous` (matched more than one). Silently dropping danglers would overstate
how complete the graph is — for a government tool, "this GR cites something we
don't have" is *information*, not an error.

### Steps (backend)
1. **`gr_edges(src_id, dst_id, dst_number, kind, resolved)`** in
   `engine/corpus_db.py`, `kind ∈ {supersedes, cites}`. Index both directions
   (`src_id`, `dst_id`) — traversal goes both ways.
2. **A normaliser** for GR numbers (Devanagari→ASCII digits, whitespace and
   punctuation collapse) so `refs` entries and `gr_documents.gr_number` compare
   on a canonical form. Store the canonical form in a column so the join is an
   index seek, not a `LIKE` scan over 18k rows per edge.
3. **`scripts/build_graph.py`** — one pass, idempotent, rebuildable, no GPU.
   Reports resolved/dangling/ambiguous counts.
4. **Traversal in `engine/graph.py`**: `neighbourhood(gr, depth)` and
   `supersede_chain(gr)` (transitive closure, newest-first). **Must guard
   cycles** — GR A cites B and B cites A does happen with amendments — with a
   visited set and a depth cap, or the traversal hangs the API.
5. **`GET /graph/{doc_id}?depth=2`** → `{nodes, edges, chain, dangling}`.
6. Rewire `officer.supersession` to read `gr_edges` instead of scanning, and
   have `/ask`'s conflict warnings use the **chain**, not just direct edges —
   "superseded by X, which was itself superseded by Y" is the answer an officer
   actually needs.

### Steps (frontend)
7. Graph panel on the Browse detail (**react-flow** or **cytoscape.js**),
   click-a-node-to-navigate, superseded nodes visually struck through, dangling
   references shown as ghost nodes so the gap is visible.

### Done when
A GR with a real supersede chain renders it, clicking a node navigates, cycles
don't hang, and `/ask` warns using the full chain.

### STATUS: BUILT, RUN AND VERIFIED (40 tests)
Steps 1-7 are implemented — `gr_edges` + schema migration,
`gr_metadata.canonical_number()`, `engine/graph.py`, `scripts/build_graph.py`,
both endpoints, `officer.supersession` reading the graph (with a fallback to the
old scan when it has not been built), `/ask` warnings using the transitive
chain, and `frontend/components/graph.tsx`.

**Running it is what found the real bug.** The graph built in seconds and was
almost edgeless: **2.0%** of references resolved. The fault was upstream, in
extraction — a GR number was being matched as a whitespace-bounded token, but
real numbers CONTAIN spaces (`एनजीसी-२०१०/(१९३/१०) /मशि-४`), so every reference
came out as a fragment. Reference lines are now parsed the way they are written
(items → comma-separated segments → trimmed at the date), the cited **date** is
extracted and used to separate documents that share a number, and
`scripts/reparse_refs.py` re-derives it all from the stored text — a re-PARSE,
not a re-ingest, so no GPU and no re-embedding.

**Measured on the 18,078-GR corpus:** 60,420 edges, **5,753 resolved (2.0% →
9.5%)**, 1,771 of them disambiguated by the cited date, **3,904 documents (was
1,123) now carry at least one resolved edge**. The remaining unresolved
references are mostly out of reach by construction: only ~21% of them even name
one of the 6 departments we ingested (of Maharashtra's ~33). Details and the
traps are in `CHECKLIST.md` and `HANDOFF.md` §5.28.

**Interview Qs:** knowledge graph vs Graph RAG; edges-in-a-relational-table vs a
graph DB (Neo4j) and when that flips; computing a supersede chain (transitive
closure) + guarding cycles / dangling references; why reference *resolution* is
the real problem and how you'd measure its precision/recall.

---

## Phase 4 — Admin dashboard (auth + roles + audit)  🟡 **IMPLEMENTED 2026-08-07, NOT YET TESTED**

**Shipped:** `backend/app/auth.py` (bcrypt hashing, JWT via PyJWT, `get_current_user`,
`require_role`, an in-process token-bucket rate limiter at 20 req/min/user, and the
`DEV_NO_AUTH=1` escape hatch), `users` + `audit_log` tables in `app/db.py`,
`POST /auth/login`, `GET /admin/audit-logs` (gated on `IT Admin`), every other route
carrying a `current_user` dependency, `/ask` writing an audit row (question +
cited GR numbers + ip — **not** the answer text), `scripts/seed_users.py` for the
four SRS roles, and on the frontend `app/login/page.tsx`, `components/UserMenu.tsx`
and `app/(portal)/admin/page.tsx`.

**What is NOT done — do not claim this phase closed:**
- **Zero tests.** `backend/tests/` has nothing for auth, roles, rate limiting or
  the audit trail. Every other module in this repo is covered; this one is not.
- **`DEV_NO_AUTH=1` is currently ON in `backend/.env`**, so nothing is actually
  gated right now and the login page is decorative until it is turned off.
- **`JWT_SECRET` defaults to the literal `"maha-secret"`** and `.env` sets exactly
  that. Fine for a demo, indefensible in a viva if asked — it needs to be a
  generated secret with the default removed.
- The **Reviewer** role ("see other officers' searches") is not wired —
  `db.list_conversations(all_users=True)` exists but no route uses it.
- The rate limiter is **in-process**, so it resets on restart and does not survive
  more than one worker. Say so rather than implying it is durable.

The original plan for this phase follows.



Know which officer ran which search; logs; the SRS roles (Desk Officer, Legal
Translator, Reviewer, IT Admin) and on-prem security. This is **FR 3.7.1**
("a *secure* web-based portal") — currently the API has no auth at all.

**None of this needs the corpus or the GPU** — it is all `app/db.py`, FastAPI
dependencies and frontend. It can be built and unit-tested the same way Phase 3
was.

### Where it lives, and why
`users` and `audit_log` go in **`app/db.py`**, NOT `engine/corpus_db.py`. That
split is already established and load-bearing: `engine/` is the retrieval engine
and must never import `app/`, while who-logged-in is portal state that belongs
next to conversations and feedback. It also means a corpus rebuild can never
touch the audit trail — which for a government system is exactly the property
you want to be able to state out loud.

### Steps
1. **Schema** — `users(id, username, password_hash, role, created_at, active)`
   and `audit_log(id, user_id, action, detail, ip, ts)`. Passwords hashed with
   **bcrypt/argon2 via passlib** — never a bare SHA. Seed script for the four
   SRS roles.
2. **JWT login** — `POST /auth/login` → short-lived access token; a
   `current_user` FastAPI dependency; `require_role(...)` for scoping.
   Store the token in memory/`sessionStorage`, **not `localStorage`** (XSS
   reach), and be ready to say why.
3. **Role scoping**, least privilege, mapped to the SRS roles:
   | Role | Can |
   |---|---|
   | Desk Officer | ask, browse, summarize, feedback |
   | Legal Translator | + force-language answers |
   | Reviewer | + see other officers' searches (read-only) |
   | IT Admin | + `/admin`, user management, audit export |
4. **Audit every `/ask`** — user, timestamp, IP, the question, and the GR ids
   cited. **Deliberately NOT the answer text or document bodies**: the audit
   trail must prove *who asked what*, not become a second uncontrolled copy of
   the corpus. Say this in the viva — knowing what NOT to log is the point.
5. **Per-user rate limiting** (token bucket, in SQLite or in-process) so one
   officer cannot exhaust a shared local LLM. Must degrade with a clear message,
   not a 500.
6. **`/admin` page** — recent searches, per-officer activity, feedback review,
   usage. Role-gated on the server too, never only in the UI.

### Two things that are easy to get wrong
- **Auth must not break the existing demo.** Add a `DEV_NO_AUTH=1` escape hatch
  (defaulting OFF) or the presentation stops working the moment this lands.
- **The frontend has no auth state today.** A login page, a token-carrying
  `lib/api.ts`, and a 401 → redirect path all have to arrive together, or every
  page breaks at once.

**Interview Qs:** JWT vs sessions (storage/expiry/refresh, and why not
`localStorage`); RBAC + least privilege for a gov system; what to log for audit
vs what NOT (PII, document text) and why; rate limiting (token bucket) for a
shared cap; how you'd prove the audit trail wasn't tampered with (append-only,
hash chain) if a reviewer asked.

---

## Non-blocking backlog (small, no corpus needed)
Worth picking up in any spare slot — none of these need the GPU or the index:
- ~~Download the referenced GR from a citation — SRS FR 3.7.4.~~ **DONE:**
  `resolve_citations` now returns a `doc` id, so every citation carries an
  "open" link and a "download" button, and Browse honours `?doc=<id>`.
  Downloaded as `.txt` (the corpus is pre-OCR'd text — a re-rendered PDF would
  be a fabricated artifact, not the source).
- Mobile/responsive sidebar on the Ask page.
- `officer.compare` still caps at 6 chunks per side; at corpus scale it should
  pick the *most relevant* chunks, not the first six.
- The 16 `Original Maha-GR` PDFs still need an OCR re-ingest once
  `tesseract-data-mar` is installed (blocked on the owner, ROADMAP Phase 1).

---

## Phase 5 — UX polish + presentation refresh  🟡 **LARGELY DONE 2026-08-07**

The portal was rebuilt on **shadcn/ui + Radix + lucide** (`components/ui/*`,
`components.json`, `lib/utils.ts`) and re-laid-out:
- `app/page.tsx` is now a **public landing page** (883 lines — architecture
  diagrams, the on-prem story, a live corpus stat) instead of the chat.
- The officer portal moved into a route group `app/(portal)/` with a shared
  layout: `/ask`, `/browse`, `/admin`.
- `app/login/page.tsx`, `components/UserMenu.tsx`, `components/nav-link.tsx`,
  `components/landing/{diagrams,live-corpus-stat,site-chrome}.tsx` are new.
- `npx tsc --noEmit` is **clean** (verified 2026-08-09).

*Still open:* mobile/responsive sidebar on Ask; loading/empty/error states are
uneven; the deck and `DEMO.md` still describe the OLD single-page portal and the
old corpus numbers.

---

## Phase 6 — Complete the corpus + complete the graph  ✅ **DONE 2026-08-10**

> **STATUS: THE CORPUS IS COMPLETE.** All **33 departments / 99,410 GRs /
> 401,573 vectors**; graph **317,250 edges, 46,412 resolved, 30,799 documents
> linked**. Done in two waves, and the comparison between them is the result
> worth presenting:
>
> | | wave A (+6 depts, 23,394 docs) | wave B/C (+21 depts, 57,939 docs) |
> |---|---|---|
> | resolved edges gained (education docs) | **+2,886** | **+20** |
> | per 1,000 documents ingested | **144** | **0.35** |
>
> A ~400x difference in marginal value, predicted in advance by
> `scripts/cited_departments.py`. Wave B/C bought **completeness**, not graph
> quality — and completeness is what makes the 14.63% resolution rate an
> ABSOLUTE ceiling: holding the entire published dataset, **76% of references
> still point outside it**. That is a finding about Maharashtra's published
> record, not about this pipeline.
>
> Scaling cost almost nothing: 2.4x the documents, **hit@5 unchanged at 18/20**.
> From 196 to 99,410 documents — 500x — hit@5 went 20/20 → 18/20.
>
> Every measured table, and the seven bugs the two waves surfaced, are in
> `CHECKLIST.md` → PLAN Phase 6 and `HANDOFF.md` §5.29-§5.38.

This is the phase the owner asked for on 2026-08-09: ingest **every** GR in the
dataset, from **every** department, then rebuild the knowledge graph on top of it.

### The numbers (measured with `scripts/fetch_mahgrs.py --list`, 2026-08-09)

| | now | after this phase |
|---|---|---|
| departments | 6 of 33 | **33 of 33** |
| GRs | 18,080 | **99,421** |
| raw `.mr.txt` | 244 MB | ~1.12 GB |
| chunks = vectors | 74,004 | **~390,000** (est., scaled on text volume) |
| `corpus.hnsw` | 323 MB | ~1.7 GB |
| `corpus.db` | 629 MB | ~3.3 GB |

All of it lives on `/mnt/win` (**137 GB free**, no longer the 47 GB in older
notes). Root `/` — still at 2.1 GB free — is not touched by any of this.

### Why this is the fix for the GRAPH, not just "more data"
Reference resolution sits at **9.5%**, and the measured reason is *corpus
coverage*, not parsing: only ~21% of unresolved references even name one of the
six departments we hold. Every reference to a Finance / GAD / Revenue order is
`dangling` **by construction**. Completing the corpus is therefore the single
change that can move that number, and the movement is itself the presentable
result — "dangling" collapsing into "resolved" as the cited departments arrive.

### Ingestion ORDER was measured, not guessed — `scripts/cited_departments.py`

Before downloading anything, the corpus was asked which departments it actually
**cites**: for all 18,080 documents, take the `वाचा` (references) block — the
same scope the edge builder uses — and count which department is named in it.

| department | docs citing it | GRs to ingest | held? |
|---|---|---|---|
| **Finance** | **5,624 (32.4%)** | **1,046** | ❌ |
| School Education & Sports | 3,785 (21.8%) | — | ✅ |
| Tribal Development | 2,531 (14.6%) | — | ✅ |
| Social Justice | 2,153 (12.4%) | — | ✅ |
| Medical Education | 1,680 (9.7%) | — | ✅ |
| **General Administration** | 1,670 (9.6%) | 6,345 | ❌ |
| **Industries, Energy & Labour** | 1,602 (9.2%) | 2,885 | ❌ |
| Higher & Technical Education | 1,416 (8.2%) | — | ✅ |
| **Public Health** | 942 (5.4%) | 7,076 | ❌ |
| **Public Works** | 682 (3.9%) | 3,809 | ❌ |
| Skill Development | 609 (3.5%) | — | ✅ |
| **Planning** | 526 (3.0%) | 2,233 | ❌ |

17,357 of 18,080 documents have a reference block; 3,836 of those name no
department at all.

**Finance is the entire argument for measuring instead of guessing.** A **third
of the corpus cites it**, and it is one of the *smallest* departments in the
dataset — 1,046 GRs, 12.4 MB, a few minutes of GPU. Every one of those thousands
of references is `dangling` today for no better reason than that we never
downloaded 12 MB. It is the best effort-to-payoff ratio in the whole project.

**It also corrected the docs.** `HANDOFF.md` had asserted the unresolved
references were mostly "Finance / GAD / Revenue orders". Finance and GAD are
right; **Revenue & Forest measured 0.5% and Law & Judiciary 0.3%** — both were
plausible, both were wrong, and both would have cost ~10,000 GRs of download and
GPU for almost no graph gain.

### Steps

**6.1 — Fetch, in two waves.**
- **Wave A — `--cluster cited`**: the 6 most-cited departments we do not hold
  (Finance, General Administration, Industries/Energy/Labour, Public Health,
  Public Works, Planning) = **23,394 GRs / 221 MB**. Roughly doubles the corpus
  and targets the graph directly.
- **Wave B — `--all`**: the remaining 21 departments, sorted **smallest first**
  (the download is resumable, so if it stops early, 20 finished small
  departments beat being 60% through one large one).

`fetch_mahgrs.py` was already resumable and idempotent; this phase added
`--all`, the `cited` cluster, and **memoization of `list_department`** —
unauthenticated GitHub allows only **60 API requests/hour** and each listing
costs one, so `--all` listing every department twice (once to sort, once to
download) would silently exhaust the quota mid-run.

**6.2 — Ingest incrementally into the SAME index.** `ingest_corpus.py` skips
every id already in `gr_documents`, so this **adds** ~316k vectors to the
existing `corpus.hnsw` / `corpus.db` — the 74,004 already embedded are never
recomputed. `EMBED_DEVICE=cuda`, `--batch-size 16` (fp16 — §5.24), Ollama and
uvicorn stopped. ~2.5–3.5 h of GPU. Checkpoint every 1000 documents; the
crash-recovery path is already proven.

**6.3 — Verify the two files still agree** — `scripts/verify_corpus.py` (new).
`ingest_corpus.py` only checks `len(store) == COUNT(*)`, and **equal counts do
not prove equal alignment**: an off-by-500 shift in the middle of the corpus
leaves both numbers identical while every citation silently names the wrong GR.
So the script re-embeds a random sample of chunks and scores each against **its
own** vector at its recorded `faiss_id`. Correct alignment measures ≈ **0.9998**
(not 1.0 — the corpus was embedded fp16 on GPU, the check runs fp32 on CPU);
a shifted index scores 0.3–0.8, so the threshold sits in a very wide gap.
Deliberately uses `score_for_index` rather than a search, because a search could
return a near-duplicate neighbouring chunk and mask the shift.
**Baseline before wave A: 74,004 vectors, 74,004 rows, 18,080 documents, worst
self-similarity 0.9998, 0 orphans — consistent.**

**6.4 — Rebuild the graph.** `scripts/build_graph.py` — one SQL pass, minutes, no
GPU. `reparse_refs.py` is **not** needed: newly ingested documents get their
`refs` parsed by the current (fixed) parser at ingest time. Report the
resolved/ambiguous/dangling table before and after.

**6.5 — Re-measure everything a 5× corpus invalidates.** In this order:
`eval_ann.py` (efSearch=512 was calibrated at 74k vectors; HNSW recall degrades
with graph size, and efSearch is a query-time dial needing no re-ingest) →
`eval_retrieval.py` (thresholds, `rerank_pool`) → `eval_answers.py` (needs the
API up). **Expect accuracy to drop again** unless `rerank_threshold` moves off
its 196-GR value of 0.85 — which is exactly why that tuning is scheduled *after*
this phase, not before: tuning against a corpus that is about to quintuple is
wasted work.

**6.6 — Portal + docs.** `/corpus/stats` already drives "Searching N GRs across M
departments" and will read 99,421 / 33 with no code change; the department filter
list grows to 33 on its own. Refresh `DEMO.md`, `README.md` and the deck.

### Done when
`/health` reports 33 departments and ~99,421 documents · index and DB agree ·
`build_graph.py` reports a resolution rate materially above 9.5% · fresh
`eval_ann` / `eval_retrieval` / `eval_answers` numbers are recorded in
`CHECKLIST.md`.

### What the resolution rate actually measures — the trap in the headline number

It reads like an accuracy score ("we only got 10% right"). It is not. A
`dangling` edge is usually **correct** work: the number was parsed properly and
the order it names simply is not in the database. When a School Education GR
cites a Finance order, that is an **inventory gap, not a bug**.

The nuance to volunteer before anyone catches it: the rate moves for **two
independent reasons**, and they must not be conflated.

1. **Parse quality** — 2.0% → 9.5% came from fixing the reference parser
   (GR numbers contain spaces; the old pattern cut them into fragments).
   That was a genuine accuracy fix.
2. **Corpus coverage** — this phase. Same parser, more documents to match
   against.

So state it as: *resolution rate is a lower bound on graph completeness, and it
is a composite of parse quality and corpus coverage.* Quoting it without saying
which of the two you just changed is the easiest way to get caught out.

And the reason danglers are kept rather than deleted (which would show a
flattering 100%): "this GR builds on an order we do not hold" is information an
officer needs — it is why the frontend draws them as dashed ghost nodes — and
keeping them is what makes the rate an honest measure instead of a self-graded
exam. Same logic for `ambiguous`: 2,138 canonical numbers are held by more than
one document, the cited **date** breaks the tie where it can (1,771 edges), and
where it fits nobody the edge stays ambiguous. **It is never guessed** — a
fabricated supersession is far worse than a missing one.

**The falsifiable prediction for this phase:** Finance is cited by 32.4% of the
corpus and all 1,046 of its GRs are now ingested, so those citations should
convert dangling → resolved in bulk. If the rate does *not* move meaningfully,
that is a real finding: it would mean the remaining danglers are parse failures
or genuinely out-of-dataset orders, not inventory gaps.

### What this phase will NOT fix — say this before someone asks
**Latency.** Abstentions return in 2.6–3.2 s while full answers take 12–16 s on
identical retrieval, so *generation* dominates and generation cost is set by
prompt size, not corpus size. The SRS's 10 s NFR still needs the
`rerank_threshold` / `max_final_k` / `num_predict` work (HANDOFF §0 JOB 2).

### Interview Qs
Why 27 more departments helps the **graph** but endangers **retrieval** (more
resolvable references, but also far more plausible-but-wrong chunks) · storage
math for 390k × 1024-d fp32 and when you would move to IVF-PQ or shard ·
why the ingest is incremental rather than a rebuild (id-keyed skip + HNSW
supports incremental `add`) · how you prove FAISS and SQLite still agree after
adding 316k vectors · why HNSW `efSearch` must be re-calibrated when the graph
grows · what "9.5% resolution" actually measures (corpus completeness, not
parser quality).

---

## Cross-cutting system-design notes (for any interview)
- **Everything local:** embeddings (bge-m3), ANN (FAISS-HNSW), rerank, docs/graph/
  audit (SQLite), and the LLM (Ollama) all run on-prem — no data leaves the machine.
- **Separation of concerns:** vector index (ANN) ≠ document/metadata store (SQLite)
  ≠ generator (LLM). Each swaps/scales independently (that's why FAISS→Qdrant or
  Groq→Ollama are config changes, not rewrites).
- **Two-stage retrieval** (fast ANN recall → precise rerank) buys speed *and*
  quality at scale.
- **Idempotent, resumable batch ingestion** is mandatory at 10k+-doc scale.
- **One GPU, two jobs:** it embeds the corpus during ingestion and runs the LLM
  during serving — scheduled so they never contend.
