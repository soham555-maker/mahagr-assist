# MahaGR Assist — completion checklist

Remaining work to take MahaGR from "demo-ready" to "complete + deployed".
Seven phases; the last is deployment. Check items off as you go.

**Status so far (done):** multilingual engine (bge-m3 + bge-reranker-v2-m3),
OCR fallback, Devanagari-aware BM25, GR metadata + supersede parsing, officer
features (summarize/explain/compare/related/supersede), slim FastAPI, Next.js
portal, model-free test suite, calibrated thresholds, swappable Groq/Ollama LLM.

---

## Phase 1 — Corpus & retrieval completeness
Goal: the index is built from your own real GRs (incl. scanned) and tuned on real data.

- [~] Ingest the 16 `Original Maha-GR` PDFs — done via text layer, but it's garbled
      (broken font encoding); pruned. Re-ingest with OCR once tesseract-mar is
      installed: `python scripts/add_pdfs.py "../Original Maha-GR" index --ocr`
- [ ] Verify OCR on a real government PDF (blocked on `tesseract-data-mar` install)
- [x] Expand the gold set from 6 → 23 questions (EN + Marathi, fee-number, out-of-corpus)
- [x] Re-run the calibration harness and re-set thresholds (rerank 0.80→0.85; text 0.55 held)
- [x] Refine metadata parsing (added शासन आदेश / आदेश / परिपत्रक / अधिसूचना number labels)

**Done when:** index built from your GRs; strong hit@k on the real gold set (hit@1 19/20,
MRR 0.975 ✓); OCR confirmed on a real government PDF (pending tesseract-mar).

## Phase 2 — Officer-assistance completeness
Goal: every SRS officer-help feature works and is exposed in the UI.

- [x] Conflict/supersede detection: `officer.supersede_warnings` flags a cited GR that a
      newer GR supersedes; surfaced on `/ask` and as an amber banner in the portal (verified)
- [x] Every feature has an API route (ask/summarize/explain/compare/supersede/related)
- [x] Every feature has a frontend action: summarize = Browse "Summarize" button;
      explain = Ask "Explain simply" mode; compare/supersede/related = Browse; ask = chat

**Done when:** all FR-3.5 features are clickable in the portal and grounded/cited. ✓

## Phase 3 — Persistence & memory
Goal: conversations and feedback survive restarts.

- [x] Local SQLite store (`app/db.py`): conversations, messages (with sources + warnings), feedback
- [x] Wire stored history into query-rewrite — `/ask` loads the conversation's turns from the DB;
      verified a follow-up ("And what about the Open category?") resolved to 12000 across turns
- [x] Conversation-history sidebar in the portal + feedback thumbs (up/down) per answer

**Done when:** close the app, reopen, and past chats + feedback are still there. ✓ (SQLite file at data/db/)

## Phase 4 — Auth, roles & security
Goal: behaves like a real government portal, not an open demo.

- [ ] Officer login (JWT sessions)
- [ ] Roles from the SRS (Desk Officer, Legal Translator, Reviewer, IT Admin) with scoped access
- [ ] Rate limiting / per-user daily quotas (graceful when the LLM limit is hit)
- [ ] Audit log (who asked what, when)

**Done when:** a user logs in, actions are role-gated, and requests are logged.

## Phase 5 — Frontend / UX completeness & polish
Goal: the portal is complete, clear, and Marathi-correct.

- [ ] Ask: language toggle + switch language mid-conversation
- [ ] Ask: clear abstention state; citations that open/download the source GR
- [ ] Browse: search + filter by department / date / language; read + download
- [ ] Polished Compare, Supersede, Related, Feedback panels
- [ ] Loading / empty / error states; responsive; verified Devanagari font rendering

**Done when:** every feature is reachable and the UI degrades gracefully (no raw errors).

## Phase 6 — On-prem LLM + testing + observability
Goal: prove "runs locally, nothing leaves" and lock quality.

- [ ] Run end-to-end on Ollama with a local Llama; tune the prompt for the smaller model
- [ ] Expand the test suite
- [ ] Add a latency check against the SRS <10s target
- [ ] Add request logging

**Done when:** the whole thing answers correctly fully-local, and tests + a latency number pass.

## Phase 7 — Deployment (final)
Goal: an on-prem bundle (flagship story) + a public cloud demo (resume URL).

### Track A — On-prem bundle (government story)
- [ ] Containerize the backend (Dockerfile)
- [ ] `docker-compose` with backend + Ollama (model pre-pulled) + frontend
- [ ] Volumes for: FAISS index, model cache, SQLite DB, Ollama models
- [ ] One-command, air-gapped bring-up (`docker compose up`, `LLM_PROVIDER=ollama`)
- [ ] Reverse proxy (Caddy/Nginx) for HTTPS; health checks; restart policies

### Track B — Public cloud demo (resume URL)
- [ ] Backend on a container host (Render / Railway / GCP Cloud Run)
- [ ] Frontend on Vercel via `NEXT_PUBLIC_API_URL`
- [ ] Cloud demo uses Groq; on-prem uses Ollama (config change, not code)
- [ ] Bake models into the image or mount a volume (avoid re-downloading 4.5 GB on cold start)

**Done when:** `docker compose up` runs the full stack offline, AND there's a live URL for the resume.
