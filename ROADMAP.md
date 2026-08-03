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

- [ ] Ingest the 16 `Original Maha-GR` PDFs (merge with, or replace, the current corpus)
- [ ] Verify the OCR fallback on a genuinely scanned GR (not just born-digital)
- [ ] Expand the gold set from 6 → ~30–40 questions (EN + Marathi, table-number, out-of-corpus)
- [ ] Re-run the calibration harness and re-set text/table/rerank thresholds
- [ ] Refine reference parsing so the supersede graph is cleaner (less noise)

**Done when:** index built from your GRs; strong hit@k on the real gold set; OCR confirmed on a real scan.

## Phase 2 — Officer-assistance completeness
Goal: every SRS officer-help feature works and is exposed in the UI.

- [ ] Add conflicting-document detection (flag + cite both when two GRs disagree)
- [ ] Ensure every feature (summarize/explain/compare/related/supersede) has an API route
- [ ] Ensure every feature has a matching frontend action (not backend-only)

**Done when:** all FR-3.5 features are clickable in the portal and grounded/cited.

## Phase 3 — Persistence & memory
Goal: conversations and feedback survive restarts.

- [ ] Add a local SQLite store: conversations, messages, feedback (thumbs + comment)
- [ ] Wire stored history into the query-rewrite step (follow-ups resolve across sessions)
- [ ] Add a conversation-history list in the portal

**Done when:** close the app, reopen, and past chats + feedback are still there.

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
