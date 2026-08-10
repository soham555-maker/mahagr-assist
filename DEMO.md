# MahaGR Assist — Demo script

A set of questions that are **verified**: each one retrieves the correct
Government Resolution and the correct number. Run `python
scripts/verify_demo.py` any time to re-check (no GROQ key needed for that).

## ⚠ First: which index are you demoing?

There are **two**, holding different documents. Pick deliberately — questions
3–5 below exist only in the flat one.

| | `VECTOR_BACKEND=hnsw` (the `.env` default) | `VECTOR_BACKEND=flat` |
|---|---|---|
| Contents | **99,410 real GRs across all 33 departments** | 196 real HTE GRs + **the 2 sample fee-table GRs** |
| Index | FAISS HNSW (~O(log n)) + SQLite | FAISS Flat (exact, O(n)), text in RAM |
| Demo it for | **the scale story**, department/date filters, cross-department retrieval | the fee-table exact-number and supersession moments (Q3–Q5) |

```bash
# the big corpus — what .env already points at
uvicorn app.api:app --port 8000

# the small fixture index, for one run only
VECTOR_BACKEND=flat MAHAGR_INDEX_DIR=index uvicorn app.api:app --port 8000
```

`curl -s localhost:8000/health` reports which one is loaded — check
`vector_backend`, `documents` and `indexed_vectors` before you present.

## Before the demo

1. Build the index and add the sample fee-table GRs (**flat index only**):
   ```
   cd backend && source .venv/bin/activate
   python scripts/ingest_text.py data/grs_text index   # real Marathi GRs (if not built)
   python scripts/add_fixtures.py                        # + the fee-table sample GRs
   ```
2. Put your key in `backend/.env`:  `GROQ_API_KEY=gsk_...`  (or use local Ollama)
3. Start backend:  `uvicorn app.api:app --port 8000`
4. Start frontend (new terminal):  `cd frontend && npm run dev`  → open http://localhost:3000

> Note on the data: the topical GRs are **real** (Higher & Technical Education,
> from orgpedia/mahGRs). The two **fee-table** GRs are realistic **sample**
> documents we made to show table + exact-number handling — say "sample GR"
> when you show those, not "real government order".

## The questions (ask these on the "Ask" page)

| # | Ask this (copy-paste) | Correct source GR | Correct answer | What it proves |
|---|---|---|---|---|
| 1 | `What procedure must a university follow to approve new colleges and courses?` | एनजीसी २०१७/(२२९/१७)/मशि-४ (2017-10-12) | The approval procedure (perspective plan, 5-year plan…) | **Cross-lingual**: English question → answer from a **Marathi** GR |
| 2 | `ग्रंथालय संचालनालयाच्या आधुनिकीकरणाबाबतचा निर्णय काय आहे?` | मराग्रं २५१७/प्र.क्र.१२१/२०१७/साशि ५ | The library-modernisation decision | **Marathi in → Marathi out**, Marathi text is read correctly |
| 3 | `What is the annual fee for OBC students for the first-year diploma in 2023-24?` | संकीर्ण-२०२३/प्र.क्र.४५/तांशि-४ | **₹6000** | **Table + exact number**, cross-lingual (EN question, Marathi table) |
| 4 | `२०२४-२५ या वर्षासाठी खुल्या प्रवर्गाचे सुधारित वार्षिक शुल्क किती आहे?` | संकीर्ण-२०२४/प्र.क्र.१२/तांशि-४ | **₹15000** | Marathi + table + exact number |
| 5 | `सुधारित शुल्करचनेनुसार इतर मागासवर्ग (OBC) प्रवर्गाचे वार्षिक शुल्क किती आहे?` | संकीर्ण-२०२४/प्र.क्र.१२/तांशि-४ | **₹7500** | Number from the revised table (sets up the compare below) |
| 6 | `What is the annual fee for a PhD in aerospace engineering at IIT Bombay?` | — (not in the documents) | *"insufficient information"* | **Groundedness**: it refuses to guess |

Every answer on screen shows its **source GR + date** — point at it and say
"you can check this."

## Two extra moments (on the "Browse" page)

- **Compare**: pick the 2023 and 2024 fee GRs → shows OBC **6000 → 7500** and
  Open **12000 → 15000**, and that 2024 replaces 2023. (Great "numbers" moment.)
- **Supersede**: open the 2024 fee GR → it shows it **supersedes** the 2023 one.

## The scale moments (on the big `hnsw` corpus)

This is what answers the feedback *"the whole corpus must be searchable — that's
the point of RAG."* Run these on `VECTOR_BACKEND=hnsw`.

- **Say the number, don't just claim it.** The Ask page's opening line reads
  *"Searching 99,410 Government Resolutions across 33 departments"* — it comes
  from `/corpus/stats`, i.e. from the database, not from a slide.
- **Cross-department retrieval.** Ask a scholarship question
  (`अनुसूचित जमातीच्या विद्यार्थ्यांना शिष्यवृत्ती`) and point out that the cited GR
  came from **Tribal Development**, not Higher & Technical Education — one query
  reached across departments that are separate portals in real life.
- **Filters, and why they're honest.** Open **Scope**, tick one department, ask
  again → the answer changes and the reply carries a *"Searched only: …"* line.
  The point to make: the filter is pushed **into** the vector search
  (FAISS `IDSelector`), not applied to the results afterwards — a post-filter
  would quietly return fewer/no results with no way to tell why.
- **Browse at scale.** The list is server-paginated with department chips; type
  a GR number and it searches all 99,410 in the database, not a copy in the
  browser.
- **The architecture line worth saying out loud:** *"FAISS holds only vectors
  and returns integer positions; SQLite turns those positions into text. So the
  corpus can grow without the server's memory growing."*

## Suggested order to present

1. Corpus-size line + a cross-department question (**scale**) → 2. Q1
(cross-lingual "wow") → 3. Scope filter (**control**) → 4. Q6 (refuses to guess
— the trust moment) → 5. switch to `flat` for Q3 + Compare 2023 vs 2024
(6000→7500) + Supersede (**numbers & conflicts**).

Repeat the three words: **Grounded. Multilingual. Explainable.**
And for this round, a fourth: **at corpus scale.**
