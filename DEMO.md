# MahaGR Assist — Demo script

A set of questions that are **verified**: each one retrieves the correct
Government Resolution and the correct number. Run `python
scripts/verify_demo.py` any time to re-check (no GROQ key needed for that).

## Before the demo

1. Build the index and add the sample fee-table GRs:
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

## Suggested order to present

1. Q1 (cross-lingual "wow") → 2. Q3 (table + exact 6000) → 3. Compare 2023 vs
2024 (6000→7500) → 4. Q6 (refuses to guess — the trust moment) → 5. Supersede.

Repeat the three words: **Grounded. Multilingual. Explainable.**
