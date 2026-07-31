# Synthetic GR fixtures

Two **synthetic** Marathi Government Resolutions for smoke-testing the pipeline
end to end without real GR data. They are fabricated for testing — not real
Government documents.

| File | What it is |
|---|---|
| `GR-2023-fees.pdf` | Diploma fee-structure GR (2023-24) with a category→fee table |
| `GR-2024-fees-revised.pdf` | Revised-fee GR (2024-25) that **supersedes** the 2023 one and cites it by number |

They are deliberately **linked** (2024 supersedes 2023 with different fees), so
they exercise not just retrieval but the cross-reference / supersede / compare
demos too.

Regenerate them with `python scripts/make_fixtures.py` (needs `reportlab` +
a Noto Sans Devanagari font installed). The `.html` sources are gone — the
reportlab generator is the source of truth, because it guarantees a clean
ToUnicode text layer and ruled table lines. Note: reportlab doesn't do complex
Devanagari shaping, but Noto renders it well enough to read; what matters is
that the extracted **text** is correct Unicode, which it is.

## Smoke test

```bash
# from backend/, with deps installed and GROQ_API_KEY in .env
python scripts/ingest_grs.py data/fixtures index      # build the index from the fixtures
python scripts/ask.py "What is the OBC diploma fee in 2023?"
python scripts/ask.py "२०२४ मध्ये खुल्या प्रवर्गाचे शुल्क किती आहे?"   # Marathi query
python scripts/ask.py "Which GR supersedes the 2023 fee resolution?"
python scripts/ask.py "Compare the 2023 and 2024 fees for OBC."
python scripts/ask.py "What is the fee for the PhD program?"          # not in corpus → should abstain
```

Expected: grounded answers with `[n]` citations to the fixture GRs; the OBC 2023
fee is **6000** and 2024 is **7500**; the last query should return "insufficient
information" rather than a guess.

## Extraction is verified

Both fixtures pass an extraction check (PyMuPDF prose + pdfplumber table):
Marathi + English text and the fee numbers (`6000`, `12000`, `7500`, `15000`)
extract intact, and `engine.table_extract` recovers the fee rows as sentences.
