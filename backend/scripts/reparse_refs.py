"""
reparse_refs.py — re-parse every ingested GR's number and reference list from
the text already stored in the corpus, without re-ingesting anything.

WHY THIS SCRIPT EXISTS
----------------------
The first reference parser matched a slash-bearing token bounded by WHITESPACE,
but a real GR number CONTAINS spaces ('एनजीसी-२०१०/(१९३/१०) /मशि-४'). Every
reference therefore came out truncated ('२०११/प्रक्र', '१३६/विशि-३') and matched
nothing: the knowledge graph resolved 2% of its edges.

Fixing the parser is only half the job — the corpus still holds the old, broken
`refs`. The cheap part is that `gr_documents.text` holds the FULL text of all
18,078 documents, so this is a re-PARSE, not a re-ingest: no download, no
chunking, no embedding, no GPU. Minutes instead of ~25 GPU-minutes.

WHAT IT DOES AND DOES NOT TOUCH
-------------------------------
Updates:  gr_number, gr_number_canon, refs, supersedes  — all derived purely
          from the document text, so re-running is idempotent.
Leaves:   date, department, title, chunks, vectors. `date` in particular has an
          ingestion-time fallback to the order id (the portal's own publication
          date, more reliable than an OCR'd header line) and `department` comes
          from the corpus folder, not the text — re-deriving either from text
          alone would be a downgrade, and `date` is what the graph uses to tell
          two same-numbered GRs apart.

Run `scripts/build_graph.py` afterwards: the edges are built from `refs`.

Usage:
    python scripts/reparse_refs.py --dry-run     # report only, change nothing
    python scripts/reparse_refs.py
    python scripts/reparse_refs.py --index /mnt/win/mahagr/index
"""

import argparse
import json
import os
import statistics
import sys

from engine import config, corpus_db, gr_metadata

BATCH = 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=None, help="index dir (default config.INDEX_DIR)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change, write nothing")
    ap.add_argument("--samples", type=int, default=3,
                    help="how many before/after examples to print")
    args = ap.parse_args()

    db_path = os.path.join(args.index or config.INDEX_DIR, "corpus.db")
    if not os.path.exists(db_path):
        print(f"No corpus at {db_path}. Run scripts/ingest_corpus.py first.")
        return 1

    corpus_db.init(db_path)
    with corpus_db.connect(db_path) as conn:
        ids = [r[0] for r in conn.execute("SELECT id FROM gr_documents ORDER BY id")]
        print(f"corpus  {db_path}  ({len(ids)} documents)"
              + ("   [DRY RUN]" if args.dry_run else ""))

        changed_number = changed_refs = 0
        old_ref_total = new_ref_total = 0
        old_lens, new_lens = [], []
        with_dates = 0
        samples = []

        for start in range(0, len(ids), BATCH):
            chunk = ids[start:start + BATCH]
            marks = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT id, gr_number, refs, text FROM gr_documents "
                f"WHERE id IN ({marks})", chunk).fetchall()
            updates = []
            for r in rows:
                meta = gr_metadata.extract(r["text"] or "")
                number = meta.get("gr_number")
                entries = meta.get("reference_details") or []

                old_refs = [e["number"] for e in corpus_db.reference_entries(r["refs"])]
                new_refs = [e["number"] for e in entries]
                old_ref_total += len(old_refs)
                new_ref_total += len(new_refs)
                old_lens += [len(gr_metadata.canonical_number(x) or "") for x in old_refs]
                new_lens += [len(gr_metadata.canonical_number(x) or "") for x in new_refs]
                with_dates += sum(1 for e in entries if e["date"])

                if number != r["gr_number"]:
                    changed_number += 1
                if new_refs != old_refs:
                    changed_refs += 1
                    if len(samples) < args.samples and old_refs and new_refs:
                        samples.append((r["id"], old_refs, new_refs, entries))

                updates.append((
                    number, gr_metadata.canonical_number(number),
                    json.dumps(entries, ensure_ascii=False),
                    1 if meta.get("supersedes") else 0,
                    r["id"]))

            if not args.dry_run:
                conn.executemany(
                    "UPDATE gr_documents SET gr_number=?, gr_number_canon=?, "
                    "refs=?, supersedes=? WHERE id=?", updates)
                conn.commit()
            done = start + len(chunk)
            print(f"\r  {done}/{len(ids)} documents", end="", flush=True)

        print("\n" + "=" * 60)
        print(f"gr_number changed on {changed_number} documents")
        print(f"reference list changed on {changed_refs} documents")
        print(f"references: {old_ref_total} -> {new_ref_total}")
        if new_ref_total:
            print(f"  with a parsable cited date: {with_dates} "
                  f"({100.0 * with_dates / new_ref_total:.0f}%)")
        # Length is the sharpest proxy for truncation: a document's OWN
        # canonical number is a median of 28 characters, so a reference that is
        # much shorter is a fragment that can never match anything.
        own = [len(r[0]) for r in conn.execute(
            "SELECT gr_number_canon FROM gr_documents WHERE gr_number_canon IS NOT NULL")]
        if old_lens and new_lens and own:
            print(f"median canonical length — own numbers {statistics.median(own):.0f}, "
                  f"references {statistics.median(old_lens):.0f} -> "
                  f"{statistics.median(new_lens):.0f}")

        for doc_id, old_refs, new_refs, entries in samples:
            print("-" * 60)
            print(f"{doc_id}")
            print(f"  before: {old_refs}")
            print(f"  after : {[(e['number'], e['date']) for e in entries]}")

        if args.dry_run:
            print("\nDRY RUN — nothing written. Re-run without --dry-run, then "
                  "python scripts/build_graph.py")
        else:
            print("\nNow rebuild the graph:  python scripts/build_graph.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
