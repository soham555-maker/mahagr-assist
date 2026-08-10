"""
build_graph.py — build the supersede/citation knowledge graph (PLAN Phase 3).

Pure metadata work over the corpus SQLite: no model, no GPU, no re-embedding,
because ingestion already persisted every document's parsed `references` and its
`supersedes` flag. On the 18k-GR corpus this is seconds, not minutes, and it is
safe to re-run at any time (the edge table is rebuilt from scratch).

Run it AFTER ingestion, and again after any re-ingest.

Usage:
    python scripts/build_graph.py
    python scripts/build_graph.py --index /mnt/win/mahagr/index
    python scripts/build_graph.py --show 201710121514029708   # inspect one GR
"""

import argparse
import os
import sys

from engine import config, corpus_db, graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=None, help="index dir (default config.INDEX_DIR)")
    ap.add_argument("--show", help="print one GR's neighbourhood after building")
    ap.add_argument("--depth", type=int, default=1)
    args = ap.parse_args()

    index_dir = args.index or config.INDEX_DIR
    db_path = os.path.join(index_dir, "corpus.db")
    if not os.path.exists(db_path):
        print(f"No corpus at {db_path}. Run scripts/ingest_corpus.py first.")
        return 1

    corpus_db.init(db_path)          # applies the gr_edges + canonical-number migration
    with corpus_db.connect(db_path) as conn:
        docs = conn.execute("SELECT COUNT(*) FROM gr_documents").fetchone()[0]
        print(f"corpus  {db_path}  ({docs} documents)")
        print("building edges ...")
        counts = graph.build_edges(conn, progress=lambda m: print(f"  {m}"))

        print("\n" + "=" * 60)
        print(f"edges        {counts['edges']}")
        print(f"  resolved   {counts['resolved']}   (both documents are in the corpus)"
              f"  — {counts['resolved_by_date']} of them disambiguated by the cited date")
        print(f"  dangling   {counts['dangling']}   (cites a GR we do not hold)")
        print(f"  ambiguous  {counts['ambiguous']}   (canonical-number collision)")
        print(f"  of which 'supersedes' edges: {counts['supersedes']}")
        if counts["edges"]:
            pct = 100.0 * counts["resolved"] / counts["edges"]
            print(f"\nResolution rate {pct:.1f}%.")
            print("""
  READ THIS BEFORE QUOTING THAT NUMBER. It is a measure of CORPUS COVERAGE, not
  of parser quality, and 100% is not the target:

    * We hold 6 of Maharashtra's ~33 departments. A Higher-Education GR
      routinely cites Finance / GAD / Revenue orders we never ingested.
      Measured on this corpus: only ~21% of unresolved references even name a
      department we ingested; the rest name another one or name none at all.
    * Unresolved references are KEPT, as `dangling`, with the date they were
      cited on. "This order builds on one we do not hold" is information an
      officer needs — dropping them would make the graph look more complete
      than the corpus actually is.
    * `ambiguous` means several documents share that canonical number and the
      cited date did not single one out. Checked: for the large majority the
      nearest candidate date is more than a month away, i.e. they are genuinely
      different orders in the same number series. Guessing one would put the
      wrong GR in front of an officer, so the edge stays unresolved.

  History: this rate was 2.0% when references were matched as whitespace-bounded
  tokens, which truncated every multi-word GR number into a fragment. Prefix
  matching those fragments was tried and REJECTED (362 recoveries vs 864 new
  ambiguities). Fixing the extraction instead — see scripts/reparse_refs.py —
  is what took it to ~9.5%.""")

        s = graph.stats(conn)
        print(f"documents with at least one resolved edge: "
              f"{s['documents_with_resolved_edges']}")

        if args.show:
            n = graph.neighbourhood(conn, args.show, depth=args.depth)
            print("\n" + "=" * 60)
            if not n["found"]:
                print(f"No GR matching '{args.show}'.")
                return 0
            r = n["root"]
            print(f"{r['id']}  {r['gr_number']}  ({r['date']})\n  {r['title']}")
            print(f"\n  neighbourhood: {len(n['nodes'])} nodes, {len(n['edges'])} edges")
            for e in n["edges"]:
                print(f"    {e['src']} --{e['kind']}--> {e['dst']}")
            if n["chain"]:
                print("\n  SUPERSEDE CHAIN (this GR may no longer be in force):")
                for i, c in enumerate(n["chain"], 1):
                    print(f"    {i}. {c['gr_number']} ({c['date']})  {c['title'][:50]}")
            if n["dangling"]:
                print(f"\n  references we do not hold ({len(n['dangling'])}):")
                for d in n["dangling"][:10]:
                    print(f"    - {d['gr_number']}  [{d['resolution']}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
