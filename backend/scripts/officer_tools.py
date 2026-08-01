"""
officer_tools.py — CLI for the Government-officer assistance features (FR 3.5).

    python scripts/officer_tools.py summarize 201710121514029708
    python scripts/officer_tools.py explain "नवीन महाविद्यालय मंजुरीची प्रक्रिया काय आहे?"
    python scripts/officer_tools.py compare 201710121514029708 201711061646497708
    python scripts/officer_tools.py supersede संकीर्ण-२०२३        # metadata only, no LLM/key
    python scripts/officer_tools.py related 201710121514029708    # similarity, no LLM/key

Add --lang en|mr to force the answer language (default: match the document).
LLM features (summarize/explain/compare) need GROQ_API_KEY; supersede/related don't.
"""

import argparse

from engine import officer


def _print_answer(res):
    print("\n" + res["answer"])
    if res.get("phantom_citations"):
        print(f"\n  !! phantom citations {res['phantom_citations']} — groundedness alarm")
    if res["sources"]:
        print("\n--- sources " + "-" * 50)
        for s in res["sources"]:
            doc = s.get("gr_number") or s.get("source_file") or s["paper_id"]
            date = f", {s['date']}" if s.get("date") else ""
            print(f"  [{s['n']}] {s['title'][:56]}\n       {doc}{date} ({s['pages']})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("summarize", "supersede", "related"):
        p = sub.add_parser(name); p.add_argument("doc_id")
    for name in ("explain",):
        p = sub.add_parser(name); p.add_argument("question")
    p = sub.add_parser("compare"); p.add_argument("doc_a"); p.add_argument("doc_b")
    for name in ("summarize", "explain", "compare"):
        sub.choices[name].add_argument("--lang", default="auto")
    sub.choices["related"].add_argument("--k", type=int, default=5)
    ap.add_argument("--index", default="index")
    args = ap.parse_args()

    # supersede is pure metadata: load only the store (no model, no GROQ key).
    if args.cmd == "supersede":
        from engine.vector_store import FaissStore
        store = FaissStore.load(args.index)
        info = officer.supersession(store, args.doc_id)
        if not info["found"]:
            print(f"No GR matching '{args.doc_id}' in the index."); return
        print(f"\nGR {info['gr_number']}  (declares supersession: {info['declares_supersession']})")
        print("  Cites:")
        for c in info["cites"] or [{"gr_number": "(none)", "in_corpus": None}]:
            tag = f"  [in corpus: {c['in_corpus']}]" if c["in_corpus"] else "  [not in corpus]"
            print(f"    - {c['gr_number']}{tag}")
        print("  Superseded by later GRs:")
        for s in info["superseded_by"] or []:
            print(f"    - {s['gr_number']} ({s['date']})  {s['title']}")
        if not info["superseded_by"]:
            print("    (none found in this corpus)")
        return

    # related needs the embedding model but no LLM/key.
    if args.cmd == "related":
        from engine.retrieval import load_default_retriever
        print("Loading index + embedding model...")
        retriever = load_default_retriever(index_dir=args.index)
        rel = officer.related(retriever, args.doc_id, k=args.k)
        print(f"\nGRs related to '{args.doc_id}':")
        for r in rel or []:
            print(f"  {r['score']:.3f}  {r['gr_number']} ({r['date']})  {r['title']}")
        if not rel:
            print("  (none — is the doc id in the index?)")
        return

    # LLM features: retriever + reranker + GROQ.
    from engine.reranker import Reranker
    from engine.retrieval import load_default_retriever
    print("Loading index + models...")
    retriever = load_default_retriever(index_dir=args.index, reranker=Reranker())

    if args.cmd == "summarize":
        _print_answer(officer.summarize(retriever, args.doc_id, language=args.lang))
    elif args.cmd == "explain":
        _print_answer(officer.explain(retriever, args.question, language=args.lang))
    elif args.cmd == "compare":
        _print_answer(officer.compare(retriever, args.doc_a, args.doc_b, language=args.lang))


if __name__ == "__main__":
    main()
