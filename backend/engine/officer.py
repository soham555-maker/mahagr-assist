"""
officer.py — the Government-officer assistance features (SRS FR 3.5), built by
composing the pieces that already exist: retrieval, GR metadata, and rag.py's
grounded-generation edge. Nothing here re-implements RAG; each feature is a
prompt + the right slice of context.

Five features:
  * summarize(doc_id)        FR 3.5.2  — plain summary of one GR
  * explain(question)        FR 3.5.1  — answer in simple language
  * compare(id_a, id_b)      FR 3.5.3  — highlight differences between two GRs
  * supersession(doc_id)     FR 3.5.5  — which GRs this replaces / is replaced by
  * related(doc_id)          FR 3.5.4  — recommend similar GRs

DESIGN: summarize/explain/compare are LLM features and reuse rag.py's PURE
helpers (format_block, trim_to_budget, parse/resolve_citations) so they stay
grounded and cited exactly like a normal answer. supersession and related are
NOT LLM features — supersession is a pure metadata-graph lookup over the GR
numbers gr_metadata already parsed, and related is a vector-similarity lookup —
so they are deterministic and free.
"""

from engine import rag


# --------------------------------------------------------------------------- #
# document helpers — a "document" is one GR spread across several chunks
# --------------------------------------------------------------------------- #

def _doc_key(m):
    """Stable identity for the GR a chunk belongs to."""
    return m.get("order_id") or m.get("gr_number") or m.get("source_file")


def _hay(m):
    return " ".join(str(m.get(k, "")) for k in ("order_id", "gr_number", "source_file", "title"))


def _corpus_db(store):
    """The SQLite path if `store` is the scaled HNSW backend, else None.

    THE ONE DISPATCH POINT. Every function below works on both backends, and
    this is the only place that asks which one it has. The flat FaissStore keeps
    all chunk text and metadata in Python lists, so those paths scan in memory;
    HnswStore holds only vectors, so its paths query SQLite instead. The
    difference is not cosmetic — at 18,000 documents the in-memory scans below
    run per call, and supersede_warnings calls one of them once per cited GR.
    """
    return getattr(store, "corpus_db_path", None)


def _graph_built(conn):
    """Has scripts/build_graph.py been run against this corpus?

    Checked because the supersede features must degrade to the older on-the-fly
    scan rather than silently return nothing on a corpus that has been ingested
    but whose graph has not been built yet. `LIMIT 1` so this is O(1), not a
    count over every edge.
    """
    try:
        return conn.execute("SELECT 1 FROM gr_edges LIMIT 1").fetchone() is not None
    except Exception:      # table not created yet (an older corpus.db)
        return False


def document_chunks(store, doc_id):
    """All chunks of the GR identified by doc_id (matched as a substring of its
    order_id / gr_number / source_file), in reading order."""
    db_path = _corpus_db(store)
    if db_path:
        from engine import corpus_db
        with corpus_db.connect(db_path, readonly=True) as conn:
            return corpus_db.document_chunks(conn, doc_id)

    hits = [{"text": t, "metadata": m, "index": i, "score": 1.0}
            for i, (t, m) in enumerate(zip(store.texts, store.metadata))
            if doc_id in _hay(m)]
    hits.sort(key=lambda h: h["metadata"].get("chunk_index", 0))
    return hits


def list_documents(store, limit=None, offset=0, q=None, departments=None):
    """{doc_key: representative_metadata} for every distinct GR in the index.

    limit/offset/q/departments apply to the corpus backend only and exist
    because "every distinct GR" is 18,000 rows there — the /documents endpoint
    has to paginate rather than serialize the whole corpus into one response.
    """
    db_path = _corpus_db(store)
    if db_path:
        from engine import corpus_db
        with corpus_db.connect(db_path, readonly=True) as conn:
            rows, _ = corpus_db.list_documents(
                conn, q=q, departments=departments, limit=limit or 50, offset=offset)
        return {r["id"]: {"order_id": r["id"], "gr_number": r["gr_number"],
                          "date": r["date"], "department": r["department"],
                          "language": r["language"], "category": r["category"],
                          "title": r["title"]}
                for r in rows}

    docs = {}
    for m in store.metadata:
        k = _doc_key(m)
        if k and k not in docs:
            docs[k] = m
    return docs


# --------------------------------------------------------------------------- #
# LLM features — grounded + cited, via rag.py's pure helpers
# --------------------------------------------------------------------------- #

SUMMARIZE_SYSTEM = (
    "You summarize Maharashtra Government documents for an officer, using ONLY "
    "the numbered context blocks. Give the purpose, the key decisions, and any "
    "amounts/dates/deadlines exactly as written. Cite blocks like [1]. Do not "
    "add anything not in the context. Answer in the same language as the "
    "document unless told otherwise.")

EXPLAIN_SYSTEM = (
    "You explain Maharashtra Government documents in SIMPLE, plain language a "
    "non-specialist can follow, using ONLY the numbered context blocks. Keep "
    "official terms but gloss them in parentheses. Do not oversimplify away the "
    "actual rule. Cite blocks like [1]. If the context doesn't cover it, say so.")

COMPARE_SYSTEM = (
    "You compare two Maharashtra Government Resolutions for an officer, using "
    "ONLY the numbered context blocks (each block header names which GR it is "
    "from). List the concrete DIFFERENCES — scope, amounts, dates, who is "
    "affected — as short bullets, each citing the block(s) it rests on. If one "
    "GR supersedes/amends the other, say so. Do not invent differences that the "
    "context doesn't support.")


def _generate(system, chunks, task, client, config, language):
    """Shared grounded-generation tail for the LLM features: numbered blocks ->
    one call -> parsed, resolved citations. Same contract shape as rag.answer."""
    config = config or rag.GenerationConfig()
    if not chunks:
        return {"answer": "No matching document found in the index.",
                "sources": [], "phantom_citations": [], "chunks": []}
    used, _ = rag.trim_to_budget(chunks, config.context_token_budget,
                                 config.tokens_per_devanagari)
    blocks = "\n\n".join(rag.format_block(i, h) for i, h in enumerate(used, 1))
    lang = rag.language_directive(language)
    user = f"{task}{lang}\n\nContext:\n{blocks}"
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    resp = rag.call_llm(messages, config, client)
    text = resp.choices[0].message.content or ""
    valid, phantom = rag.parse_citations(text, len(used))
    return {"answer": text, "sources": rag.resolve_citations(valid, used),
            "phantom_citations": phantom, "chunks": used}


def summarize(retriever, doc_id, client=None, config=None, language="auto"):
    client = client or rag.make_client(config)
    chunks = document_chunks(retriever.store, doc_id)
    return _generate(SUMMARIZE_SYSTEM, chunks,
                     f"Summarize Government Resolution '{doc_id}'.",
                     client, config, language)


def explain(retriever, question, client=None, config=None, language="auto"):
    client = client or rag.make_client(config)
    result = retriever.retrieve(question)
    return _generate(EXPLAIN_SYSTEM, result["chunks"],
                     f"Explain in simple language: {question}",
                     client, config, language)


def compare(retriever, doc_id_a, doc_id_b, client=None, config=None, language="auto",
            per_side=6):
    client = client or rag.make_client(config)
    a = document_chunks(retriever.store, doc_id_a)[:per_side]
    b = document_chunks(retriever.store, doc_id_b)[:per_side]
    if not a or not b:
        missing = doc_id_a if not a else doc_id_b
        return {"answer": f"No document matching '{missing}' in the index.",
                "sources": [], "phantom_citations": [], "chunks": []}
    # Cap each side BEFORE trimming so the budget can't drop one GR entirely.
    return _generate(COMPARE_SYSTEM, a + b,
                     f"Compare GR '{doc_id_a}' and GR '{doc_id_b}'.",
                     client, config, language)


# --------------------------------------------------------------------------- #
# metadata / vector features — deterministic, no LLM
# --------------------------------------------------------------------------- #

def supersession(store, doc_id):
    """Which GRs this one supersedes/cites, and which later GRs supersede it —
    read straight from the parsed metadata graph (gr_metadata's references +
    supersedes flag). No LLM: this is a fact lookup, and it must be exact."""
    db_path = _corpus_db(store)
    if db_path:
        from engine import corpus_db, graph
        with corpus_db.connect(db_path, readonly=True) as conn:
            target = corpus_db.find_document(conn, doc_id)
            if target is None:
                return {"found": False, "doc_id": doc_id}
            own = target.get("gr_number") or ""

            # Prefer the PREBUILT graph (scripts/build_graph.py): its edges were
            # matched on canonical GR numbers, so it resolves the OCR spelling
            # variants that a raw string comparison misses, and it can give the
            # TRANSITIVE chain. Fall back to the on-the-fly scan when the graph
            # has not been built yet, so this endpoint never simply goes blank
            # on a freshly ingested corpus.
            if _graph_built(conn):
                out = graph.outgoing(conn, target["id"])
                chain = graph.supersede_chain(conn, target["id"])
                return {
                    "found": True,
                    "doc_id": doc_id,
                    "gr_number": own,
                    "declares_supersession": bool(target.get("supersedes")),
                    "cites": [{"gr_number": e["dst_number"],
                               "in_corpus": e["dst_id"]} for e in out],
                    # Direct supersessions, for backwards compatibility...
                    "superseded_by": [
                        {"doc": n["id"], "gr_number": n["gr_number"],
                         "date": n["date"], "title": n["title"][:80]}
                        for n in chain[:1]],
                    # ...plus the full chain, which is what an officer actually
                    # needs: "replaced by B" is misleading if B is also dead.
                    "supersede_chain": [
                        {"doc": n["id"], "gr_number": n["gr_number"],
                         "date": n["date"], "title": n["title"][:80]}
                        for n in chain],
                }

            cites = [e["number"]
                     for e in corpus_db.reference_entries(target.get("refs"))]
            present = corpus_db.documents_for_gr_numbers(conn, cites)
            return {
                "found": True,
                "doc_id": doc_id,
                "gr_number": own,
                "declares_supersession": bool(target.get("supersedes")),
                "cites": [{"gr_number": r, "in_corpus": present.get(r)} for r in cites],
                "superseded_by": corpus_db.superseding_documents(conn, own),
                "supersede_chain": [],
            }

    docs = list_documents(store)
    target = next((m for k, m in docs.items() if doc_id in _hay(m)), None)
    if target is None:
        return {"found": False, "doc_id": doc_id}

    own = target.get("gr_number") or ""
    cites = target.get("references", []) or []

    def _present(ref):
        # is a cited GR number actually in our corpus?
        return next((_doc_key(m) for m in docs.values()
                     if ref and ref in str(m.get("gr_number", ""))), None)

    superseded_by = [
        {"doc": _doc_key(m), "gr_number": m.get("gr_number"), "date": m.get("date"),
         "title": (m.get("title") or "")[:80]}
        for m in docs.values()
        if own and m.get("supersedes") and any(own in str(r) for r in (m.get("references") or []))
    ]
    return {
        "found": True,
        "doc_id": doc_id,
        "gr_number": own,
        "declares_supersession": bool(target.get("supersedes")),
        "cites": [{"gr_number": r, "in_corpus": _present(r)} for r in cites],
        "superseded_by": superseded_by,
    }


def supersede_warnings(store, sources):
    """Conflict/supersede check for an answer's cited GRs (FR: "highlight
    conflicting Government documents"). For each cited GR, look up the supersede
    graph; if a NEWER GR in the corpus supersedes it, return a warning so the
    officer isn't shown a cancelled order without notice. Deterministic (metadata
    only) — it complements the LLM prompt's own conflict flagging."""
    seen, warnings = set(), []
    for s in sources or []:
        gr = s.get("gr_number")
        if not gr or gr in seen:
            continue
        seen.add(gr)
        info = supersession(store, gr)
        if not info.get("found"):
            continue

        # Prefer the transitive chain when the graph is built. Telling an
        # officer "GR X was replaced by Y" is actively misleading when Y has
        # itself been replaced since — the last link is the order in force.
        chain = info.get("supersede_chain") or []
        if len(chain) > 1:
            last = chain[-1]
            warnings.append(
                f"GR {gr} may be superseded by GR {chain[0]['gr_number']} "
                f"({chain[0].get('date') or 'unknown date'}), which was itself "
                f"superseded — the latest appears to be GR {last['gr_number']} "
                f"({last.get('date') or 'unknown date'}). Verify before relying on it.")
            continue

        for sb in (chain or info.get("superseded_by") or []):
            date = sb.get("date") or "unknown date"
            warnings.append(
                f"GR {gr} may be superseded by GR {sb['gr_number']} ({date}) — verify before relying on it.")
    return warnings


def related(retriever, doc_id, k=5):
    """Recommend GRs similar to doc_id by embedding its title/opening and
    finding the nearest OTHER documents in the shared vector space."""
    chunks = document_chunks(retriever.store, doc_id)
    if not chunks:
        return []
    m0 = chunks[0]["metadata"]
    probe = (m0.get("title") or "") + " " + chunks[0]["text"][:400]
    vec = retriever.model.encode([probe], normalize_embeddings=True)[0]

    # HnswStore returns only ids + scores (no text lives in it), so the hits are
    # rehydrated from SQLite before the dedupe-by-document below — which is
    # otherwise identical for both backends.
    db_path = _corpus_db(retriever.store)
    if db_path:
        from engine import corpus_db
        raw = retriever.store.search(vec, k=k * 6)
        with corpus_db.connect(db_path, readonly=True) as conn:
            rows = corpus_db.chunks_by_faiss_ids(conn, [h["index"] for h in raw])
        hits = [{"score": h["score"], **rows[h["index"]]}
                for h in raw if h["index"] in rows]
    else:
        hits = retriever.store.search(vec, k=k * 6)

    best = {}  # doc_key -> best score
    for hit in hits:
        key = _doc_key(hit["metadata"])
        if key and doc_id not in _hay(hit["metadata"]):  # exclude self
            if key not in best or hit["score"] > best[key][0]:
                best[key] = (hit["score"], hit["metadata"])
    ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:k]
    return [{"doc": key, "score": round(sc, 3),
             "gr_number": m.get("gr_number"), "date": m.get("date"),
             "title": (m.get("title") or "")[:80]}
            for key, (sc, m) in ranked]
