"""
graph.py — the supersede / citation knowledge graph over the GR corpus
(PLAN Phase 3).

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
It is a DOMAIN knowledge graph: nodes are Government Resolutions, edges are the
relations a GR states about other GRs — `supersedes` (this order replaces that
one) and `cites` (this order refers to that one). The relations are not
inferred; they are read from the "वाचा / संदर्भ" reference block that
`gr_metadata.extract()` already parses, so every edge is traceable to text in
the document.

It is NOT Graph RAG. Graph RAG has an LLM extract entities and relations from
the whole corpus and then *retrieves by traversing* that graph. This graph is
built by deterministic parsing, is used for provenance and conflict warnings,
and sits beside vector retrieval rather than replacing it. Worth being precise
about in a viva: the two get conflated constantly.

THE REAL PROBLEM IS REFERENCE RESOLUTION, NOT TRAVERSAL
-------------------------------------------------------
Traversing a few thousand edges is trivial. The hard part is deciding whether
'संकीर्ण-२०२३/प्र.क्र.४५/तांशि-४' printed in one OCR'd document is the same order
as a `gr_number` parsed out of another OCR'd document. They differ by spacing,
by Devanagari vs ASCII digits, and by punctuation the scanner dropped. So
matching happens on `gr_metadata.canonical_number()`, and every edge records
how it resolved:

    resolved   matched exactly one document in the corpus
    dangling   a real reference to an order we do not hold
    ambiguous  matched several documents (a canonical-form collision)

Dangling edges are KEPT. "This GR builds on an order that is not in our corpus"
is something an officer must be told, not something to quietly drop — and the
dangling rate is also the honest measure of how complete the corpus is.

CYCLES ARE REAL, AND WILL HANG YOU
----------------------------------
Two GRs amending each other, or an OCR collision that makes A resolve to B and
B to A, produce a cycle. Every traversal here carries a `seen` set and a depth
cap. Without them `/graph` is an infinite loop behind an HTTP request.
"""

from collections import deque

from engine import corpus_db, gr_metadata

SUPERSEDES = "supersedes"
CITES = "cites"


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #

def build_edges(conn, progress=None):
    """(Re)build gr_edges from what is already stored in gr_documents.

    Pure metadata work — no model, no GPU, no re-embedding — because Phase 2
    already persisted each document's parsed `refs` and `supersedes` flag.
    Idempotent: the table is rebuilt from scratch, so running it twice gives the
    same graph rather than doubled edges.

    Returns a counts dict: {edges, resolved, resolved_by_date, dangling,
    ambiguous, supersedes}.
    """
    backfill = corpus_db.backfill_canonical_numbers(conn)
    if progress and backfill:
        progress(f"backfilled {backfill} canonical GR numbers")

    # canonical number -> [doc_id, ...]. Built once; the alternative is a LIKE
    # scan over 18,000 rows for every reference, which is O(docs x refs).
    # `dates` rides along because 2,138 canonical numbers in this corpus are
    # held by more than one document, and the cited date is what separates them.
    lookup, dates = {}, {}
    for row in conn.execute("SELECT id, gr_number_canon, date FROM gr_documents"):
        dates[row["id"]] = row["date"]
        if row["gr_number_canon"]:
            lookup.setdefault(row["gr_number_canon"], []).append(row["id"])

    conn.execute("DELETE FROM gr_edges")
    counts = {"edges": 0, "resolved": 0, "resolved_by_date": 0, "dangling": 0,
              "ambiguous": 0, "supersedes": 0}
    batch = []
    columns = "(src_id, dst_id, dst_number, dst_canon, kind, resolution, dst_date)"

    for row in conn.execute("SELECT id, refs, supersedes FROM gr_documents"):
        refs = corpus_db.reference_entries(row["refs"])
        # A GR that cites others AND declares a supersession supersedes what it
        # cites. The text says WHICH one far too inconsistently to parse, so the
        # honest modelling is: every reference of a superseding GR is a
        # candidate `supersedes` edge. Over-reporting here is safe because the
        # UI phrases it as "may be superseded — verify", which is what an
        # officer needs; silently missing one is not.
        kind = SUPERSEDES if row["supersedes"] else CITES
        seen_refs = set()
        for ref in refs:
            canon = gr_metadata.canonical_number(ref["number"])
            if not canon or canon in seen_refs:
                continue
            seen_refs.add(canon)
            found = lookup.get(canon, [])
            matches = [m for m in found if m != row["id"]]
            if found and not matches:
                # The reference resolves to this document itself. That is not an
                # edge AND not a dangling reference — recording it as dangling
                # would inflate the "we don't hold this order" count with
                # self-references. gr_metadata drops the obvious ones while
                # parsing; this catches the OCR variants that only match once
                # canonicalised.
                continue
            resolution = "dangling"
            dst = None
            if len(matches) == 1:
                dst, resolution = matches[0], "resolved"
            elif len(matches) > 1:
                # Same number, several documents. Only the cited DATE can say
                # which one is meant. If it names exactly one, that is a real
                # resolution; if it names none or several, the edge stays
                # `ambiguous` — an unresolved edge is honest, a guessed one
                # would put the wrong GR in front of an officer.
                same_date = [m for m in matches if ref["date"] and dates.get(m) == ref["date"]]
                if len(same_date) == 1:
                    dst, resolution = same_date[0], "resolved"
                    counts["resolved_by_date"] += 1
                else:
                    resolution = "ambiguous"
            batch.append((row["id"], dst, str(ref["number"]), canon, kind,
                          resolution, ref["date"]))
            counts["edges"] += 1
            counts[resolution] += 1
            if kind == SUPERSEDES:
                counts["supersedes"] += 1
            if len(batch) >= 5000:
                conn.executemany(
                    f"INSERT OR REPLACE INTO gr_edges {columns} VALUES (?,?,?,?,?,?,?)",
                    batch)
                batch = []

    if batch:
        conn.executemany(
            f"INSERT OR REPLACE INTO gr_edges {columns} VALUES (?,?,?,?,?,?,?)", batch)
    conn.commit()
    return counts


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #

def _node(conn, doc_id):
    row = conn.execute(
        "SELECT id, gr_number, date, title, department FROM gr_documents WHERE id=?",
        (doc_id,)).fetchone()
    if not row:
        return None
    return {"id": row["id"], "gr_number": row["gr_number"], "date": row["date"],
            "title": (row["title"] or "")[:120], "department": row["department"]}


def outgoing(conn, doc_id):
    """What this GR references (resolved, dangling and ambiguous alike)."""
    return [dict(r) for r in conn.execute(
        "SELECT src_id, dst_id, dst_number, dst_date, kind, resolution FROM gr_edges "
        "WHERE src_id=?", (doc_id,))]


def incoming(conn, doc_id):
    """Which GRs reference this one. This is the direction that answers
    'has this order been superseded?', which is why dst_id is indexed."""
    return [dict(r) for r in conn.execute(
        "SELECT src_id, dst_id, dst_number, dst_date, kind, resolution FROM gr_edges "
        "WHERE dst_id=?", (doc_id,))]


def supersede_chain(conn, doc_id, max_depth=10):
    """The transitive chain of GRs that supersede this one, newest link last.

    Walks `incoming` supersedes-edges repeatedly: if B supersedes A and C
    supersedes B, then asking about A returns [B, C]. That is the answer an
    officer actually needs — being told A was replaced by B is misleading when B
    has itself been replaced.

    CYCLE-SAFE by construction: `seen` stops A->B->A dead, and max_depth bounds
    even a pathological graph. Both matter — this runs behind an HTTP request.
    """
    chain, seen, current, depth = [], {doc_id}, doc_id, 0
    while depth < max_depth:
        nxt = [e["src_id"] for e in incoming(conn, current)
               if e["kind"] == SUPERSEDES and e["src_id"] not in seen]
        if not nxt:
            break
        # Deterministic when several GRs claim to supersede the same order:
        # take the newest, since that is the one in force.
        rows = [_node(conn, i) for i in nxt]
        rows = [r for r in rows if r]
        if not rows:
            break
        rows.sort(key=lambda r: (r["date"] or "", r["id"]), reverse=True)
        nextnode = rows[0]
        chain.append(nextnode)
        seen.add(nextnode["id"])
        current = nextnode["id"]
        depth += 1
    return chain


def neighbourhood(conn, doc_id, depth=1, max_nodes=60):
    """The graph around one GR, out to `depth` hops in BOTH directions.

    Returns {found, root, nodes, edges, chain, dangling}. `nodes` are corpus
    documents; `dangling` are references we could not resolve, returned
    separately so the UI can render them as ghost nodes — visible gaps rather
    than an invisibly incomplete graph.

    `max_nodes` is a hard cap: a well-connected GR in an 18k corpus can pull in
    hundreds of neighbours at depth 2, which is unreadable as a diagram and slow
    to serialize. BFS order means the cap keeps the CLOSEST neighbours.
    """
    root = _node(conn, doc_id)
    if root is None:
        return {"found": False, "doc_id": doc_id, "nodes": [], "edges": [],
                "chain": [], "dangling": []}

    nodes = {doc_id: root}
    edges, dangling = [], []
    seen_edges = set()
    queue = deque([(doc_id, 0)])
    visited = {doc_id}

    while queue:
        current, d = queue.popleft()
        if d >= depth:
            continue
        for e in outgoing(conn, current) + incoming(conn, current):
            key = (e["src_id"], e["dst_id"], e["dst_number"], e["kind"])
            if key in seen_edges:
                continue
            seen_edges.add(key)

            if e["resolution"] != "resolved" or not e["dst_id"]:
                # Only surface unresolved references FROM a node we're showing;
                # otherwise every ghost in the corpus would follow an edge home.
                if e["src_id"] == current:
                    dangling.append({"src_id": e["src_id"],
                                     "gr_number": e["dst_number"],
                                     "date": e["dst_date"],
                                     "kind": e["kind"],
                                     "resolution": e["resolution"]})
                continue

            edges.append({"src": e["src_id"], "dst": e["dst_id"], "kind": e["kind"]})
            for other in (e["src_id"], e["dst_id"]):
                if other in visited or len(nodes) >= max_nodes:
                    continue
                node = _node(conn, other)
                if node:
                    nodes[other] = node
                    visited.add(other)
                    queue.append((other, d + 1))

    return {
        "found": True,
        "doc_id": doc_id,
        "root": root,
        "nodes": list(nodes.values()),
        # Drop edges whose endpoints got cut by max_nodes, so the client never
        # has to render an edge pointing at a node it was not given.
        "edges": [e for e in edges if e["src"] in nodes and e["dst"] in nodes],
        "chain": supersede_chain(conn, doc_id),
        "dangling": dangling,
    }


def stats(conn):
    """Graph shape — how many edges, and how well references resolved. The
    resolution breakdown is the honest measure of corpus completeness."""
    total = conn.execute("SELECT COUNT(*) FROM gr_edges").fetchone()[0]
    by_res = {r[0]: r[1] for r in conn.execute(
        "SELECT resolution, COUNT(*) FROM gr_edges GROUP BY resolution")}
    by_kind = {r[0]: r[1] for r in conn.execute(
        "SELECT kind, COUNT(*) FROM gr_edges GROUP BY kind")}
    linked = conn.execute(
        "SELECT COUNT(DISTINCT src_id) FROM gr_edges WHERE dst_id IS NOT NULL").fetchone()[0]
    return {"edges": total, "by_resolution": by_res, "by_kind": by_kind,
            "documents_with_resolved_edges": linked}
