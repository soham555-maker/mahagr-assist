"""
corpus_db.py — the document/metadata half of the scaled corpus (PLAN Phase 2).

WHY A SECOND SQLITE MODULE, SEPARATE FROM app/db.py
---------------------------------------------------
app/db.py is PORTAL state — conversations, messages, feedback. This is CORPUS
state — the GRs themselves and their chunks. They are split for two reasons that
are not stylistic:

  1. LAYERING. engine/retrieval.py has to read chunk text on every query. If that
     lived in app/db.py, the engine would import the app, inverting the
     dependency (the app is supposed to sit on top of the engine). Everything in
     engine/ must be runnable with no FastAPI in the picture — that is what lets
     scripts/ and the tests use it headless.
  2. DISK. Root '/' has ~3 GB free; the corpus is ~200 MB of text and grows.
     app/db.py's file is small and lives with the code; this one defaults to
     /mnt/win (HANDOFF §5). Two files also means a corpus rebuild can't corrupt
     the officers' conversation history.

WHAT THIS TABLE IS FOR — THE ACTUAL SCALE FIX
---------------------------------------------
The old FaissStore kept every chunk's TEXT in a Python list, loaded from a JSON
sidecar at startup. At 713 vectors that is 1 MB; at 65,000 it is ~200 MB of
Python strings held resident for the life of the process, plus a JSON parse on
every boot. FAISS only ever returns integer positions, so the text does not need
to be in RAM at all — it needs to be *addressable by that integer*. That is
exactly what a SQLite primary key is. So `gr_chunks.faiss_id` IS the position of
the vector in the FAISS index, and a search becomes: FAISS gives ids -> one
indexed SELECT gives the text. RAM stays flat as the corpus grows.

KEYWORD SEARCH AT SCALE (FTS5, not rank_bm25)
---------------------------------------------
The hybrid retriever needs BM25 alongside the dense vectors. rank_bm25 holds a
tokenized copy of the whole corpus in RAM (a dict per document) — fine for 713
chunks, several GB at 65k. SQLite's FTS5 does the same BM25 ranking off DISK, in
the database we already have. Verified on this machine that FTS5's `unicode61`
tokenizer handles Devanagari correctly — 'शिष्यवृत्ती' and 'निर्णय' match as whole
words and are NOT split at their vowel marks, which is the exact bug that made
Python's `\\w` unusable for Marathi (HANDOFF §5.4).

FTS5 QUERY SYNTAX IS A TRAP: characters like '-', '"', '*', ':' and 'OR' are
MATCH operators, so passing a raw user question straight to MATCH raises
"fts5: syntax error". Every query goes through hybrid.tokenize() first (the same
Devanagari-aware tokenizer the dense path uses) and is rejoined with explicit
ORs — see search_bm25.
"""

import json
import os
import sqlite3
from contextlib import contextmanager

from engine import hybrid

# The corpus DB lives beside the FAISS index on the big partition by default.
CORPUS_DB = os.environ.get("MAHAGR_CORPUS_DB", "/mnt/win/mahagr/index/corpus.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS gr_documents (
    id            TEXT PRIMARY KEY,   -- orgpedia order id; first 8 digits = YYYYMMDD
    gr_number     TEXT,
    department    TEXT,
    date          TEXT,               -- ISO YYYY-MM-DD
    category      TEXT,
    language      TEXT,               -- 'mr' | 'en'
    title         TEXT,
    source_file   TEXT,
    n_chunks      INTEGER,
    refs          TEXT,               -- JSON list of {number, date} (older
                                      -- corpora hold plain strings; see
                                      -- reference_entries())
    supersedes    INTEGER,            -- 1 if this GR replaces something it cites
    text          TEXT                -- full document text (for /documents/{id}/text)
);

-- faiss_id is NOT a surrogate key: it is the vector's position in the FAISS
-- index. INTEGER PRIMARY KEY makes it SQLite's rowid, so the lookup FAISS
-- forces us to do on every query is the fastest one the engine has.
CREATE TABLE IF NOT EXISTS gr_chunks (
    faiss_id     INTEGER PRIMARY KEY,
    gr_id        TEXT NOT NULL,
    chunk_index  INTEGER,
    page_start   INTEGER,
    page_end     INTEGER,
    content_type TEXT,
    text         TEXT
);

CREATE INDEX IF NOT EXISTS ix_chunks_gr    ON gr_chunks(gr_id);
CREATE INDEX IF NOT EXISTS ix_docs_dept    ON gr_documents(department, date);
CREATE INDEX IF NOT EXISTS ix_docs_date    ON gr_documents(date);
CREATE INDEX IF NOT EXISTS ix_docs_lang    ON gr_documents(language);

-- External-content FTS index: content='gr_chunks' means FTS5 stores only the
-- inverted index and reads the text back from gr_chunks, so the corpus text is
-- not duplicated on disk. content_rowid ties an FTS row to its faiss_id.
CREATE VIRTUAL TABLE IF NOT EXISTS gr_chunks_fts USING fts5(
    text,
    content='gr_chunks',
    content_rowid='faiss_id',
    tokenize='unicode61'
);

-- The supersede/citation knowledge graph (PLAN Phase 3). One row per reference
-- a GR makes, whether or not we hold the referenced document.
--
-- `resolution` is deliberately three-valued rather than "edge exists or not":
--   resolved  - dst_number matched exactly one document in the corpus
--   dangling  - a real reference to a GR we do not hold
--   ambiguous - matched more than one document (OCR collision)
-- Dropping the danglers would overstate how complete the graph is. For a
-- government tool "this GR cites an order we don't have" is INFORMATION the
-- officer needs, not an error to hide.
CREATE TABLE IF NOT EXISTS gr_edges (
    src_id     TEXT NOT NULL,          -- the citing document's id
    dst_id     TEXT,                   -- resolved document id; NULL unless resolved
    dst_number TEXT NOT NULL,          -- the reference exactly as printed
    dst_canon  TEXT,                   -- canonical form used for the match
    kind       TEXT NOT NULL,          -- 'supersedes' | 'cites'
    resolution TEXT NOT NULL,
    PRIMARY KEY (src_id, dst_number, kind)
);

-- Both directions are indexed: traversal asks "what does X cite?" AND
-- "who cites X?", and the second is the one that answers "was this superseded?"
CREATE INDEX IF NOT EXISTS ix_edges_src ON gr_edges(src_id);
CREATE INDEX IF NOT EXISTS ix_edges_dst ON gr_edges(dst_id);
CREATE INDEX IF NOT EXISTS ix_edges_canon ON gr_edges(dst_canon);
"""

# Columns added after the first corpus was already built. SQLite's
# CREATE TABLE IF NOT EXISTS does NOT add columns to an existing table, so a
# schema change needs an explicit migration or the 18k-document corpus would
# have to be re-embedded (~25 min of GPU) for what is a metadata-only change.
_MIGRATIONS = [
    # canonical GR number, so edge resolution is an indexed equality join
    # instead of a LIKE scan over 18,000 rows per reference.
    ("gr_documents", "gr_number_canon", "TEXT"),
    # the DATE printed beside the cited number ('..., दि. ३०.१०.२०१०'). It is
    # what separates two documents that share a GR number, and it is also what
    # the UI shows on a dangling reference so the gap is identifiable.
    ("gr_edges", "dst_date", "TEXT"),
]


def _migrate(conn):
    """Add any missing columns, then backfill the ones that can be derived."""
    for table, column, decl in _MIGRATIONS:
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_docs_canon "
                 "ON gr_documents(gr_number_canon)")


def backfill_canonical_numbers(conn):
    """Populate gr_documents.gr_number_canon for rows that lack it.

    Runs over documents already ingested (the canonical form is computed in
    upsert_document for new ones). Returns how many rows were updated.
    """
    from engine import gr_metadata
    rows = conn.execute(
        "SELECT id, gr_number FROM gr_documents "
        "WHERE gr_number IS NOT NULL AND gr_number_canon IS NULL").fetchall()
    updates = [(gr_metadata.canonical_number(r["gr_number"]), r["id"]) for r in rows]
    updates = [(c, i) for c, i in updates if c]
    conn.executemany("UPDATE gr_documents SET gr_number_canon=? WHERE id=?", updates)
    return len(updates)


@contextmanager
def connect(path=None, readonly=False):
    """A connection with the pragmas this workload actually needs.

    WAL: ingestion writes while the API may be reading; the default rollback
    journal takes a whole-database lock and they would block each other.
    synchronous=NORMAL: a bulk ingest of 65k chunks fsyncs on every commit
    otherwise, which on this filesystem dominates the runtime. The cost is that
    a power loss can lose the last transaction — acceptable, because ingestion
    is resumable by design and would simply redo it.
    """
    db_path = path or CORPUS_DB
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    if not readonly:
        conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init(path=None):
    with connect(path) as c:
        c.executescript(_SCHEMA)
        _migrate(c)


# --------------------------------------------------------------------------
# writing (ingestion)
# --------------------------------------------------------------------------

def upsert_document(conn, doc_id, meta, text, n_chunks, source_file=None):
    """Insert/replace one GR's document row. INSERT OR REPLACE (not INSERT) is
    what makes re-ingesting a document idempotent — the ingest script re-runs
    over the whole corpus and must not accumulate duplicates."""
    from engine import gr_metadata
    conn.execute(
        """INSERT OR REPLACE INTO gr_documents
           (id, gr_number, gr_number_canon, department, date, category, language,
            title, source_file, n_chunks, refs, supersedes, text)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (doc_id, meta.get("gr_number"),
         gr_metadata.canonical_number(meta.get("gr_number")),
         meta.get("department"), meta.get("date"),
         meta.get("category"), meta.get("language"), meta.get("title"),
         source_file or f"{doc_id}.pdf", n_chunks,
         json.dumps(meta.get("reference_details") or meta.get("references") or [],
                    ensure_ascii=False),
         1 if meta.get("supersedes") else 0, text))


def _fts_remove(conn, ids):
    """Un-index rows from the FTS table.

    AN EXTERNAL-CONTENT FTS5 TABLE CANNOT BE DELETED FROM WITH `DELETE`. The FTS
    table stores only the inverted index (term -> rowid); to remove a row it has
    to re-tokenize that row's ORIGINAL text and subtract those terms, so it needs
    the text handed back to it via the special 'delete' command. A plain
    `DELETE FROM gr_chunks_fts` silently leaves every term pointing at a rowid
    that no longer exists — BM25 then returns ghost ids that hydrate to nothing.
    Caught by test_delete_chunks_from_is_crash_recovery.

    Consequence: this must run BEFORE the gr_chunks rows are deleted, while the
    original text is still readable.
    """
    ids = list(ids)
    if not ids:
        return
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT faiss_id, text FROM gr_chunks WHERE faiss_id IN ({marks})", ids).fetchall()
    conn.executemany(
        "INSERT INTO gr_chunks_fts(gr_chunks_fts, rowid, text) VALUES('delete',?,?)",
        [(r[0], r[1]) for r in rows])


def insert_chunks(conn, rows):
    """rows: [(faiss_id, gr_id, chunk_index, page_start, page_end, content_type, text)].
    The FTS row is written explicitly because an external-content FTS5 table is
    NOT auto-synced — SQLite leaves that to the writer (or to triggers, which we
    skip: at ingest scale one explicit batched insert is much faster)."""
    # Clear whatever these ids indexed before. Normally a no-op (faiss ids are
    # append-only), but after a crash-recovery truncation the same ids ARE
    # reused by the re-ingest, and the old terms must go with them.
    _fts_remove(conn, [r[0] for r in rows])
    conn.executemany(
        "INSERT OR REPLACE INTO gr_chunks VALUES (?,?,?,?,?,?,?)", rows)
    conn.executemany(
        "INSERT INTO gr_chunks_fts(rowid, text) VALUES (?,?)",
        [(r[0], r[6]) for r in rows])


def delete_chunks_from(conn, faiss_id):
    """Drop every chunk at or above `faiss_id`.

    CRASH RECOVERY. The FAISS index and this DB are two files that must agree:
    chunk row N must describe vector N. They cannot be written in one atomic
    transaction, so the ingest checkpoints FAISS first and treats the SAVED
    INDEX AS THE AUTHORITY. On restart, any chunk rows past the index's ntotal
    are from a transaction whose vectors never made it to disk — they are
    deleted here, and their documents get re-ingested. Without this the two
    files drift by a few rows and every citation after that point silently
    points at the wrong GR.
    """
    ids = [r[0] for r in conn.execute(
        "SELECT faiss_id FROM gr_chunks WHERE faiss_id >= ?", (faiss_id,))]
    if not ids:
        return 0
    _fts_remove(conn, ids)          # must precede the delete — see _fts_remove
    conn.execute("DELETE FROM gr_chunks WHERE faiss_id >= ?", (faiss_id,))
    return len(ids)


def ingested_ids(conn):
    """The set of document ids already in the corpus — how the ingest script
    skips work on a resumed run."""
    return {r[0] for r in conn.execute("SELECT id FROM gr_documents")}


# --------------------------------------------------------------------------
# reading (retrieval / API)
# --------------------------------------------------------------------------

def chunks_by_faiss_ids(conn, faiss_ids):
    """{faiss_id: chunk-dict} for the ids FAISS just returned, in ONE query.

    The chunk dict is shaped like the old FaissStore hit metadata (source_file,
    content_type, chunk_index, page_start, page_end + the document's GR
    metadata) so retrieval, citation resolution and the officer tools did not
    have to change shape when the storage moved out of RAM.
    """
    if not faiss_ids:
        return {}
    marks = ",".join("?" * len(faiss_ids))
    rows = conn.execute(
        f"""SELECT c.faiss_id, c.gr_id, c.chunk_index, c.page_start, c.page_end,
                   c.content_type, c.text,
                   d.gr_number, d.department, d.date, d.category, d.language,
                   d.title, d.source_file
            FROM gr_chunks c JOIN gr_documents d ON d.id = c.gr_id
            WHERE c.faiss_id IN ({marks})""", list(faiss_ids)).fetchall()
    out = {}
    for r in rows:
        out[r["faiss_id"]] = {
            "text": r["text"],
            "metadata": {
                "order_id": r["gr_id"], "source_file": r["source_file"],
                "source_type": "gr", "content_type": r["content_type"],
                "chunk_index": r["chunk_index"], "page_start": r["page_start"],
                "page_end": r["page_end"], "gr_number": r["gr_number"],
                "department": r["department"], "date": r["date"],
                "category": r["category"], "language": r["language"],
                "title": r["title"],
            },
        }
    return out


def search_bm25(conn, question, k, faiss_id_filter=None):
    """Top-k (faiss_id, bm25_score) by FTS5 BM25, best first.

    bm25() returns a NEGATIVE number where more-negative is better, so it is
    negated to make "higher is better" like every other score in this codebase.
    Returns [] for a query with no usable tokens rather than raising.
    """
    tokens = hybrid.tokenize(question)
    if not tokens:
        return []
    # Quote every token: it neutralises FTS5's operator characters, so a GR
    # number like 'संकीर्ण-२०२३/प्र.क्र.४५' is a literal, not an expression.
    match = " OR ".join('"' + t.replace('"', '') + '"' for t in tokens)
    sql = ("SELECT rowid, bm25(gr_chunks_fts) AS s FROM gr_chunks_fts "
           "WHERE gr_chunks_fts MATCH ? ORDER BY s LIMIT ?")
    try:
        rows = conn.execute(sql, (match, k)).fetchall()
    except sqlite3.OperationalError:
        return []
    return [(r["rowid"], -r["s"]) for r in rows]


def filter_faiss_ids(conn, departments=None, date_from=None, date_to=None,
                     language=None):
    """The set of faiss_ids whose GR matches the given facets, or None when no
    facet is set (meaning "no filtering" — the caller then skips the whole
    membership test rather than building a 65k-element set for nothing).

    THIS IS THE HONEST ANSWER TO FAISS'S BIGGEST LIMITATION. A FAISS HNSW index
    cannot pre-filter: it walks a graph built over all vectors and has no idea
    what a 'department' is. So filtering is done HERE, in SQLite, where the
    metadata actually lives and is indexed — and retrieval over-fetches from
    HNSW so that enough survivors remain after the filter (see
    HnswStore.search's `where`). Native pre-filtering is what Qdrant/pgvector
    would buy us; at this corpus size the over-fetch is cheaper than running
    another server (PLAN Phase 2, "metadata filtering caveat").
    """
    clauses, params = [], []
    if departments:
        clauses.append("d.department IN (%s)" % ",".join("?" * len(departments)))
        params += list(departments)
    if date_from:
        clauses.append("d.date >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("d.date <= ?")
        params.append(date_to)
    if language:
        clauses.append("d.language = ?")
        params.append(language)
    if not clauses:
        return None
    sql = ("SELECT c.faiss_id FROM gr_chunks c JOIN gr_documents d ON d.id = c.gr_id "
           "WHERE " + " AND ".join(clauses))
    return {r[0] for r in conn.execute(sql, params)}


def reference_entries(raw):
    """Normalise the `refs` column to [{'number', 'date'}, ...].

    Two shapes exist on disk and both must keep working: a corpus ingested
    before the cited date was parsed holds plain strings, a current one holds
    dicts. Re-ingesting 18k documents to change a JSON shape would cost ~25 min
    of GPU for nothing, so the READER absorbs the difference instead.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw or "[]")
        except ValueError:
            return []
    out = []
    for r in raw or []:
        if isinstance(r, dict):
            if r.get("number"):
                out.append({"number": r["number"], "date": r.get("date")})
        elif r:
            out.append({"number": str(r), "date": None})
    return out


def get_document(conn, doc_id):
    r = conn.execute("SELECT * FROM gr_documents WHERE id=?", (doc_id,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["reference_details"] = reference_entries(d.pop("refs"))
    d["references"] = [e["number"] for e in d["reference_details"]]
    d["supersedes"] = bool(d["supersedes"])
    return d


def list_documents(conn, q=None, departments=None, limit=50, offset=0):
    """Document list for the Browse page. `q` is a LIKE over title/gr_number —
    deliberately not FTS: Browse is a metadata lookup ("find me GR 123"), while
    FTS is for the retrieval path over chunk bodies."""
    clauses, params = [], []
    if q:
        clauses.append("(title LIKE ? OR gr_number LIKE ? OR id LIKE ?)")
        params += [f"%{q}%"] * 3
    if departments:
        clauses.append("department IN (%s)" % ",".join("?" * len(departments)))
        params += list(departments)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = conn.execute(
        f"""SELECT id, gr_number, department, date, category, language, title, n_chunks
            FROM gr_documents {where} ORDER BY date DESC, id DESC LIMIT ? OFFSET ?""",
        params + [limit, offset]).fetchall()
    total = conn.execute(
        f"SELECT COUNT(*) FROM gr_documents {where}", params).fetchone()[0]
    return [dict(r) for r in rows], total


def find_document(conn, doc_id):
    """The one GR matching doc_id, matched the way the officer tools always have
    — as a SUBSTRING of the order id, GR number or title, so an officer can
    paste a partial reference. Exact id first, so a full id never loses to some
    longer document that merely contains it."""
    exact = conn.execute("SELECT * FROM gr_documents WHERE id = ?", (doc_id,)).fetchone()
    if exact:
        return dict(exact)
    like = f"%{doc_id}%"
    row = conn.execute(
        """SELECT * FROM gr_documents
           WHERE id LIKE ? OR gr_number LIKE ? OR source_file LIKE ?
           ORDER BY LENGTH(COALESCE(gr_number, id)) LIMIT 1""",
        (like, like, like)).fetchone()
    return dict(row) if row else None


def document_chunks(conn, doc_id):
    """Every chunk of one GR, in reading order, as retrieval-shaped hit dicts.

    Uses ix_chunks_gr, so this is an index seek regardless of corpus size — the
    old implementation scanned all 65k chunks in Python for every summarize/
    compare call.
    """
    doc = find_document(conn, doc_id)
    if not doc:
        return []
    rows = conn.execute(
        """SELECT faiss_id, chunk_index, page_start, page_end, content_type, text
           FROM gr_chunks WHERE gr_id = ? ORDER BY chunk_index""",
        (doc["id"],)).fetchall()
    meta_base = {
        "order_id": doc["id"], "source_file": doc["source_file"],
        "source_type": "gr", "gr_number": doc["gr_number"],
        "department": doc["department"], "date": doc["date"],
        "category": doc["category"], "language": doc["language"],
        "title": doc["title"],
    }
    return [{"text": r["text"], "index": r["faiss_id"], "score": 1.0,
             "metadata": {**meta_base, "content_type": r["content_type"],
                          "chunk_index": r["chunk_index"],
                          "page_start": r["page_start"], "page_end": r["page_end"]}}
            for r in rows]


def superseding_documents(conn, gr_number):
    """GRs that cite `gr_number` AND declare a supersession — i.e. the later
    orders that may have cancelled it.

    The candidate set is narrowed in SQL (supersedes=1 AND refs LIKE '%num%')
    and only then checked exactly in Python. `refs` is a JSON list, so LIKE can
    match a substring across list elements; the exact re-check is what stops
    'GR-12' from matching 'GR-123'. Scanning every document in Python — what
    the flat backend does — is what this replaces.
    """
    if not gr_number:
        return []
    rows = conn.execute(
        """SELECT id, gr_number, date, title, refs FROM gr_documents
           WHERE supersedes = 1 AND refs LIKE ?""", (f"%{gr_number}%",)).fetchall()
    out = []
    for r in rows:
        refs = reference_entries(r["refs"])
        if any(gr_number in e["number"] for e in refs):
            out.append({"doc": r["id"], "gr_number": r["gr_number"],
                        "date": r["date"], "title": (r["title"] or "")[:80]})
    return out


def documents_for_gr_numbers(conn, gr_numbers):
    """{cited_number: doc_id or None} — is each cited GR number actually in the
    corpus? One query for the whole list instead of one scan per reference."""
    out = {}
    for num in gr_numbers or []:
        row = conn.execute(
            "SELECT id FROM gr_documents WHERE gr_number LIKE ? LIMIT 1",
            (f"%{num}%",)).fetchone()
        out[num] = row[0] if row else None
    return out


def stats(conn):
    """Corpus-size facts for the portal header ("searching N GRs across M
    departments") and for /health."""
    docs = conn.execute("SELECT COUNT(*) FROM gr_documents").fetchone()[0]
    chunks = conn.execute("SELECT COUNT(*) FROM gr_chunks").fetchone()[0]
    depts = conn.execute(
        """SELECT department, COUNT(*) n FROM gr_documents
           WHERE department IS NOT NULL GROUP BY department ORDER BY n DESC"""
    ).fetchall()
    span = conn.execute(
        "SELECT MIN(date), MAX(date) FROM gr_documents WHERE date IS NOT NULL").fetchone()
    return {
        "documents": docs,
        "chunks": chunks,
        "departments": [{"name": r[0], "documents": r[1]} for r in depts],
        "date_from": span[0],
        "date_to": span[1],
    }
