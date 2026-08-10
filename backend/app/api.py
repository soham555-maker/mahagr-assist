"""
api.py — slim FastAPI over the MahaGR engine (Phase 3).

Deliberately dependency-light and on-prem friendly: it loads the FAISS index +
bge-m3 + the reranker ONCE at startup and serves the RAG and officer-assistance
features over HTTP. No Supabase, no auth, no cloud storage — the whole retrieval
stack runs locally; only the LLM call is remote (Groq), and that's the one piece
meant to be swapped for a local Llama (Ollama) for a NIC/on-premise deployment.

Run:
    cd backend && uvicorn app.api:app --reload
    # index dir via MAHAGR_INDEX (default "index"); LLM needs GROQ_API_KEY in .env

Endpoints:
    GET  /health                      liveness + backend + index size + model
    GET  /corpus/stats                documents/chunks/departments/date span
    GET  /documents?q&department&limit&offset    list GRs (PAGINATED)
    GET  /documents/{doc_id}/text     full text of one GR (view/download)
    POST /ask   {question, language?, departments?, date_from?, date_to?,
                 doc_language?, conversation_id?}      grounded, cited answer
    POST /summarize   {doc_id, language?}               FR 3.5.2
    POST /explain     {question, language?}             FR 3.5.1
    POST /compare     {doc_a, doc_b, language?}         FR 3.5.3
    GET  /supersede/{doc_id}                            FR 3.5.5 (no LLM)
    GET  /related/{doc_id}?k=5                          FR 3.5.4 (no LLM)
    GET  /graph/{doc_id}?depth&max_nodes   supersede/citation neighbourhood
    GET  /graph/stats/summary              edge counts + resolution breakdown
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from engine import config, officer, rag
from app import db, auth

INDEX_DIR = os.environ.get("MAHAGR_INDEX") or config.INDEX_DIR
_state = {}   # retriever, reranker, corpus stats — loaded once at startup


def _corpus_stats():
    """Corpus shape (documents / chunks / departments / date span) for the
    portal header and /health. Read ONCE at startup and cached: it is a COUNT
    over 18,000 rows that only changes when the corpus is re-ingested, and the
    frontend asks for it on every page load."""
    store = _state.get("retriever").store if _state.get("retriever") else None
    db_path = getattr(store, "corpus_db_path", None)
    if not db_path or not os.path.exists(db_path):
        return None
    from engine import corpus_db
    with corpus_db.connect(db_path, readonly=True) as conn:
        return corpus_db.stats(conn)


@asynccontextmanager
async def lifespan(app):
    from engine.reranker import Reranker
    from engine.retrieval import load_default_retriever
    db.init()
    print(f"Loading index ({INDEX_DIR}, backend={config.VECTOR_BACKEND}) "
          f"+ bge-m3 + reranker ...")
    _state["retriever"] = load_default_retriever(index_dir=INDEX_DIR, reranker=Reranker())
    _state["corpus"] = _corpus_stats()
    stats = _state["corpus"]
    print(f"Ready. {len(_state['retriever'].store)} vectors indexed."
          + (f" {stats['documents']} GRs across {len(stats['departments'])} departments."
             if stats else ""))
    yield
    _state.clear()


app = FastAPI(title="MahaGR Assist API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _retriever():
    r = _state.get("retriever")
    if r is None:
        raise HTTPException(503, "index not loaded")
    return r


def _client():
    """Groq client, created on demand so the no-LLM endpoints (documents,
    supersede, related, health) work without GROQ_API_KEY."""
    try:
        return rag.make_client()
    except RuntimeError as e:
        raise HTTPException(503, str(e))


# --- request bodies ---
class AskReq(BaseModel):
    question: str
    language: str = "auto"
    conversation_id: str | None = None   # persist + load history across sessions
    # Scope the search to part of the corpus (FR 3.2 / PLAN Phase 2). Resolved
    # to a set of allowed chunk ids in SQLite and pushed into BOTH the ANN and
    # the BM25 search — never applied afterwards, which would silently drop
    # results without saying so.
    departments: list[str] | None = None
    date_from: str | None = None         # ISO YYYY-MM-DD
    date_to: str | None = None
    doc_language: str | None = None      # 'mr' | 'en' — the DOCUMENT's language

    def filters(self):
        """None when nothing is set, so retrieval skips the filter path
        entirely rather than building a 65k-element id set for no reason."""
        f = {"departments": self.departments, "date_from": self.date_from,
             "date_to": self.date_to, "language": self.doc_language}
        f = {k: v for k, v in f.items() if v}
        return f or None

class FeedbackReq(BaseModel):
    conversation_id: str | None = None
    message_id: str | None = None
    rating: str                          # "up" | "down"
    comment: str | None = None

class DocLangReq(BaseModel):
    doc_id: str
    language: str = "auto"

class ExplainReq(BaseModel):
    question: str
    language: str = "auto"

class CompareReq(BaseModel):
    doc_a: str
    doc_b: str
    language: str = "auto"


# --- endpoints ---
@app.get("/health")
def health():
    r = _state.get("retriever")
    gc = rag.GenerationConfig()
    stats = _state.get("corpus")
    out = {"status": "ok" if r else "loading",
           "indexed_vectors": len(r.store) if r else 0,
           "vector_backend": config.VECTOR_BACKEND,
           "documents": stats["documents"] if stats else None,
           "departments": len(stats["departments"]) if stats else None,
           "embedding_model": config.EMBED_MODEL,
           "reranker_model": config.RERANK_MODEL,
           "llm_provider": gc.provider,
           "llm_model": gc.ollama_model if gc.provider == "ollama" else gc.model}
    # WHERE the local model is running. A partial CPU offload is a 4x slowdown
    # that raises no error anywhere (see OllamaClient.placement), so it is
    # reported on the one endpoint an operator always checks.
    if gc.provider == "ollama":
        try:
            placement = rag.make_client(gc).placement()
        except Exception:
            placement = None          # health must never fail because of this
        out["llm_placement"] = placement
        if placement and not placement["fully_on_gpu"]:
            out["warning"] = (
                f"The local model is running {placement['processor']} — decode is "
                f"~4x slower than fully on GPU. Restart Ollama BEFORE this API so "
                f"it sees the whole card.")
    return out


@app.post("/auth/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = db.get_user_by_username(form_data.username)
    if not user or not auth.verify_password(form_data.password, user["password_hash"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    access_token = auth.create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer", "role": user["role"], "username": user["username"]}


@app.get("/corpus/stats")
def corpus_stats():
    """What the corpus actually contains — powers the portal's "searching N GRs
    across M departments" line and the department/date filter options."""
    stats = _state.get("corpus")
    if not stats:
        store = _retriever().store
        return {"documents": len(getattr(store, "metadata", [])), "chunks": len(store),
                "departments": [], "date_from": None, "date_to": None}
    return stats


@app.get("/documents")
def documents(q: str | None = None, department: list[str] | None = Query(None),
              limit: int = 50, offset: int = 0,
              current_user: dict = Depends(auth.get_current_user)):
    """List GRs. PAGINATED — the corpus is 18,000 documents, so the old
    "serialize every document" response would be tens of MB per page load."""
    store = _retriever().store
    docs = officer.list_documents(store, limit=limit, offset=offset, q=q,
                                  departments=department)
    return [{"doc": k,
             "gr_number": m.get("gr_number"), "date": m.get("date"),
             "department": m.get("department"), "language": m.get("language"),
             "title": m.get("title")}
            for k, m in docs.items()]



@app.get("/documents/{doc_id:path}/text")
def document_text(doc_id: str):
    store = _retriever().store
    db_path = getattr(store, "corpus_db_path", None)
    if db_path:
        # Prefer the stored full document text over stitching chunks back
        # together: chunks overlap by 50 words, so joining them repeats a line
        # at every boundary. The flat backend has no full text to fall back on.
        from engine import corpus_db
        with corpus_db.connect(db_path, readonly=True) as conn:
            doc = corpus_db.find_document(conn, doc_id)
        if not doc:
            raise HTTPException(404, f"no document matching '{doc_id}'")
        return {"doc_id": doc["id"], "gr_number": doc["gr_number"],
                "date": doc["date"], "department": doc["department"],
                "title": doc["title"], "text": doc["text"]}

    chunks = officer.document_chunks(store, doc_id)
    if not chunks:
        raise HTTPException(404, f"no document matching '{doc_id}'")
    m = chunks[0]["metadata"]
    return {"doc_id": doc_id, "gr_number": m.get("gr_number"), "date": m.get("date"),
            "title": m.get("title"),
            "text": "\n".join(c["text"] for c in chunks)}


@app.post("/ask")
def ask(req: AskReq, request: Request, current_user: dict = Depends(auth.get_current_user)):
    auth.check_rate_limit(current_user["id"])
    
    # "Legal Translator" can force language, others use auto or document language
    if current_user["role"] not in ["Legal Translator", "IT Admin"] and req.language != "auto":
        req.language = "auto"
        
    r = _retriever()
    # A conversation persists across sessions; load its prior turns from the DB
    # so rag.answer's query-rewrite resolves follow-ups even after a restart.
    cid = req.conversation_id or db.create_conversation(req.question, current_user["id"])
    history = [{"role": m["role"], "content": m["content"]} for m in db.get_messages(cid)]
    res = rag.answer(req.question, r, _client(), history=history,
                     language=req.language, filters=req.filters())
    # conflict/supersede check over the cited GRs (deterministic, metadata-based)
    res["warnings"] = officer.supersede_warnings(r.store, res["sources"])
    res.pop("chunks", None)   # drop raw chunk bodies; sources carry provenance
    # persist this turn
    db.add_message(cid, "user", req.question)
    res["message_id"] = db.add_message(cid, "assistant", res["answer"], res["sources"], res["warnings"])
    res["conversation_id"] = cid
    
    # Audit log
    db.log_audit(
        user_id=current_user["id"],
        action="ask",
        detail={"question": req.question, "cited_gr_ids": [s.get("gr_number") for s in res["sources"]]},
        ip=request.client.host if request.client else "unknown"
    )
    
    return res


@app.get("/conversations")
def conversations(current_user: dict = Depends(auth.get_current_user)):
    all_users = current_user["role"] in ["Reviewer", "IT Admin"]
    return db.list_conversations(current_user["id"], all_users=all_users)


@app.get("/conversations/{cid}")
def conversation(cid: str, current_user: dict = Depends(auth.get_current_user)):
    # TODO: verify current_user owns cid or is Reviewer/Admin
    return {"conversation_id": cid, "messages": db.get_messages(cid)}


@app.delete("/conversations/{cid}")
def delete_conversation(cid: str, current_user: dict = Depends(auth.get_current_user)):
    db.delete_conversation(cid)
    return {"deleted": cid}


@app.post("/feedback")
def feedback(req: FeedbackReq, current_user: dict = Depends(auth.get_current_user)):
    fid = db.add_feedback(req.conversation_id, req.message_id, req.rating, req.comment)
    return {"feedback_id": fid}


@app.post("/summarize")
def summarize(req: DocLangReq, current_user: dict = Depends(auth.get_current_user)):
    if current_user["role"] not in ["Legal Translator", "IT Admin"] and req.language != "auto":
        req.language = "auto"
    return officer.summarize(_retriever(), req.doc_id, _client(), language=req.language)


@app.post("/explain")
def explain(req: ExplainReq, current_user: dict = Depends(auth.get_current_user)):
    if current_user["role"] not in ["Legal Translator", "IT Admin"] and req.language != "auto":
        req.language = "auto"
    return officer.explain(_retriever(), req.question, _client(), language=req.language)


@app.post("/compare")
def compare(req: CompareReq, current_user: dict = Depends(auth.get_current_user)):
    if current_user["role"] not in ["Legal Translator", "IT Admin"] and req.language != "auto":
        req.language = "auto"
    return officer.compare(_retriever(), req.doc_a, req.doc_b, _client(), language=req.language)


@app.get("/supersede/{doc_id:path}")
def supersede(doc_id: str, current_user: dict = Depends(auth.get_current_user)):
    info = officer.supersession(_retriever().store, doc_id)
    if not info["found"]:
        raise HTTPException(404, f"no GR matching '{doc_id}'")
    return info


@app.get("/graph/stats/summary")
def graph_stats(current_user: dict = Depends(auth.get_current_user)):
    """Edge counts + the reference-resolution breakdown (resolved / dangling /
    ambiguous) — the honest measure of how complete the corpus is."""
    store = _retriever().store
    db_path = getattr(store, "corpus_db_path", None)
    if not db_path:
        raise HTTPException(501, "the graph requires the hnsw corpus backend")
    from engine import corpus_db, graph
    with corpus_db.connect(db_path, readonly=True) as conn:
        return graph.stats(conn)


@app.get("/graph/{doc_id:path}")
def document_graph(doc_id: str, depth: int = 1, max_nodes: int = 60, current_user: dict = Depends(auth.get_current_user)):
    """The supersede/citation neighbourhood of one GR (PLAN Phase 3).

    Returns {found, root, nodes, edges, chain, dangling}. `chain` is the
    TRANSITIVE supersede chain — being told a GR was replaced by B is
    misleading when B has itself been replaced. `dangling` are references to
    orders the corpus does not hold, returned separately so the UI can show
    them as visible gaps rather than omitting them.

    Needs the graph to have been built: `python scripts/build_graph.py`.
    """
    store = _retriever().store
    db_path = getattr(store, "corpus_db_path", None)
    if not db_path:
        raise HTTPException(501, "the graph requires the hnsw corpus backend")
    from engine import corpus_db, graph
    with corpus_db.connect(db_path, readonly=True) as conn:
        info = graph.neighbourhood(conn, doc_id, depth=depth, max_nodes=max_nodes)
    if not info["found"]:
        raise HTTPException(404, f"no GR matching '{doc_id}'")
    return info





@app.get("/related/{doc_id:path}")
def related(doc_id: str, k: int = 5, current_user: dict = Depends(auth.get_current_user)):
    return {"doc_id": doc_id, "related": officer.related(_retriever(), doc_id, k=k)}


@app.get("/admin/audit-logs")
def get_audit_logs(current_user: dict = Depends(auth.require_role(["IT Admin"])), limit: int = 100):
    with db._db() as c:
        rows = c.execute("SELECT * FROM audit_log ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]
