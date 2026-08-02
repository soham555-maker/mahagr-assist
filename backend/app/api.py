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
    GET  /health                      liveness + index size + model
    GET  /documents                   list GRs in the index
    GET  /documents/{doc_id}/text     full text of one GR (view/download)
    POST /ask         {question, language?, history?}   grounded, cited answer
    POST /summarize   {doc_id, language?}               FR 3.5.2
    POST /explain     {question, language?}             FR 3.5.1
    POST /compare     {doc_a, doc_b, language?}         FR 3.5.3
    GET  /supersede/{doc_id}                            FR 3.5.5 (no LLM)
    GET  /related/{doc_id}?k=5                          FR 3.5.4 (no LLM)
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine import config, officer, rag

INDEX_DIR = os.environ.get("MAHAGR_INDEX", "index")
_state = {}   # retriever, reranker — loaded once at startup


@asynccontextmanager
async def lifespan(app):
    from engine.reranker import Reranker
    from engine.retrieval import load_default_retriever
    print(f"Loading index ({INDEX_DIR}) + bge-m3 + reranker ...")
    _state["retriever"] = load_default_retriever(index_dir=INDEX_DIR, reranker=Reranker())
    print(f"Ready. {len(_state['retriever'].store)} vectors indexed.")
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
    history: list[dict] | None = None

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
    return {"status": "ok" if r else "loading",
            "indexed_vectors": len(r.store) if r else 0,
            "embedding_model": config.EMBED_MODEL,
            "reranker_model": config.RERANK_MODEL,
            "llm_provider": gc.provider,
            "llm_model": gc.ollama_model if gc.provider == "ollama" else gc.model}


@app.get("/documents")
def documents():
    store = _retriever().store
    docs = officer.list_documents(store)
    return [{"doc": k,
             "gr_number": m.get("gr_number"), "date": m.get("date"),
             "department": m.get("department"), "language": m.get("language"),
             "title": m.get("title")}
            for k, m in docs.items()]


@app.get("/documents/{doc_id}/text")
def document_text(doc_id: str):
    chunks = officer.document_chunks(_retriever().store, doc_id)
    if not chunks:
        raise HTTPException(404, f"no document matching '{doc_id}'")
    m = chunks[0]["metadata"]
    return {"doc_id": doc_id, "gr_number": m.get("gr_number"), "date": m.get("date"),
            "title": m.get("title"),
            "text": "\n".join(c["text"] for c in chunks)}


@app.post("/ask")
def ask(req: AskReq):
    res = rag.answer(req.question, _retriever(), _client(), history=req.history)
    # drop the raw chunk bodies from the wire response; sources carry provenance
    res.pop("chunks", None)
    return res


@app.post("/summarize")
def summarize(req: DocLangReq):
    return officer.summarize(_retriever(), req.doc_id, _client(), language=req.language)


@app.post("/explain")
def explain(req: ExplainReq):
    return officer.explain(_retriever(), req.question, _client(), language=req.language)


@app.post("/compare")
def compare(req: CompareReq):
    return officer.compare(_retriever(), req.doc_a, req.doc_b, _client(), language=req.language)


@app.get("/supersede/{doc_id}")
def supersede(doc_id: str):
    info = officer.supersession(_retriever().store, doc_id)
    if not info["found"]:
        raise HTTPException(404, f"no GR matching '{doc_id}'")
    return info


@app.get("/related/{doc_id}")
def related(doc_id: str, k: int = 5):
    return {"doc_id": doc_id, "related": officer.related(_retriever(), doc_id, k=k)}
