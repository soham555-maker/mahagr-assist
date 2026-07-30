"""
app/main.py — the FastAPI box: wraps the proven engine/ pipeline
(engine/retrieval.py + engine/rag.py) in accounts, isolated chats, and a
token quota. Nothing about retrieval or generation changes here; this file
is auth, persistence, and routing only.

MULTI-TENANCY, the one mechanism everything else leans on
-----------------------------------------------------------
get_user_id() verifies the caller's Supabase JWT (a network hop to Auth —
fine at demo scale; local JWT verification is the later upgrade) and returns
user_id. Every private endpoint depends on it, and every DB query is scoped
`user_id = <that value>`. This client uses the SERVICE-ROLE key (trusted
server, bypasses RLS) — but RLS stays enabled on both tables in Supabase as
a backstop: if a bug ever let a query slip through unscoped, the database
itself would still refuse the cross-user read. App-level filter + DB-level
backstop, not one or the other.

TOKEN QUOTA
-----------
Every assistant message stores the REAL prompt_tokens/completion_tokens
Groq returns (rag.answer()'s usage.*, not an estimate). The quota is the sum
of both, over this user's messages created today (UTC), checked BEFORE
calling Groq — so a user already at the limit can't trigger another paid
call. Prompt tokens count deliberately: retrieval context is the expensive
part, and a user pays (in quota) for what they trigger.

Run (from backend/, the working-directory convention this whole layout uses):
    uvicorn app.main:app --reload --port 8000
Requires GROQ_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY in .env.
"""

import os
import tempfile
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from supabase import Client, create_client

from engine.documents import (ASSET_BUCKET, MAX_UPLOAD_BYTES, clean_storage_prefix,
                              hybrid_search, ingest_upload, new_document_id,
                              persist_visual_assets)
from engine import config
from engine.ingest import IngestionPipeline
from engine.rag import GenerationConfig, answer, make_client, rewrite_query
from engine.reranker import Reranker
from engine.retrieval import load_default_retriever

EMBEDDING_MODEL_NAME = config.EMBED_MODEL  # bge-m3 — single source of truth (config.py)
TITLE_MAX_CHARS = 60
# ASSET_BUCKET lives in documents.py now — it's the shared home for both the
# upload-persist and corpus-persist paths (build_index.py uses the same one).


def _required_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in."
        )
    return value


SUPABASE_URL = _required_env("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = _required_env("SUPABASE_SERVICE_ROLE_KEY")
DAILY_TOKEN_LIMIT = int(os.environ.get("DAILY_TOKEN_LIMIT", "80000"))
CORS_ORIGINS = [o.strip() for o in
                os.environ.get("CORS_ORIGIN", "http://localhost:3000").split(",")]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the engine once at process start, exactly like the CLI does —
    the retriever holds the FAISS corpus + embedding model in memory for the
    life of the process; the Groq client and Supabase client are cheap but
    still shared, not re-created per request."""
    print("Loading index + embedding model + reranker...")
    # ONE reranker instance for both retrieval paths (corpus + uploads) — the
    # scores must come from the same model to stay commensurable at merge time.
    app.state.reranker = Reranker()
    app.state.retriever = load_default_retriever(reranker=app.state.reranker)
    app.state.groq_client = make_client()
    app.state.supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    app.state.gen_config = GenerationConfig()
    # Reuses the already-loaded embedding model (see ingest.py's model= param)
    # instead of loading a second copy just for handling uploads.
    app.state.ingestion_pipeline = IngestionPipeline(model=app.state.retriever.model)
    print(f"Ready. Corpus: {len(app.state.retriever.store)} chunks.")
    yield


app = FastAPI(title="ResearchOS API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bearer_scheme = HTTPBearer()


def get_user_id(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> str:
    """THE multi-tenancy mechanism (see module docstring). Any failure to
    verify — expired, malformed, forged — collapses to one 401; the caller
    never learns which, so this boundary can't leak auth internals."""
    try:
        resp = request.app.state.supabase.auth.get_user(credentials.credentials)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if resp is None or resp.user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return resp.user.id


class ConversationCreate(BaseModel):
    title: str | None = None


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    question: str


def _truncate_title(question: str) -> str:
    q = " ".join(question.split())  # collapse whitespace/newlines
    return q if len(q) <= TITLE_MAX_CHARS else q[:TITLE_MAX_CHARS - 1].rstrip() + "…"


def _today_usage(supabase: Client, user_id: str) -> int:
    """Sum of prompt_tokens + completion_tokens across this user's messages
    created today (UTC). The quota query — indexed on (user_id, created_at)
    for exactly this access pattern."""
    today_start = datetime.combine(
        date.today(), datetime.min.time(), tzinfo=timezone.utc
    ).isoformat()
    resp = (
        supabase.table("messages")
        .select("prompt_tokens, completion_tokens")
        .eq("user_id", user_id)
        .gte("created_at", today_start)
        .execute()
    )
    return sum(r["prompt_tokens"] + r["completion_tokens"] for r in resp.data)


def _get_owned_conversation(supabase: Client, conversation_id: str, user_id: str):
    """404, not 403, on a conversation that isn't this user's — a 403 would
    confirm the ID exists at all, which is itself a cross-tenant leak."""
    resp = (
        supabase.table("conversations")
        .select("id, title")
        .eq("id", conversation_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not resp.data:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return resp.data[0]


def _load_recency_window(supabase: Client, conversation_id: str, config: GenerationConfig):
    """Last config.history_window messages of this conversation, oldest
    first — exactly the shape rag.build_prompt's history param expects."""
    resp = (
        supabase.table("messages")
        .select("role, content")
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=True)
        .limit(config.history_window)
        .execute()
    )
    return list(reversed(resp.data))


def _fetch_user_document_chunks(supabase: Client, user_id: str):
    """All of this user's uploaded-document chunks (see engine/documents.py),
    reshaped to the same {'text', 'embedding', 'metadata'} ingest.py already
    produces, so documents.hybrid_search scores them exactly like any other
    chunk — no per-source branching downstream."""
    resp = (
        supabase.table("document_chunks")
        .select("content, embedding, metadata")
        .eq("user_id", user_id)
        .execute()
    )
    return [
        {"text": r["content"], "embedding": r["embedding"], "metadata": r["metadata"]}
        for r in resp.data
    ]


@app.get("/health")
def health(request: Request):
    """Deploy smoke check: index loaded, model loaded, process alive."""
    return {
        "status": "ok",
        "corpus_chunks": len(request.app.state.retriever.store),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranker_model": request.app.state.reranker.model_name,
        "generation_model": request.app.state.gen_config.model,
    }


@app.get("/conversations")
def list_conversations(request: Request, user_id: str = Depends(get_user_id)):
    resp = (
        request.app.state.supabase.table("conversations")
        .select("id, title, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return resp.data


@app.post("/conversations")
def create_conversation(
    body: ConversationCreate,
    request: Request,
    user_id: str = Depends(get_user_id),
):
    row = {"user_id": user_id, "title": body.title or "New chat"}
    resp = request.app.state.supabase.table("conversations").insert(row).execute()
    return resp.data[0]


@app.get("/conversations/{conversation_id}/messages")
def get_messages(
    conversation_id: str,
    request: Request,
    user_id: str = Depends(get_user_id),
):
    supabase = request.app.state.supabase
    _get_owned_conversation(supabase, conversation_id, user_id)  # 404 if not theirs
    resp = (
        supabase.table("messages")
        .select("id, role, content, prompt_tokens, completion_tokens, sources, "
                "low_confidence, created_at")
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=False)
        .execute()
    )
    return resp.data


@app.get("/usage")
def usage(request: Request, user_id: str = Depends(get_user_id)):
    used = _today_usage(request.app.state.supabase, user_id)
    return {
        "used_today": used,
        "limit": DAILY_TOKEN_LIMIT,
        "remaining": max(DAILY_TOKEN_LIMIT - used, 0),
    }


@app.post("/chat")
def chat(
    body: ChatRequest,
    request: Request,
    user_id: str = Depends(get_user_id),
):
    supabase = request.app.state.supabase
    config = request.app.state.gen_config

    # Quota gate BEFORE calling Groq — an already-over-limit user can't
    # trigger another paid call.
    used_before = _today_usage(supabase, user_id)
    if used_before >= DAILY_TOKEN_LIMIT:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "daily_token_limit_reached",
                "message": "Daily token limit reached. Try again tomorrow.",
                "used_today": used_before,
                "limit": DAILY_TOKEN_LIMIT,
            },
        )

    # Resolve (or create) the conversation.
    if body.conversation_id:
        conversation = _get_owned_conversation(supabase, body.conversation_id, user_id)
    else:
        row = {"user_id": user_id, "title": _truncate_title(body.question)}
        conversation = supabase.table("conversations").insert(row).execute().data[0]
    conversation_id = conversation["id"]

    history = _load_recency_window(supabase, conversation_id, config)

    # Retrieval only ever embeds/searches on the literal text handed to it —
    # it has no access to `history` the way generation does (build_prompt
    # inserts history as real chat turns; retrieval never sees it). A
    # follow-up like "explain the importance of this paper" would otherwise
    # get searched on that literal string, with no signal about which paper
    # "this" refers to. rewrite_query() resolves that BEFORE either
    # retrieval call below runs, using the same history generation already
    # gets — skips the LLM call entirely on a conversation's first message
    # (no history to resolve against). body.question itself is left
    # untouched: it's still what's shown to the model, stored in messages,
    # and used for the conversation title.
    retrieval_query = rewrite_query(body.question, history,
                                    request.app.state.groq_client, config)

    # Retrieval happens HERE, not inside answer(), so a corpus figure/formula's
    # image_path can be resolved to a signed image_url BEFORE generation —
    # same treatment extra_chunks (uploads) get below. Without this, only
    # upload-side figures would ever reach the vision model; corpus figures
    # would be retrieved and cited but never actually SEEN by the model.
    retrieval_result = request.app.state.retriever.retrieve(retrieval_query)
    _resolve_image_urls(supabase, retrieval_result["chunks"])

    # Only pays the extra DB read (+ a second query embedding) when this
    # user actually has uploads — the common case (no uploads yet) is a
    # single cheap query returning nothing, no change to the request cost.
    extra_chunks = []
    user_chunks = _fetch_user_document_chunks(supabase, user_id)
    if user_chunks:
        query_vec = request.app.state.retriever.embed_query(retrieval_query)
        extra_chunks = hybrid_search(retrieval_query, query_vec, user_chunks,
                                     reranker=request.app.state.reranker)
        _resolve_image_urls(supabase, extra_chunks)

    result = answer(
        body.question,
        request.app.state.retriever,
        request.app.state.groq_client,
        config,
        history=history,
        extra_chunks=extra_chunks,
        retrieval_result=retrieval_result,
    )

    # Persist both turns. The user turn carries no token cost of its own —
    # the real cost (context + history + question, all of it) lands on the
    # assistant turn via Groq's usage.*, which is what the quota sums.
    supabase.table("messages").insert({
        "conversation_id": conversation_id,
        "user_id": user_id,
        "role": "user",
        "content": body.question,
    }).execute()

    supabase.table("messages").insert({
        "conversation_id": conversation_id,
        "user_id": user_id,
        "role": "assistant",
        "content": result["answer"],
        "prompt_tokens": result["usage"]["prompt_tokens"],
        "completion_tokens": result["usage"]["completion_tokens"],
        "sources": result["sources"],
        "low_confidence": result["low_confidence"],
    }).execute()

    used_after = used_before + result["usage"]["prompt_tokens"] + result["usage"]["completion_tokens"]

    return {
        "conversation_id": conversation_id,
        "answer": result["answer"],
        "sources": result["sources"],
        "phantom_citations": result["phantom_citations"],
        "low_confidence": result["low_confidence"],
        "truncated": result["truncated"],
        "usage": result["usage"],
        "model": result["model"],   # which model answered (text vs vision) — UI + debug
        "quota": {
            "used_today": used_after,
            "limit": DAILY_TOKEN_LIMIT,
            "remaining": max(DAILY_TOKEN_LIMIT - used_after, 0),
        },
    }


ASSET_URL_TTL = 3600  # seconds a signed figure/formula URL stays valid


def _sign_asset(supabase: Client, path: str):
    """Short-lived signed URL for a Storage object. Returns None on failure so
    a missing/expired crop degrades to 'no image' rather than 500-ing the chat."""
    try:
        res = supabase.storage.from_(ASSET_BUCKET).create_signed_url(path, ASSET_URL_TTL)
        return res.get("signedURL") or res.get("signedUrl")
    except Exception:
        return None


def _resolve_image_urls(supabase: Client, chunks):
    """For every hit that has an image_path (figure/formula, corpus OR
    upload), mint a signed URL and inject it as image_url so build_prompt can
    attach the image and answer() switches to the vision model. Applied to
    BOTH the corpus retrieval result and upload extra_chunks in /chat — a
    figure from either source must reach the vision model the same way."""
    for chunk in chunks:
        path = chunk["metadata"].get("image_path")
        if path:
            signed = _sign_asset(supabase, path)
            if signed:
                chunk["metadata"]["image_url"] = signed


@app.get("/documents")
def list_documents(request: Request, user_id: str = Depends(get_user_id)):
    resp = (
        request.app.state.supabase.table("documents")
        .select("id, filename, title, status, error, chunk_count, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return resp.data


@app.post("/documents")
def upload_document(
    request: Request,
    user_id: str = Depends(get_user_id),
    file: UploadFile = File(...),
):
    """
    Runs the uploaded PDF through the SAME ingestion pipeline that built the
    corpus (see documents.py), then stores its chunks as this user's own
    private rows. Synchronous — no background job queue, matching the rest
    of this sprint's "no infra we don't need yet" stance; a single paper
    takes a few seconds on CPU, acceptable for one request/response cycle.
    Nothing is written to `documents` on failure — a failed upload just
    leaves no row, simpler than reasoning about a half-ingested one.
    """
    is_pdf = (file.content_type == "application/pdf"
              or (file.filename or "").lower().endswith(".pdf"))
    if not is_pdf:
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    supabase = request.app.state.supabase
    document_id = new_document_id()
    filename = file.filename or "upload.pdf"
    title = os.path.splitext(filename)[0]

    tmp_path = None
    try:
        contents = file.file.read(MAX_UPLOAD_BYTES + 1)
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit)",
            )

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        chunks = ingest_upload(
            tmp_path, request.app.state.ingestion_pipeline, document_id, title
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not process PDF: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No extractable text found in this PDF (scanned image?)",
        )

    # Upload figure/formula crops to Storage and fold their paths into metadata
    # (mutates chunks: strips image_bytes) BEFORE the rows are written.
    persist_visual_assets(supabase, chunks, f"{user_id}/{document_id}")

    supabase.table("documents").insert({
        "id": str(document_id),
        "user_id": user_id,
        "filename": filename,
        "title": title,
        "status": "ready",
        "chunk_count": len(chunks),
    }).execute()

    supabase.table("document_chunks").insert([
        {
            "document_id": str(document_id),
            "user_id": user_id,
            "content": c["text"],
            "embedding": [float(x) for x in c["embedding"]],
            "metadata": c["metadata"],
        }
        for c in chunks
    ]).execute()

    return {
        "id": str(document_id),
        "filename": filename,
        "title": title,
        "status": "ready",
        "chunk_count": len(chunks),
    }


@app.delete("/documents/{document_id}")
def delete_document(
    document_id: str, request: Request, user_id: str = Depends(get_user_id)
):
    supabase = request.app.state.supabase
    resp = (
        supabase.table("documents")
        .delete()
        .eq("id", document_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not resp.data:
        raise HTTPException(status_code=404, detail="Document not found")
    # DB rows cascade via FK; Storage objects don't, so clean them explicitly.
    clean_storage_prefix(supabase, f"{user_id}/{document_id}")
    return {"deleted": True}


@app.get("/documents/asset-url")
def asset_url(path: str, request: Request, user_id: str = Depends(get_user_id)):
    """A fresh signed URL for one figure/formula crop. Persisted message
    sources hold the durable image_path (signed URLs expire), so the frontend
    calls this to render a figure citation. Two valid prefixes: a user's own
    `{user_id}/...` (upload isolation, unchanged), or the shared `corpus/...`
    (every corpus figure is already visible to every authenticated user via
    /chat — the corpus has no owner to check, so being logged in IS the
    authorization, same as reading corpus chunks at all)."""
    is_own = path.startswith(f"{user_id}/")
    is_corpus = path.startswith("corpus/")
    if not (is_own or is_corpus):
        raise HTTPException(status_code=404, detail="Not found")
    signed = _sign_asset(request.app.state.supabase, path)
    if not signed:
        raise HTTPException(status_code=404, detail="Not found")
    return {"url": signed}
