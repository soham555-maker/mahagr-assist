"""
rag.py — the Day 5 box: retrieved chunks + question in, cited grounded answer out.

This is the NON-deterministic half of the RAG loop, so it's built around one
boundary rule: everything below the Groq call (prompt assembly, citation
parsing, trimming) is a pure function, testable with assertions and zero API
cost (test_rag.py); the LLM call itself is a thin, retry-wrapped edge.

GROUNDING — three mechanics, three failure modes prevented
----------------------------------------------------------
1. Context-only instruction: without it, Llama answers about these arXiv
   classics from its own training data while ignoring our retrieval entirely —
   plausible answers that owe nothing to the pipeline. Invisible failure.
2. Numbered blocks + mandatory [n] citations: every claim becomes mechanically
   checkable — each [n] maps back to (paper, pages) via chunk metadata. A
   citation to a block that doesn't exist (a "phantom") is DETECTED and
   SURFACED, never silently stripped: it's our groundedness monitor firing.
3. Permission to refuse: models hallucinate hardest when they think they must
   answer. Refusal is an explicitly sanctioned outcome, and retrieval's
   low_confidence flag (the floor fired — see retrieval.py) strengthens it:
   the model is told the context matched weakly and to say so if it doesn't
   actually answer. Retrieval's knowledge of its own failure flows into the
   prompt instead of being discarded.

Citations prove PROVENANCE, not correctness of synthesis: a cited answer tells
you where to look to verify, not that the model read the block right.

TOKEN BUDGET (interim, formalized on Day 8)
-------------------------------------------
Estimated as words x 1.33 (English rule of thumb; arXiv prose runs higher).
The budget only constrains the CONTEXT blocks — system prompt + question are
small fixed costs, and max_tokens caps the answer separately. When over
budget, the lowest-scoring chunks are dropped first: the relevance ranking is
exactly the value-per-token order, so the cheapest cut is the least relevant
chunk. Real counts come back from the API in usage.* on every call — the
estimator can be sanity-checked for free.

Usage:
    python rag.py "What BLEU score did the Transformer get on EN-DE?"
    python rag.py            # interactive REPL
Requires GROQ_API_KEY in .env (copy .env.example; key from console.groq.com).
"""

import os
import re
import sys
import time
from dataclasses import dataclass

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()  # makes GROQ_API_KEY from .env visible to the groq client


@dataclass
class GenerationConfig:
    model: str = "llama-3.3-70b-versatile"   # free-tier quality pick (text-only)
    vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"  # used only when a figure/formula image is in context
    scratch_model: str = "llama-3.1-8b-instant"  # cheap runs while iterating
    temperature: float = 0.2   # low = boring and faithful; 0.0 for reproducible tests
    max_tokens: int = 1024     # output cap (cost/runaway guard, not a compressor)
    context_token_budget: int = 6000  # estimated tokens allowed for context blocks
    history_window: int = 6    # recency-window turns (3 exchanges) sent as real roles
    max_images: int = 3        # cap figure/formula images attached per request (tokens + rate)
    max_retries: int = 2       # extra attempts after a 429, with backoff
    retry_base_delay: float = 2.0  # seconds; doubles each retry


TOKENS_PER_WORD = 1.33  # English heuristic; checked against usage.prompt_tokens


def estimate_tokens(text):
    """Rough token count: words x 1.33. Good enough to keep prompts in a safe
    zone until Day 8 wires in real counting."""
    return int(len(text.split()) * TOKENS_PER_WORD)


# --------------------------------------------------------------------------- #
# prompt assembly — pure functions, no API, no I/O
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """\
You are an assistant for Maharashtra Government officers. You answer questions \
about official Government documents — Government Resolutions (GRs), circulars, \
notifications and office orders — using ONLY the numbered context blocks \
provided in the user message.

Rules:
- Every factual claim must cite the block(s) it came from, like [1] or [2][3].
- Copy GR numbers, dates, amounts, and figures EXACTLY as they appear in the \
context; never invent, estimate, or round them.
- If the context does not contain the answer, say so plainly and stop there \
— do not follow a refusal with an answer from your own general knowledge. \
Refusing is a correct, complete answer on its own. This matters more here than \
sounding helpful: an officer may act on what you say.
- If two context blocks conflict (e.g. one GR supersedes or amends another), \
say so explicitly and cite both, rather than silently picking one.
- Be concise and direct. Do not narrate your reasoning ("Step 1..."), do not \
restate the question, do not describe the context blocks themselves — just \
answer.

Language:
- Answer in the SAME language as the question (Marathi question -> Marathi \
answer; English question -> English answer) unless the user explicitly asks \
for another language.
- Preserve official Government terminology. When you quote or refer to the \
exact wording of a GR, keep it in its ORIGINAL language (usually Marathi) even \
if the rest of your answer is in English — do not translate legal/official \
phrasing away from its source.

Formatting:
- Write in Markdown. Use headers/bold/lists only where they genuinely help; \
a short answer needs none of them.
- If a context block is a figure or scanned image, do NOT redraw or transcribe \
it as an ASCII diagram — the actual image is shown to the user next to your \
citation. Cite it (e.g. "shown in [1]") and, only if useful, describe it in one \
or two plain sentences."""

LOW_CONFIDENCE_ADDENDUM = """

IMPORTANT: retrieval matched this question only WEAKLY — the blocks below are \
the closest chunks found, but nothing scored as a confident match. If they do \
not actually answer the question, respond exactly along the lines of: "The \
corpus does not appear to cover this. The closest material found is about \
<topic of the blocks>." Do not stretch weak context into an answer."""


def format_block(n, hit):
    """One numbered context block with a provenance header. The header is what
    makes a later [n] citation resolvable to (paper, pages) — and it also
    tells the model what it's reading (a table row rendering vs. prose)."""
    m = hit["metadata"]
    pages = (f"page {m['page_start']}" if m["page_start"] == m["page_end"]
             else f"pages {m['page_start']}-{m['page_end']}")
    title = m.get("title") or m.get("source_file", "unknown")
    return (f"[{n}] ({title} — {m.get('paper_id', '?')}, {pages}, "
            f"{m['content_type']})\n{hit['text']}")


def trim_to_budget(chunks, budget):
    """
    Keep as many chunks as fit in the estimated token budget, dropping the
    LOWEST-scoring first (chunks arrive sorted by score desc, so we cut from
    the tail). Never drops below one block: an over-budget single chunk still
    goes through, because an empty prompt is worse than a long one.
    Returns (kept_chunks, dropped_count).
    """
    kept, spent = [], 0
    for hit in chunks:
        cost = estimate_tokens(hit["text"])
        if kept and spent + cost > budget:
            break  # sorted desc => everything after is lower-scoring
        kept.append(hit)
        spent += cost
    return kept, len(chunks) - len(kept)


def build_prompt(question, retrieval_result, config=None, history=None):
    """
    The whole prompt as data. Returns (messages, used_chunks, dropped):
      messages     — [system, *history turns, user] for the chat-completions
                     call
      used_chunks  — the chunks that made it into the prompt, IN BLOCK ORDER,
                     so used_chunks[n-1] is what citation [n] refers to.
      dropped      — chunk count trimmed by the budget

    history — optional list of prior {"role": "user"|"assistant", "content"}
    turns from the CURRENT conversation, oldest first (e.g. loaded from the
    conversation's message rows). Only the last config.history_window are
    used, inserted as REAL alternating turns between the system prompt and
    the final context+question message — not text pasted into the user
    message, because the chat-completions API treats roles differently from
    quoted text, and that's what the roles are for. The chat API is
    stateless, so this window has to be resent on every call; it's what
    makes a multi-turn conversation feel continuous. Its estimated token
    cost is deducted from the context budget BEFORE chunks are trimmed, so a
    long recency window can't silently blow past the prompt size — the
    tradeoff is explicit, not accidental.

    MULTIMODAL: a figure/formula chunk (content_type figure|formula) may carry
    a resolved metadata['image_url'] (injected by api.py from Supabase Storage —
    this function never touches Storage itself, contract #5). When any used
    chunk has one, the final user message content becomes the OpenAI-style list
    form: the text block, then up to config.max_images `image_url` blocks in
    block order, so the vision model actually SEES the figure whose caption it
    already read in the numbered context. No image → the content stays a plain
    string, byte-identical to the text-only path (every existing caller and
    test_rag.py's assertions are unaffected).

    Pure function: same inputs, same prompt, no side effects — which is what
    lets test_rag.py assert on it and lets prompt wording be iterated against
    one cached retrieval result without re-running retrieval or the model.
    """
    config = config or GenerationConfig()

    recent_history = (history or [])[-config.history_window:]
    history_tokens = sum(estimate_tokens(m["content"]) for m in recent_history)
    context_budget = max(config.context_token_budget - history_tokens, 0)

    used, dropped = trim_to_budget(retrieval_result["chunks"], context_budget)

    system = SYSTEM_PROMPT
    if retrieval_result["low_confidence"]:
        system += LOW_CONFIDENCE_ADDENDUM

    blocks = "\n\n".join(format_block(i, h) for i, h in enumerate(used, start=1))
    user = f"Context:\n{blocks}\n\nQuestion: {question}"

    messages = [{"role": "system", "content": system}]
    messages.extend(recent_history)
    messages.append({"role": "user", "content": _user_content(user, used, config)})
    return messages, used, dropped


def _user_content(text, used, config):
    """The user message content: a plain string normally, or the OpenAI-style
    [text, image_url...] list when used chunks carry resolved image URLs.
    Images are attached highest-scored-first (used is score-sorted), capped at
    config.max_images, and the text names which blocks they map to so the model
    can align image N to context block [n]."""
    image_blocks = [(i, h) for i, h in enumerate(used, start=1)
                    if h["metadata"].get("image_url")][:config.max_images]
    if not image_blocks:
        return text

    refs = ", ".join(f"[{n}]" for n, _ in image_blocks)
    text += (f"\n\nThe attached image(s) are the visuals for context block(s) "
             f"{refs}, in that order. Read them to answer. If one is a formula, "
             f"transcribe it as LaTeX ($...$ or $$...$$), not plain text. If "
             f"one is a diagram/figure, describe it in prose — do not redraw "
             f"it as an ASCII diagram; the real image is already shown to the "
             f"user next to your citation.")
    content = [{"type": "text", "text": text}]
    for _, hit in image_blocks:
        content.append({"type": "image_url",
                        "image_url": {"url": hit["metadata"]["image_url"]}})
    return content


# --------------------------------------------------------------------------- #
# citation handling — pure functions
# --------------------------------------------------------------------------- #

CITATION_RE = re.compile(r"\[(\d+)\]")


def parse_citations(answer_text, n_blocks):
    """
    Pull every [n] out of the answer. Returns (valid, phantom) — sorted,
    de-duplicated. A phantom is a citation to a block that was never sent
    ([4] when there were 3): a groundedness failure we surface loudly rather
    than strip, because hiding it converts a visible model failure into an
    invisible one.
    """
    seen = {int(m) for m in CITATION_RE.findall(answer_text)}
    valid = sorted(n for n in seen if 1 <= n <= n_blocks)
    phantom = sorted(n for n in seen if not (1 <= n <= n_blocks))
    return valid, phantom


def resolve_citations(valid, used_chunks):
    """Map block numbers back to human-readable provenance — the payoff of
    Day 1-2's page tracking: a checkable 'open paper X at page Y'.
    source_type ('corpus' | 'upload') lets the UI mark a citation as the
    user's own uploaded paper rather than a shared corpus one — same field
    ingest.py already stamps on every chunk (see documents.py)."""
    sources = []
    for n in valid:
        m = used_chunks[n - 1]["metadata"]
        pages = (f"p{m['page_start']}" if m["page_start"] == m["page_end"]
                 else f"p{m['page_start']}-{m['page_end']}")
        sources.append({
            "n": n,
            "paper_id": m.get("paper_id", "?"),
            "title": m.get("title", ""),
            "pages": pages,
            "content_type": m["content_type"],
            "source_type": m.get("source_type", "corpus"),
            "image_path": m.get("image_path"),   # set for figure/formula chunks; the UI renders it
            "score": used_chunks[n - 1]["score"],
        })
    return sources


# --------------------------------------------------------------------------- #
# the LLM edge — the only non-deterministic, networked part
# --------------------------------------------------------------------------- #

def make_client():
    """Fail early and helpfully if the key isn't configured."""
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and paste "
            "your key (free at https://console.groq.com/keys)."
        )
    # pyrefly: ignore [missing-import]
    from groq import Groq
    return Groq()


def call_llm(messages, config, client, model=None, temperature=None, max_tokens=None):
    """
    One chat-completions call with polite 429 handling: exponential backoff
    (2s, 4s), a bounded number of retries, then a clean failure. A 429 is the
    service saying "slower", not a bug — same etiquette as fetch_corpus.py's
    arXiv delay, different protocol. Anything else (timeout, auth) propagates
    immediately: retrying won't fix those and would just hide them.

    model overrides config.model — answer() passes the vision model when the
    prompt carries images. temperature/max_tokens override config's the same
    way — rewrite_query() wants deterministic, short output, not the answer's
    own generation settings.
    """
    # pyrefly: ignore [missing-import]
    from groq import RateLimitError

    model = model or config.model
    temperature = config.temperature if temperature is None else temperature
    max_tokens = config.max_tokens if max_tokens is None else max_tokens
    attempt = 0
    while True:
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except RateLimitError:
            if attempt >= config.max_retries:
                raise
            delay = config.retry_base_delay * (2 ** attempt)
            print(f"  (rate limited — waiting {delay:.0f}s, "
                  f"retry {attempt + 1}/{config.max_retries})")
            time.sleep(delay)
            attempt += 1


REWRITE_SYSTEM_PROMPT = (
    "Rewrite the user's latest message as a standalone question that makes "
    "sense with no prior context, using the conversation history to resolve "
    "any pronouns or implicit references (\"this paper\", \"it\", \"that "
    "algorithm\", etc). If the message is already standalone, return it "
    "unchanged. Output ONLY the rewritten question — no preamble, no quotes, "
    "no explanation."
)


def rewrite_query(question, history, client, config=None):
    """
    Resolve conversational references BEFORE retrieval runs, not after.

    Retrieval only ever embeds/searches on `question` itself — it has no
    access to `history` the way build_prompt's generation step does (history
    is loaded straight from the messages table and inserted as real chat
    turns; retrieval never sees any of it, see retrieval.py's Retriever.
    retrieve()). So a follow-up like "explain the importance of this paper"
    gets searched on that literal text, with zero signal about which paper
    "this" refers to — an observed failure: the corpus match came from an
    unrelated paper, and the model correctly refused rather than hallucinate
    from it, but the refusal was only necessary because retrieval was blind
    to a reference generation already understood from history.

    Skips the LLM call entirely when there's no history — a conversation's
    first message can't contain a dangling reference, so the common case
    (a fresh, standalone question) pays nothing extra.

    Uses config.scratch_model, not config.model: this is a narrow rewriting
    task, not the answer itself, so it doesn't need the larger model's
    reasoning quality, and it sits on the request's critical path before
    generation even starts.

    Falls back to the original `question` on any Groq API failure (rate
    limit, timeout, transient error) rather than failing the whole request —
    this step is a retrieval-quality improvement, not a correctness
    requirement; degrading to today's already-correct behavior beats a hard
    500 over an optional preprocessing step.
    """
    if not history:
        return question

    # pyrefly: ignore [missing-import]
    from groq import GroqError

    config = config or GenerationConfig()
    transcript = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content":
            f"Conversation history:\n{transcript}\n\nLatest message: {question}"},
    ]
    try:
        response = call_llm(messages, config, client, model=config.scratch_model,
                             temperature=0.0, max_tokens=120)
        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten or question
    except GroqError:
        return question


def merge_extra_chunks(retrieval_result, extra_chunks):
    """
    Pure merge step: fold documents.py's per-user upload hits into a
    Retriever result, re-sorted by score. Extracted from answer() so it's
    assertion-testable like build_prompt/trim_to_budget, without a fake
    retriever or client. See answer()'s docstring for why this is valid
    (commensurable scores) and why any non-empty extra_chunks clears
    low_confidence.
    """
    if not extra_chunks:
        return retrieval_result
    merged = sorted(retrieval_result["chunks"] + extra_chunks,
                    key=lambda h: h["score"], reverse=True)
    return {"chunks": merged, "low_confidence": False}


def answer(question, retriever, client=None, config=None, history=None,
           extra_chunks=None, retrieval_result=None):
    """
    The full loop for one question: retrieve -> build_prompt -> Groq -> parse.

    Returns a dict with everything a caller (CLI now, FastAPI backend) needs,
    and everything a debugger needs to split any bad answer into "retrieval
    fetched wrong material" vs. "generation misused right material":
      answer            the model's text
      sources           resolved valid citations [{n, paper_id, title, pages, ...}]
      phantom_citations block numbers cited but never sent (groundedness alarm)
      chunks            what the model actually saw (post-trim, block order)
      dropped           chunks trimmed away by the token budget
      low_confidence    retrieval's own flag, passed through
      truncated         True if the answer hit max_tokens (finish_reason)
      usage             real token counts from the API (prompt/completion)

    history — optional recency window (see build_prompt), forwarded as-is.
    The caller owns loading it (e.g. the backend's last-N message rows for
    this conversation); this function doesn't know or care where it lives.

    extra_chunks — optional pre-scored hits from OUTSIDE the shared corpus
    (documents.py's per-user uploaded-paper search), same {'score', 'text',
    'metadata'} shape as a Retriever hit. Merged into retrieval_result by
    score before build_prompt runs — valid because both come from the same
    embedding model + cosine metric, so the scores are directly comparable
    (retrieval.py's own justification for merging text/table hits, reused
    here for merging corpus/upload hits). retriever.retrieve() itself is
    NOT touched: it stays "the shared corpus contract," and this function
    stays the one place results from different sources get combined.
    Each extra_chunk already cleared documents.py's own confidence
    threshold by construction, so any non-empty extra_chunks list is enough
    to lift retrieval's low_confidence flag even if the corpus alone came
    up weak — the corpus not knowing something doesn't mean the user's own
    paper doesn't.

    retrieval_result — optional pre-computed result (same shape retriever.
    retrieve() returns). When given, retriever.retrieve(question) is skipped
    entirely (no duplicate work — reranking already happened once, wherever
    this was computed). api.py uses this to resolve CORPUS figure/formula
    image_url from Storage BEFORE generation, the same treatment extra_chunks
    already gets — otherwise only upload-side figures would ever reach the
    vision model, not corpus ones. rag.py still never touches Storage itself;
    it just accepts chunks that may already carry a resolved image_url.
    When retrieval_result is NOT given, this function rewrites the query
    against `history` (see rewrite_query()) before retrieving — a caller
    that pre-computes retrieval_result (api.py) owns that step itself
    instead, since it needs the rewritten query for upload search too.
    """
    config = config or GenerationConfig()
    client = client or make_client()

    if retrieval_result is None:
        retrieval_query = rewrite_query(question, history, client, config)
        retrieval_result = retriever.retrieve(retrieval_query)
    retrieval_result = merge_extra_chunks(retrieval_result, extra_chunks)

    messages, used_chunks, dropped = build_prompt(question, retrieval_result,
                                                  config, history)

    # Switch to the vision model iff a figure/formula image actually made it
    # into the prompt (api.py resolved its URL). Text-only requests stay on the
    # cheaper text model — you only pay for vision when a visual is in context.
    uses_images = any(h["metadata"].get("image_url") for h in used_chunks)
    model = config.vision_model if uses_images else config.model
    response = call_llm(messages, config, client, model=model)

    choice = response.choices[0]
    text = choice.message.content or ""
    valid, phantom = parse_citations(text, len(used_chunks))

    return {
        "answer": text,
        "sources": resolve_citations(valid, used_chunks),
        "phantom_citations": phantom,
        "chunks": used_chunks,
        "dropped": dropped,
        "low_confidence": retrieval_result["low_confidence"],
        "truncated": choice.finish_reason == "length",
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
        "model": model,
    }


# --------------------------------------------------------------------------- #
# CLI — one-shot and REPL, always showing what the model saw
# --------------------------------------------------------------------------- #

def print_result(result):
    """Debug view first (what the model saw), then the answer, then resolved
    sources. Every bad answer immediately splits into a retrieval problem or
    a generation problem because both halves are on screen."""
    print("\n--- retrieved context (what the model saw) " + "-" * 29)
    flag = "  [LOW CONFIDENCE — floor fired]" if result["low_confidence"] else ""
    print(f"{len(result['chunks'])} block(s), {result['dropped']} trimmed by "
          f"budget{flag}")
    for i, hit in enumerate(result["chunks"], start=1):
        m = hit["metadata"]
        preview = " ".join(hit["text"].split())[:100]
        print(f"  [{i}] cos={hit['score']:.3f} [{m['content_type']:5s}] "
              f"{m.get('paper_id', '?')}  {preview}...")

    print("\n--- answer " + "-" * 61)
    print(result["answer"])

    if result["truncated"]:
        print("\n  !! answer hit max_tokens and was cut off mid-thought")
    if result["phantom_citations"]:
        print(f"\n  !! PHANTOM CITATIONS {result['phantom_citations']} — the "
              f"model cited blocks that were never sent. Groundedness failure;"
              f" treat this answer with suspicion.")

    if result["sources"]:
        print("\n--- sources " + "-" * 60)
        for s in result["sources"]:
            print(f"  [{s['n']}] {s['title'][:56]}  ({s['paper_id']}, "
                  f"{s['pages']}, {s['content_type']})")

    u = result["usage"]
    print(f"\n({u['prompt_tokens']} prompt + {u['completion_tokens']} "
          f"completion tokens, {result['model']})")


def main():
    from retrieval import load_default_retriever

    client = make_client()  # fail on a missing key BEFORE loading the model
    print("Loading index + embedding model...")
    retriever = load_default_retriever()

    if len(sys.argv) > 1:  # one-shot
        question = " ".join(sys.argv[1:])
        print_result(answer(question, retriever, client))
        return

    print('Ready. Ask about the corpus (blank line or Ctrl-D to quit).')
    while True:  # REPL
        try:
            question = input("\n?> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        try:
            print_result(answer(question, retriever, client))
        except Exception as e:
            print(f"  error: {e}")


if __name__ == "__main__":
    main()
