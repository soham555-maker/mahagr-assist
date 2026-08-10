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
from dataclasses import dataclass, field

from engine import config   # also loads backend/.env (GROQ_API_KEY, LLM_PROVIDER, ...)


@dataclass
class GenerationConfig:
    # LLM provider seam (see engine/config.py). "groq" (default) or "ollama".
    # default_factory reads the env at instantiation, so LLM_PROVIDER set in .env
    # (loaded at import) is honored, not just a shell export.
    provider: str = field(default_factory=lambda: os.environ.get("LLM_PROVIDER", config.LLM_PROVIDER).lower())
    ollama_base_url: str = field(default_factory=lambda: os.environ.get("OLLAMA_BASE_URL", config.OLLAMA_BASE_URL))
    ollama_model: str = field(default_factory=lambda: os.environ.get("OLLAMA_MODEL", config.OLLAMA_MODEL))   # one local model, all roles
    model: str = "llama-3.1-8b-instant"   # groq: high daily-token limit, fine for grounded QA
    vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"  # groq: figure/formula image context
    scratch_model: str = "llama-3.1-8b-instant"  # groq: cheap query rewrites
    temperature: float = 0.2   # low = boring and faithful; 0.0 for reproducible tests
    # Output cap. Also counts toward Groq's per-minute token limit, and on the
    # LOCAL model it is the single biggest determinant of worst-case latency:
    # decode runs at ~25-40 tok/s, so 512 tokens IS ~15-20 s on its own.
    # Resolved per provider in __post_init__.
    max_tokens: int = field(default=None)
    # HARD abstention floor on the top retrieval score. Below it, answer()
    # refuses WITHOUT calling the model at all (see _hard_abstention).
    #
    # WHY THIS IS NOT A PROMPT RULE. Abstention is FR 3.3.5 and it was being
    # left to a 3B model's willingness to refuse. Measured 2026-08-09, 5 repeats
    # per question at temperature 0.2:
    #     "How do I apply for a new passport in Mumbai?"   top score 0.006 -> 0/5 refused
    #     "PhD aerospace fee at IIT Bombay?"               top score 0.028 -> 4/5 refused
    # The passport question was answered with FABRICATED Marathi prose carrying
    # citations, while retrieval was already reporting 0.006 and
    # low_confidence=True. The signal was correct and available; only the model
    # ignored it. HANDOFF §5.10 records six prompt variants that each traded one
    # failure for another, so the fix is not a seventh — it is to stop asking.
    #
    # WHY 0.10: the cross-encoder scores measured on the gold set separate
    # cleanly at this end of the range — RELEVANT top hits have p10 = 0.946,
    # while genuinely out-of-corpus questions score 0.006-0.028. 0.10 sits two
    # orders of magnitude below the weakest real hit, so it fires only on
    # questions the corpus demonstrably cannot answer. It is deliberately NOT
    # near rerank_threshold (0.85): that gate decides which chunks are good
    # enough to SHOW, this one decides whether to speak at all, and conflating
    # them would abstain on answerable questions.
    #
    # Set to 0 to disable (restores the prompt-only behaviour).
    abstain_floor: float = field(
        default_factory=lambda: float(os.environ.get("LLM_ABSTAIN_FLOOR", "0.10")))
    # The context budget is capped by DIFFERENT things per provider, so one number
    # cannot serve both:
    #   groq   — a 6000 tokens/MINUTE rate limit. Small requests are the point.
    #   ollama — no rate limit at all; the only ceiling is the model's context
    #            WINDOW (config.OLLAMA_NUM_CTX), and unused window costs nothing.
    # Sharing groq's tiny budget starved the local model: a single ~1900-char
    # Marathi GR chunk priced out the whole 2200 allowance, so build_prompt sent
    # ONE block and dropped the other ten, and the model — given a GR header and
    # no body — answered with a bare "[1]". See tokens_per_devanagari below.
    context_token_budget: int = field(default=None)   # resolved in __post_init__
    history_token_budget: int = 800   # hard cap on the recency window (drops oldest turns)
    history_window: int = 4    # recency-window turns (2 exchanges) sent as real roles
    max_images: int = 3        # cap figure/formula images attached per request (tokens + rate)
    max_retries: int = 2       # extra attempts after a 429, with backoff
    retry_base_delay: float = 2.0  # seconds; doubles each retry
    # Devanagari tokens per character, which is TOKENIZER-specific — the whole
    # budget is denominated in it, so a wrong value silently starves or floods
    # the prompt. Measured, not guessed (see TOKENS_PER_DEVANAGARI).
    tokens_per_devanagari: float = field(default=None)   # resolved in __post_init__
    # The local model's context window; must match the Ollama server's
    # OLLAMA_CONTEXT_LENGTH or prompts get silently truncated (see config.py).
    num_ctx: int = field(default_factory=lambda: config.OLLAMA_NUM_CTX)
    # Use the short prompt written for a small local model. Defaults on for
    # ollama, off for groq — see SYSTEM_PROMPT_COMPACT for the measurement.
    compact_prompt: bool = field(default=None)   # resolved in __post_init__

    def __post_init__(self):
        """Fill the two provider-dependent numbers, unless the caller set them."""
        local = self.provider == "ollama"
        if self.context_token_budget is None:
            # ollama: sized to fit OLLAMA_NUM_CTX alongside the system prompt,
            # history and the reply. groq: unchanged in REAL tokens — the old
            # 2200 at the old (2.0) rate was ~1200 real Devanagari tokens, and
            # 1300 at the measured rate is the same volume, so the rate-limit
            # behaviour that budget existed to protect does not change.
            # 6000 measured better than 3500 on the gold set (CORRECT 19-20/20 vs
            # 17/20): a second and third block is often where the answer actually
            # is. It costs a few seconds of prompt processing, which is the right
            # trade while the p50 stays well inside the SRS's 10s.
            # The local number MUST fit inside num_ctx alongside the system
            # prompt, the history window and the reply, or Ollama silently drops
            # the oldest tokens — which are the system prompt carrying every
            # grounding and citation rule (HANDOFF §5.9). The arithmetic at
            # num_ctx=8192: 8192 - 768 reply - ~800 history - ~250 system and
            # reminder = ~6300 left for context blocks, so 6000 with margin.
            # Raise this ONLY together with OLLAMA_NUM_CTX and the server's
            # OLLAMA_CONTEXT_LENGTH; warn_if_over_context() checks the sum.
            self.context_token_budget = int(os.environ.get("LLM_CONTEXT_BUDGET")
                                            or (6000 if local else 4000))
        if self.max_tokens is None:
            # LOCAL: decode is the dominant cost of a request (measured: prefill
            # runs at ~4,200 tok/s, decode at ~25-40 tok/s), so the output cap
            # is what bounds the WORST case against the SRS's 10 s — an
            # uncapped 512 is ~15-20 s of decoding by itself. A grounded, cited
            # answer measured a median of 140 completion tokens, so 500 leaves
            # ~3.5x headroom while still bounding the tail (~12 s worst-case
            # decode). 320 was too tight: it truncated 4/20 gold answers
            # mid-sentence, costing their citations and dropping CORRECT by 2.
            # 768: a detailed grounded answer measures ~200-350 completion
            # tokens, so this is ~2-3x headroom and only bites on a runaway.
            # It is also the SRS's 10 s NFR in disguise — decode runs at ~25-40
            # tok/s, so the cap IS the worst-case latency: 768 is ~20-30 s worst
            # case, 2048 would be ~50-80 s for a single answer.
            self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS")
                                  or (768 if local else 1024))
        if self.tokens_per_devanagari is None:
            self.tokens_per_devanagari = (TOKENS_PER_DEVANAGARI_QWEN if local
                                          else TOKENS_PER_DEVANAGARI)
        if self.compact_prompt is None:
            self.compact_prompt = local


TOKENS_PER_WORD = 1.33          # Latin heuristic
# Llama/Groq byte-BPEs Devanagari at ~2 tokens/char — this is the constant that
# explained the 6629-token request against a 6000/min cap (HANDOFF §5.5).
TOKENS_PER_DEVANAGARI = 2.0
# qwen2.5 has real Devanagari vocabulary, so it is ~half as dense. MEASURED on
# qwen2.5:3b via the reported prompt_tokens: 1.09 tokens/char over Marathi GR
# prose; 1.2 is used to keep a margin. Applying Llama's 2.0 here over-counted
# Marathi by 1.7x, which is what silently reduced the context to a single chunk.
# This constant is a MEASUREMENT, not a dial — the budget below is the dial.
# Inflating it does not "leave more room", it makes every chunk look bigger than
# it is, so fewer of them fit and the model is starved of exactly the context a
# detailed answer needs.
TOKENS_PER_DEVANAGARI_QWEN = 1.2

_DEVA_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_RE = re.compile(r"[A-Za-z0-9]+")


def estimate_tokens(text, tokens_per_devanagari=TOKENS_PER_DEVANAGARI):
    """Approximate the LLM token count. Critical for Marathi: a plain English
    words×1.33 estimate under-counts Devanagari several-fold, which is how a
    "small" context blew past Groq's 6000-token/min cap.

    The Devanagari rate is per-TOKENIZER and the caller must pass the one for the
    model in use (GenerationConfig.tokens_per_devanagari): Llama/Groq byte-BPEs
    it at ~2.0 tokens/char, qwen2.5 has real Devanagari vocabulary at ~1.1. The
    default stays at Llama's higher figure so an un-configured call over-counts
    rather than under-counts — over-counting only wastes context, while
    under-counting means a rate-limit 429 or a silently truncated prompt."""
    deva = len(_DEVA_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    return int(latin * TOKENS_PER_WORD + deva * tokens_per_devanagari)


# --------------------------------------------------------------------------- #
# prompt assembly — pure functions, no API, no I/O
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """\
You are an expert assistant for Maharashtra Government officers.
Answer using ONLY the provided numbered context blocks.

Give a full, substantive answer. State what the provision actually SAYS — the
amounts, dates, eligibility and conditions — and explain it. Naming the GR that
contains the answer is not an answer; the officer already has the GR list.

- Cite the block each fact came from, like [1]. Cite only blocks you were given.
- Copy GR numbers, dates and amounts EXACTLY as written. Never invent a number.
- If two blocks conflict — one supersedes or amends the other — say so and cite
  both, and say which appears to be in force.
- If the blocks genuinely do not contain the answer, say that plainly and cite
  nothing. A clear refusal is a correct answer.
- Reply in the same language as the question, keeping official Marathi
  terminology in Marathi."""

# The same rules as SYSTEM_PROMPT, compressed for a SMALL local model.
#
# WHY A SECOND PROMPT EXISTS: prompt length is not free on a 3B model. Measured
# on qwen2.5:3b with identical retrieved context, one question:
#   full SYSTEM_PROMPT + rules  ->   4 completion tokens, output was "[1]"
#   a two-line prompt           ->  99 completion tokens, a correct cited answer
# The instructions were not being disobeyed so much as drowned: ~350 words of
# prose, most of it about what NOT to do, collapses a small model into emitting
# the one token it is sure about — the citation. A large model reads the same
# text and behaves better for it, which is why the long version is kept and
# still used on groq. Same requirements, different budget for saying them.
#
# "Different budget for saying them" is not licence to drop one. A 2026-08-07
# rewrite shortened rules 4 and 5 and silently lost BOTH the conflict rule
# (SRS FR 3.3.3) and "cite nothing" — nothing failed except
# test_both_system_prompts_keep_the_four_grounding_rules, which is exactly why
# that test exists. Shorten the wording; keep the four properties.
SYSTEM_PROMPT_COMPACT = """\
You answer questions for Maharashtra Government officers using ONLY the numbered context blocks provided.

1. State the actual provision (amounts, dates, eligibility). "This is covered in GR X" is not an answer.
2. Cite the block after each fact using exactly the format [1]. Do not write "Block [1] states", just state the fact and append [1].
3. Copy GR numbers, dates and amounts EXACTLY. Never invent a number.
4. If blocks conflict — one supersedes or amends another — say so, cite both, and say which appears to be in force.
5. If the blocks do not answer the question, say so plainly and cite nothing. If they are merely off-topic, briefly describe what they ARE about and cite them.
6. You MUST reply in the exact same language as the question. Keep official Marathi terms in Marathi."""

# Appended to the system prompt when retrieval itself flagged the match as weak.
# It must NOT read as "try harder" — retrieval has already decided nothing scored
# as a confident match, and a hedged instruction measurably had the model stretch
# unrelated blocks into a confident-sounding answer. It asks for the honest shape
# instead: say it isn't covered, then say what the nearby material IS about,
# which is genuinely useful without being a guess presented as fact.
LOW_CONFIDENCE_ADDENDUM = """

IMPORTANT: retrieval matched this question only WEAKLY — the blocks below are the
closest chunks found, but nothing scored as a confident match. If they do not
actually answer the question, say that the available Government documents do not
appear to cover it. Then briefly describe what the closest blocks ARE about,
citing [n] for each fact you state — an officer needs to see which GR you are
describing. Do not stretch weak context into a confident answer."""


def final_reminder_compact(n_blocks, low_confidence=False):
    """One line, for the small local model, placed AFTER the question.

    Kept to a single sentence on purpose: measured on qwen2.5:3b, ~350 words of
    mostly-prohibitive rules collapsed the reply to a bare "[1]" (4 completion
    tokens). The guardrails live in SYSTEM_PROMPT_COMPACT; this only restates
    the two things the model must get right in its very last instruction —
    answer with substance, and stay inside the block range it was given.
    """
    span = "[1]" if n_blocks == 1 else f"[1]-[{n_blocks}]"
    if low_confidence:
        # Retrieval flagged a weak match: the blocks may not answer the question,
        # but an officer still needs to see WHICH GR the system is describing —
        # an uncited paraphrase is useless. Cite the blocks you reference; just
        # don't pretend a weak match is a confident answer.
        return (f"These blocks scored as weak matches. If they do not directly "
                f"answer the question, say so, then describe what the closest "
                f"blocks ARE about — and cite {span} for each fact you state. "
                f"Do not present a guess as a confident answer.")
    # The language rule is repeated HERE, not left to the system prompt alone.
    # Measured: with it only in SYSTEM_PROMPT_COMPACT, a Marathi question got an
    # English answer — the recency effect that makes this reminder work at all
    # (HANDOFF §5.10) also means a rule absent from it is the one that gets
    # dropped. Costs one clause.
    return (f"Answer in full, IN THE SAME LANGUAGE AS THE QUESTION: state the "
            f"actual provision — amounts, dates, conditions — not just which GR "
            f"contains it. Cite {span} after each fact, and cite nothing if the "
            f"blocks do not answer the question.")


def final_reminder(n_blocks, low_confidence=False):
    """The rules restated after the context and before the question, for a large
    model. It can carry more rules than the compact version without collapsing,
    but the two failure modes are the same and both are pinned here: citing a
    block number that was never sent, and answering anyway when the blocks do
    not cover the question.
    """
    span = "[1]" if n_blocks == 1 else f"[1] to [{n_blocks}]"
    rules = [
        "Rules for your answer:",
        f"1. You were given {n_blocks} numbered context block(s): {span}. "
        f"Cite ONLY those numbers — never a number outside that range.",
        "2. Answer with substance: give the actual amounts, dates, eligibility "
        "and conditions and explain them. Naming the GR is not an answer.",
        "3. Copy GR numbers, dates and amounts exactly as written; never invent "
        "a number.",
        "4. If the blocks do not actually answer the question, say plainly that "
        "the documents do not cover it. Then describe what the closest blocks "
        "ARE about, citing [n] for each fact — the officer needs provenance "
        "even for a weak match. Do not present this as a confident answer.",
    ]
    if low_confidence:
        rules.append("5. Retrieval matched this question only WEAKLY, so rule 4 "
                     "is the most likely correct response here. Still cite the "
                     "blocks you describe.")
    return "\n".join(rules)


# The portal's language toggle sends ISO codes ("en"/"mr"), but "Write the answer
# in mr." is a poor instruction — models follow a real language NAME far better,
# and a small one barely follows the code at all. Marathi/Hindi are given in
# their own script too, which is the strongest signal of all. Anything unknown is
# passed through unchanged, so "simple English" still works.
LANGUAGE_NAMES = {"en": "English", "mr": "Marathi (मराठी)", "hi": "Hindi (हिंदी)"}


def language_directive(language):
    """One line pinning the answer's language, or "" for "auto" (which leaves
    the system prompt's match-the-question's-language rule in charge). Shared by
    rag.build_prompt and officer._generate so the toggle behaves identically on
    /ask, /summarize, /explain and /compare."""
    if not language or language == "auto":
        return ""
    name = LANGUAGE_NAMES.get(language.strip().lower(), language)
    return f"\nWrite the answer in {name}."


def format_block(n, hit):
    """One numbered context block with a provenance header. The header is what
    makes a later [n] citation resolvable to a source — and it also tells the
    model what it's reading (a table row rendering vs. prose). For GRs it
    carries the GR number and date, so the model can cite them precisely and
    reason about which resolution is newer / supersedes which."""
    m = hit["metadata"]
    pages = (f"page {m['page_start']}" if m["page_start"] == m["page_end"]
             else f"pages {m['page_start']}-{m['page_end']}")
    title = m.get("title") or m.get("source_file", "unknown")
    # Prefer the GR number as the document id; fall back to paper_id / filename.
    doc_id = m.get("gr_number") or m.get("paper_id") or m.get("source_file", "?")
    ident = " — ".join(p for p in (title, doc_id, m.get("date")) if p)
    return (f"[{n}] ({ident}, {pages}, {m['content_type']})\n{hit['text']}")


def trim_to_budget(chunks, budget, tokens_per_devanagari=TOKENS_PER_DEVANAGARI):
    """
    Keep as many chunks as fit in the estimated token budget, dropping the
    LOWEST-scoring first (chunks arrive sorted by score desc, so we cut from
    the tail). Never drops below one block: an over-budget single chunk still
    goes through, because an empty prompt is worse than a long one.
    Returns (kept_chunks, dropped_count).
    """
    kept, spent = [], 0
    for hit in chunks:
        cost = estimate_tokens(hit["text"], tokens_per_devanagari)
        if kept and spent + cost > budget:
            break  # sorted desc => everything after is lower-scoring
        kept.append(hit)
        spent += cost
    return kept, len(chunks) - len(kept)


def build_prompt(question, retrieval_result, config=None, history=None,
                 language="auto"):
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

    tpd = config.tokens_per_devanagari
    recent_history = (history or [])[-config.history_window:]
    # Hard-cap the history tokens (drop oldest turns) so a long Marathi thread
    # can't push the request past Groq's per-minute limit.
    while recent_history and sum(estimate_tokens(m["content"], tpd) for m in recent_history) > config.history_token_budget:
        recent_history = recent_history[1:]
    history_tokens = sum(estimate_tokens(m["content"], tpd) for m in recent_history)
    context_budget = max(config.context_token_budget - history_tokens, 0)

    used, dropped = trim_to_budget(retrieval_result["chunks"], context_budget, tpd)

    system = SYSTEM_PROMPT_COMPACT if config.compact_prompt else SYSTEM_PROMPT
    if retrieval_result["low_confidence"] and not config.compact_prompt:
        # the compact path folds the weak-retrieval steer into its one-line
        # reminder instead, so the small model still reads only one instruction
        system += LOW_CONFIDENCE_ADDENDUM

    blocks = "\n\n".join(format_block(i, h) for i, h in enumerate(used, start=1))
    # language: "auto" (default) leaves the system prompt's "answer in the
    # question's language" rule in charge; an explicit value is the portal's
    # language toggle, which must win — that's FR 3.4.2, and FR 3.4.5 (switching
    # language mid-conversation) is just this changing between turns.
    lang = language_directive(language)
    low_conf = retrieval_result["low_confidence"]
    if config.compact_prompt:
        # Instruction LAST, after the question — the arrangement that measured
        # 99 completion tokens against 4 for the rules-before-question layout.
        user = (f"Context:\n{blocks}\n\nQuestion: {question}\n"
                f"{final_reminder_compact(len(used), low_conf)}{lang}")
    else:
        user = (f"Context:\n{blocks}\n\n{final_reminder(len(used), low_conf)}"
                f"{lang}\n\nQuestion: {question}")

    messages = [{"role": "system", "content": system}]
    messages.extend(recent_history)
    messages.append({"role": "user", "content": _user_content(user, used, config)})
    return messages, used, dropped


def warn_if_over_context(messages, config):
    """Loudly flag a prompt that the LOCAL model's context window cannot hold.

    Ollama does not reject an over-long prompt — it silently drops the oldest
    tokens, i.e. the system prompt carrying every grounding and citation rule.
    The answer still comes back and still looks plausible, so this failure is
    invisible from the outside; only the quality quietly collapses. Since the
    context budget is now large enough to matter, that has to be observable.

    Returns the estimated prompt size so callers/tests can assert on it.
    """
    if config.provider != "ollama":
        return 0
    total = sum(estimate_tokens(m["content"], config.tokens_per_devanagari)
                for m in messages if isinstance(m.get("content"), str))
    headroom = config.num_ctx - config.max_tokens
    if total > headroom:
        print(f"WARNING: prompt ~{total} tokens exceeds the usable window "
              f"({config.num_ctx} ctx - {config.max_tokens} reply = {headroom}). "
              f"Ollama will TRUNCATE it silently, dropping the system prompt. "
              f"Lower context_token_budget or raise OLLAMA_NUM_CTX (and the "
              f"server's OLLAMA_CONTEXT_LENGTH).", file=sys.stderr)
    return total


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
            # GR-domain provenance (present when ingest parsed it; None otherwise)
            # `doc` is the id the officer can actually ACT on: it is what
            # /documents/{id}/text and /graph/{id} take, so a citation in the UI
            # becomes a working "open / download this GR" link (SRS FR 3.7.4)
            # instead of a dead label. Falls back to the source filename's stem
            # for the flat/fixture index, which has no order_id.
            "doc": m.get("order_id") or str(m.get("source_file", "")).split(".")[0] or None,
            "gr_number": m.get("gr_number"),
            "date": m.get("date"),
            "department": m.get("department"),
            "language": m.get("language"),
            "source_file": m.get("source_file"),
        })
    return sources


# --------------------------------------------------------------------------- #
# the LLM edge — the only non-deterministic, networked part
# --------------------------------------------------------------------------- #

# Penalty applied to tokens already emitted, to break the small model's
# repeat-the-paragraph loop. 0.3 is deliberately mild: high values make a model
# avoid re-using words it legitimately needs, and a GR answer must repeat exact
# terms (a GR number, "अनुसूचित जमाती", an amount) rather than paraphrase them.
OLLAMA_FREQUENCY_PENALTY = float(os.environ.get("OLLAMA_FREQUENCY_PENALTY", "0.3"))


class OllamaClient:
    """Minimal LOCAL LLM client via Ollama's OpenAI-compatible endpoint
    (POST {base}/v1/chat/completions), using only the standard library — no
    cloud SDK, nothing leaves the machine. Exposes the same
    `.chat.completions.create(...)` surface and OpenAI-shaped result
    (`.choices[0].message.content`, `.choices[0].finish_reason`, `.usage.*`)
    that the Groq client does, so call_llm/answer treat the two identically."""

    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")
        self.chat = _Namespace(completions=_Namespace(create=self._create))

    def placement(self):
        """Where the loaded model is actually RUNNING — `{'model', 'processor',
        'context'}` — or None if nothing is loaded / Ollama is unreachable.

        This exists because of a silent 4x regression. Ollama decides how many
        layers fit on the GPU AT LOAD TIME, from the VRAM free at that instant.
        Load it while the API already holds the reranker on a 6 GB card and it
        quietly puts ~15% of the layers on the CPU — which costs nothing at
        prefill (2400 tok/s either way) but drops DECODE from 41 to 10 tok/s,
        and decode is what a 100-token answer is made of. Nothing errors; the
        system is just 4x slower. Measured, then surfaced on /health so it can
        never be invisible again. Start Ollama BEFORE the API.
        """
        import json
        import urllib.request
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/ps", timeout=2) as r:
                models = (json.loads(r.read()) or {}).get("models") or []
        except Exception:
            return None
        if not models:
            return None
        m = models[0]
        total = m.get("size") or 0
        gpu = m.get("size_vram") or 0
        pct = round(100.0 * gpu / total) if total else 0
        return {"model": m.get("name"),
                "processor": "100% GPU" if pct >= 100 else f"{100 - pct}% CPU / {pct}% GPU",
                "fully_on_gpu": pct >= 100,
                "context": m.get("context_length")}

    def _create(self, model, messages, temperature, max_tokens):
        import json
        import urllib.request
        # frequency_penalty guards the REPETITION LOOP, which is the small
        # quantized model's characteristic failure: asked for a detailed answer
        # it writes a good paragraph, then restates it almost verbatim until it
        # hits max_tokens. Measured on a Marathi scholarship question: the reply
        # burned all 768 completion tokens on one paragraph said twice (17.5 s,
        # finish_reason=length). It is a decode-time penalty on tokens already
        # emitted, so it costs nothing and does not touch the grounding rules —
        # unlike shortening max_tokens, which would just truncate mid-sentence
        # and cost the citation at the end.
        body = json.dumps({
            "model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "stream": False,
            "frequency_penalty": OLLAMA_FREQUENCY_PENALTY,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                payload = json.loads(r.read())
        except Exception as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url} ({e}). Is it running "
                f"(`ollama serve`) and is the model pulled (`ollama pull <model>`)?"
            ) from e
        return self._to_response(payload)

    @staticmethod
    def _to_response(payload):
        """OpenAI-shaped dict -> attribute object matching the Groq result.
        Pure and testable without a live server."""
        choice = (payload.get("choices") or [{}])[0]
        usage = payload.get("usage") or {}
        return _Namespace(
            choices=[_Namespace(
                message=_Namespace(content=(choice.get("message") or {}).get("content", "")),
                finish_reason=choice.get("finish_reason", "stop"),
            )],
            usage=_Namespace(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            ),
        )


class _Namespace:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def make_client(gen_config=None):
    """Build the LLM client for the configured provider. groq (default) needs
    GROQ_API_KEY; ollama needs nothing (a local server) — the on-prem path."""
    gen_config = gen_config or GenerationConfig()
    if gen_config.provider == "ollama":
        return OllamaClient(gen_config.ollama_base_url)
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and paste your "
            "key (free at https://console.groq.com/keys) — or set LLM_PROVIDER="
            "ollama to run a local model instead."
        )
    # pyrefly: ignore [missing-import]
    from groq import Groq
    return Groq()


def call_llm(messages, config, client, model=None, temperature=None, max_tokens=None):
    """
    One chat-completions call. On groq, polite 429 handling: exponential backoff
    (2s, 4s), bounded retries, then clean failure — a 429 means "slower", not a
    bug. Anything else (timeout, auth) propagates immediately. On ollama there's
    no rate limit, so the call is made directly.

    model overrides config.model (answer() passes the vision model for image
    context; rewrite_query() the scratch model) — but ONLY on groq. On ollama
    there is one local model (config.ollama_model) used for every role.
    """
    temperature = config.temperature if temperature is None else temperature
    max_tokens = config.max_tokens if max_tokens is None else max_tokens

    if config.provider == "ollama":
        return client.chat.completions.create(
            model=config.ollama_model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )

    # pyrefly: ignore [missing-import]
    from groq import RateLimitError
    model = model or config.model
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
    # Only the last couple of turns are needed to resolve a pronoun/reference,
    # and keeping the transcript short keeps this call under Groq's per-minute
    # token cap even in a long Marathi thread.
    recent = history[-config.history_window:]
    transcript = "\n".join(f"{turn['role']}: {turn['content']}" for turn in recent)
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


# The refusal returned by the hard gate. Deliberately says only two things: the
# documents do not cover this, and what to do next. It cites NOTHING, because
# the whole premise of firing is that nothing retrieved is relevant — a citation
# here would be the exact phantom-provenance failure the system exists to avoid.
ABSTAIN_MESSAGES = {
    "en": ("The available Government Resolutions do not appear to cover this "
           "question. Nothing in the indexed corpus matched it closely enough "
           "to answer from. Try rephrasing with the scheme or department name, "
           "or the document may not be in this corpus."),
    "mr": ("उपलब्ध शासन निर्णयांमध्ये या प्रश्नाची माहिती आढळत नाही. "
           "निर्देशांकित संग्रहातील कोणताही दस्तऐवज या प्रश्नाशी पुरेसा जुळत नाही. "
           "कृपया योजनेचे किंवा विभागाचे नाव वापरून प्रश्न पुन्हा विचारा, "
           "अथवा सदर दस्तऐवज या संग्रहात नसावा."),
}


def _abstain_language(question, language):
    """Which language to refuse in.

    An explicit portal toggle wins. Otherwise detect from the QUESTION, not the
    retrieved text: a refusal means the retrieved text is irrelevant, so its
    language says nothing about the officer's. Measured failure this prevents:
    an English passport question was answered in Marathi.
    """
    lang = (language or "auto").lower()
    if lang.startswith("mr") or "marathi" in lang or "मराठी" in lang:
        return "mr"
    if lang.startswith("en") or "english" in lang:
        return "en"
    return "mr" if _DEVA_RE.search(question or "") else "en"


def _hard_abstention(question, retrieval_result, config, language):
    """A refusal dict when retrieval scored below config.abstain_floor, else None.

    Returns the SAME shape as answer() so callers need no special case. Note
    sources=[] and phantom_citations=[]: an abstention is trivially grounded,
    and counting it as such is correct rather than generous — it asserted
    nothing, so it invented nothing.
    """
    if not config.abstain_floor:
        return None
    chunks = retrieval_result.get("chunks") or []
    # No chunks at all (e.g. a filter matched nothing) is the strongest possible
    # signal, and max() would raise on an empty sequence.
    top = max((c.get("score") or 0.0 for c in chunks), default=0.0)
    if top >= config.abstain_floor:
        return None
    return {
        "answer": ABSTAIN_MESSAGES[_abstain_language(question, language)],
        "sources": [],
        "phantom_citations": [],
        "chunks": [],
        "dropped": [],
        "low_confidence": True,
        "abstained": True,
        "top_score": round(top, 4),
        "truncated": False,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "model": None,
    }


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
           extra_chunks=None, retrieval_result=None, language="auto",
           filters=None):
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

    filters — optional {'departments', 'date_from', 'date_to', 'language'} that
    scopes retrieval to part of the corpus (PLAN Phase 2). Forwarded to
    retrieve() only when set, so the flat backend — whose retrieve() takes no
    filters — is untouched.

    language — "auto" (answer in the question's own language) or an explicit
    language name from the portal's toggle, forwarded to build_prompt. Because
    it is per-call rather than per-conversation, an officer can switch language
    mid-thread and the next answer follows (FR 3.4.2 / FR 3.4.5).

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
    client = client or make_client(config)

    # Per-stage wall clock. The SRS gives a 10 s budget (NFR Performance), and
    # which stage spends it was until now INFERRED — from the fact that
    # abstentions came back faster than full answers. Three time() calls turn
    # that guess into a number, and the number is what decides whether to tune
    # retrieval or generation.
    t0 = time.time()
    rewrite_s = 0.0

    if retrieval_result is None:
        retrieval_query = rewrite_query(question, history, client, config)
        rewrite_s = time.time() - t0
        # `filters` is passed ONLY when set: the flat Retriever's retrieve()
        # takes no such argument, and the scale backend is a config switch that
        # must not become a hard dependency of the generation path.
        retrieval_result = (retriever.retrieve(retrieval_query, filters=filters)
                            if filters else retriever.retrieve(retrieval_query))
    retrieval_result = merge_extra_chunks(retrieval_result, extra_chunks)
    retrieval_s = time.time() - t0 - rewrite_s

    # Hard abstention gate — BEFORE generation, on purpose. Two reasons it sits
    # here rather than in the prompt: it makes FR 3.3.5 deterministic instead of
    # dependent on a 3B model's mood, and it skips the whole generation step,
    # which is ~91% of request latency. Refusing gets FASTER, not slower.
    refusal = _hard_abstention(question, retrieval_result, config, language)
    if refusal is not None:
        refusal["timings"] = {
            "rewrite": round(rewrite_s, 3),
            "retrieval": round(retrieval_s, 3),
            "generation": 0.0,
            "total": round(time.time() - t0, 3),
        }
        return refusal

    messages, used_chunks, dropped = build_prompt(question, retrieval_result,
                                                  config, history, language)
    warn_if_over_context(messages, config)
    t_gen = time.time()

    # Switch to the vision model iff a figure/formula image actually made it
    # into the prompt (api.py resolved its URL). Text-only requests stay on the
    # cheaper text model — you only pay for vision when a visual is in context.
    # (On ollama there's one local model; call_llm ignores this and reports it.)
    uses_images = any(h["metadata"].get("image_url") for h in used_chunks)
    model = config.vision_model if uses_images else config.model
    response = call_llm(messages, config, client, model=model)
    generation_s = time.time() - t_gen
    if config.provider == "ollama":
        model = config.ollama_model

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
        # Always present, so a caller never has to distinguish "did not abstain"
        # from "this build predates the gate".
        "abstained": False,
        "truncated": choice.finish_reason == "length",
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
        # Seconds per stage, against the SRS's 10 s. `rewrite` is a second LLM
        # call (the follow-up resolver) and is billed separately from
        # `generation` on purpose — it is easy to forget that one question can
        # mean two round-trips to the model.
        "timings": {
            "rewrite": round(rewrite_s, 3),
            "retrieval": round(retrieval_s, 3),
            "generation": round(generation_s, 3),
            "total": round(time.time() - t0, 3),
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
            doc_id = s.get("gr_number") or s.get("source_file") or s["paper_id"]
            date = f", {s['date']}" if s.get("date") else ""
            print(f"  [{s['n']}] {s['title'][:56]}")
            print(f"       {doc_id}{date}  ({s['pages']}, {s['content_type']})")

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
