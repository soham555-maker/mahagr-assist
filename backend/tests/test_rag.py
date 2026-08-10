"""Grounding helpers: citation parsing, budget trimming, block/citation formatting."""

from engine import rag


def _hit(text, score=1.0, **meta):
    base = {"content_type": "text", "page_start": 1, "page_end": 1}
    base.update(meta)
    return {"text": text, "score": score, "metadata": base}


def test_parse_citations_valid_and_phantom():
    valid, phantom = rag.parse_citations("As in [1] and [3], but [5] does not exist.", n_blocks=3)
    assert valid == [1, 3]
    assert phantom == [5]  # cited a block that was never sent — a groundedness alarm


def test_trim_to_budget_drops_lowest_scoring_tail():
    chunks = [_hit("word " * 100, score=0.9), _hit("word " * 100, score=0.5)]
    kept, dropped = rag.trim_to_budget(chunks, budget=140)  # ~1 chunk fits
    assert len(kept) == 1 and dropped == 1
    assert kept[0]["score"] == 0.9  # highest kept


def test_trim_to_budget_never_empties():
    kept, _ = rag.trim_to_budget([_hit("w " * 1000, score=0.9)], budget=1)
    assert len(kept) == 1  # an over-budget single chunk still goes through


def test_format_block_includes_gr_number_and_date():
    header = rag.format_block(1, _hit("body", gr_number="संकीर्ण-२०२४/x", date="2024-04-10", title="Fee GR"))
    assert header.startswith("[1]")
    assert "संकीर्ण-२०२४/x" in header and "2024-04-10" in header


def test_language_directive_maps_portal_codes_to_names():
    # The portal toggle sends ISO codes; "Write the answer in mr." is a much
    # weaker instruction than naming the language, especially for a small model.
    assert rag.language_directive("auto") == ""
    assert rag.language_directive(None) == ""
    assert "Marathi" in rag.language_directive("mr")
    assert "English" in rag.language_directive("en")
    assert "simple English" in rag.language_directive("simple English")  # passthrough


def _result(chunks, low_confidence=False):
    return {"chunks": chunks, "low_confidence": low_confidence}


def _cfg(**kw):
    """A prompt config that does NOT depend on this machine's .env — otherwise
    these assertions flip with LLM_PROVIDER, since the prompt style follows it."""
    kw.setdefault("provider", "groq")
    return rag.GenerationConfig(**kw)


def test_build_prompt_restates_rules_after_the_context():
    """The reminder must come AFTER the blocks and BEFORE the question — small
    models follow the last instruction they read (see final_reminder)."""
    msgs, used, _ = rag.build_prompt("Q?", _result([_hit("a"), _hit("b")]), _cfg())
    user = msgs[-1]["content"]
    assert user.index("Context:") < user.index("Rules for your answer:") < user.index("Question: Q?")
    assert len(used) == 2


def test_compact_prompt_is_selected_for_a_local_model_only():
    """A 3B model given the full prompt replied "[1]" and nothing else; the
    compact prompt is what makes it answer. groq keeps the long one."""
    assert rag.GenerationConfig(provider="ollama").compact_prompt is True
    assert rag.GenerationConfig(provider="groq").compact_prompt is False
    # ...and an explicit setting still wins over the provider default
    assert rag.GenerationConfig(provider="ollama", compact_prompt=False).compact_prompt is False


def test_compact_prompt_puts_the_instruction_after_the_question():
    """Measured: rules-then-question gave 4 completion tokens, question-then-rule
    gave 99. Ordering is load-bearing, so it is pinned here.

    Asserted against final_reminder_compact's actual output rather than a quoted
    phrase from it: the wording of that reminder is tuned regularly, and pinning
    a literal made this test fail for a prompt edit that had not broken anything.
    """
    cfg = _cfg(provider="ollama")
    msgs, _, _ = rag.build_prompt("Q?", _result([_hit("a"), _hit("b")]), cfg)
    user = msgs[-1]["content"]
    reminder = rag.final_reminder_compact(2)
    assert reminder in user
    assert user.index("Question: Q?") < user.index(reminder)
    assert msgs[0]["content"] == rag.SYSTEM_PROMPT_COMPACT
    assert "Rules for your answer:" not in user      # the long list stays off


def test_compact_reminder_bounds_citations_and_allows_refusal():
    assert "[1]" in rag.final_reminder_compact(1)
    assert "[1]-[4]" in rag.final_reminder_compact(4)
    assert "cite\nnothing" in rag.final_reminder_compact(2).replace(" ", "\n")
    # the weak-retrieval steer is folded in here, not into the system prompt
    # a weak retrieval replaces the instruction outright with a refusal order,
    # rather than hedging — a hedged version still let the model stretch
    # near-zero-scoring blocks into an answer.
    #
    # Asserted as PROPERTIES rather than quoted phrases: the exact wording is
    # tuned often, but these three must survive every rewrite, and one rewrite
    # dropped all of them at once ("do your best to extract any relevant
    # information"), which quietly removes abstention.
    weak = rag.final_reminder_compact(2, low_confidence=True).lower()
    assert "weak" in weak                              # a weak match is flagged
    assert "cite" in weak                              # and carries citations for described blocks
    assert "guess" in weak or "confident" in weak      # and is not a hedge
    # the normal instruction must not leak into the weak one
    assert rag.final_reminder_compact(2) != rag.final_reminder_compact(2, True)


def test_both_system_prompts_keep_the_four_grounding_rules():
    """REGRESSION GUARD, and the reason it exists is worth stating.

    A prompt rewrite aimed purely at making answers longer replaced both system
    prompts with "be helpful and detailed" and, in doing so, silently dropped
    every safety rule the corpus measurements depend on. Nothing failed — the
    tests all still passed, the API still answered, and the only way to notice
    was to re-read the diff. GROUNDED 20/20 and ABSTAINS 2/3 in CHECKLIST.md are
    measurements OF THESE RULES, so losing them invalidates the numbers without
    changing them.

    Length and tone are free to be tuned. These four properties are not.
    """
    for name, prompt in (("SYSTEM_PROMPT", rag.SYSTEM_PROMPT),
                         ("SYSTEM_PROMPT_COMPACT", rag.SYSTEM_PROMPT_COMPACT)):
        # collapse wrapping: these prompts are hard-wrapped for readability, so
        # a rule can legitimately straddle a newline
        low = " ".join(prompt.lower().split())
        # 1. answer ONLY from what was retrieved
        assert "only" in low, name
        # 2. never fabricate a figure — the worst failure for a government tool
        assert "never invent" in low, name
        # 3. surface a conflict rather than silently picking a side
        assert "conflict" in low, name
        # 4. refusing is allowed, and costs no citation
        assert "cite nothing" in low, name


def test_low_confidence_addendum_orders_abstention_not_effort():
    """It must not read as "try harder": retrieval has already decided nothing
    scored as a confident match, and a hedged version measurably had the model
    stretch unrelated blocks into a confident-sounding answer."""
    low = " ".join(rag.LOW_CONFIDENCE_ADDENDUM.lower().split())
    assert "do not appear to cover" in low
    assert "do not stretch" in low


def test_final_reminder_bounds_citations_to_the_blocks_actually_sent():
    """Naming the range is what stopped the model inventing [2][3] when only
    one block existed."""
    assert "[1]" in rag.final_reminder(1) and "[1] to [" not in rag.final_reminder(1)
    assert "[1] to [4]" in rag.final_reminder(4)


def test_final_reminder_always_describes_nearby_blocks_with_citations():
    """Regression: a blanket 'NO [n] anywhere' in the refusal rule killed
    citations even when the model was describing what nearby blocks contained.
    Now rule 4 asks for citations when describing blocks, while staying honest
    that the match is weak."""
    for low in (False, True):
        text = rag.final_reminder(3, low_confidence=low)
        assert "cite" in text.lower()              # citations are always required
        assert "do not cover" in text.lower() or "do not actually answer" in text.lower()
    # a weak retrieval additionally steers toward that description
    assert "WEAKLY" in rag.final_reminder(3, low_confidence=True)
    assert "WEAKLY" not in rag.final_reminder(3, low_confidence=False)


def test_build_prompt_language_toggle_reaches_the_prompt():
    for provider in ("groq", "ollama"):     # the toggle must work on both paths
        cfg = _cfg(provider=provider)
        msgs, _, _ = rag.build_prompt("Q?", _result([_hit("body")]), cfg, language="mr")
        assert "Write the answer in Marathi" in msgs[-1]["content"]
        # ...and "auto" adds no directive at all
        msgs_auto, _, _ = rag.build_prompt("Q?", _result([_hit("body")]), cfg)
        assert "Write the answer in" not in msgs_auto[-1]["content"]


def test_low_confidence_addendum_has_no_placeholder_to_copy():
    """A literal "<...>" slot in the abstention template was reproduced verbatim
    by the small model ("...is about <topic of the blocks>")."""
    msgs, _, _ = rag.build_prompt("Q?", _result([_hit("body")], low_confidence=True), _cfg())
    assert "<" not in rag.LOW_CONFIDENCE_ADDENDUM
    assert rag.LOW_CONFIDENCE_ADDENDUM.strip() in msgs[0]["content"]


def test_resolve_citations_carries_gr_fields():
    used = [_hit("body", gr_number="GR/1", date="2023-06-15", department="HTE", language="mr", title="T")]
    src = rag.resolve_citations([1], used)[0]
    assert src["gr_number"] == "GR/1" and src["date"] == "2023-06-15"
    assert src["department"] == "HTE" and src["n"] == 1


# --------------------------------------------------------------------------- #
# HARD abstention gate (FR 3.3.5). Deterministic on purpose: the prompt-only
# version depended on a 3B model choosing to refuse, and measured 0/5 on a
# question whose top retrieval score was 0.006.
# --------------------------------------------------------------------------- #

def _ooc(score):
    return {"chunks": [_hit("irrelevant text", score=score)], "low_confidence": True}


def test_hard_abstention_fires_below_the_floor():
    cfg = _cfg(provider="ollama", abstain_floor=0.10)
    out = rag._hard_abstention("How do I apply for a passport?", _ooc(0.006), cfg, "auto")
    assert out is not None
    assert out["abstained"] is True
    # An abstention must cite NOTHING — the premise of firing is that nothing
    # retrieved is relevant, so a citation here is phantom provenance.
    assert out["sources"] == []
    assert out["phantom_citations"] == []
    assert out["low_confidence"] is True


def test_hard_abstention_does_not_fire_on_a_real_hit():
    """The weakest RELEVANT top hit measured on the gold set is 0.946. The floor
    must sit far below that, or the system refuses answerable questions."""
    cfg = _cfg(provider="ollama", abstain_floor=0.10)
    assert rag._hard_abstention("Q?", _ooc(0.946), cfg, "auto") is None
    assert rag._hard_abstention("Q?", _ooc(0.11), cfg, "auto") is None


def test_hard_abstention_fires_when_retrieval_returned_nothing():
    """A filter that matches no document yields chunks=[]. max() over an empty
    sequence would raise, and 'no chunks' is the strongest possible signal."""
    cfg = _cfg(provider="ollama", abstain_floor=0.10)
    out = rag._hard_abstention("Q?", {"chunks": [], "low_confidence": True}, cfg, "auto")
    assert out is not None and out["top_score"] == 0.0


def test_hard_abstention_can_be_disabled():
    cfg = _cfg(provider="ollama", abstain_floor=0)
    assert rag._hard_abstention("Q?", _ooc(0.0), cfg, "auto") is None


def test_abstention_language_follows_the_QUESTION_not_the_context():
    """Measured failure this prevents: an English passport question was answered
    in Marathi. When we refuse, the retrieved text is by definition irrelevant,
    so its language says nothing about the officer's."""
    assert rag._abstain_language("How do I apply for a new passport?", "auto") == "en"
    assert rag._abstain_language("कर्जमाफी योजनेची रक्कम किती आहे?", "auto") == "mr"
    # an explicit portal toggle still wins
    assert rag._abstain_language("How do I apply?", "mr") == "mr"
    assert rag._abstain_language("कर्जमाफी?", "en") == "en"


def test_abstention_messages_are_recognisable_as_refusals():
    """eval_answers.py detects abstention by phrase. If the canned message ever
    stops matching that intent, the metric silently reads 0."""
    import re
    detector = re.compile(r"do(es)? not (appear to )?(cover|contain|provide)|"
                          r"आढळत नाही|उपलब्ध नाही", re.IGNORECASE)
    for lang, msg in rag.ABSTAIN_MESSAGES.items():
        assert detector.search(msg), lang
