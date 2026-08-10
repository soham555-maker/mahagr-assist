"""
eval_answers.py — measure the GENERATED answer, not just retrieval.

eval_retrieval.py proves the right GR reaches the prompt. This proves the model
then uses it properly, which is a different question and the one that actually
degrades when you swap in a small local model (PLAN Phase 1). It runs the gold
set through the live API and scores the four things the SRS is explicit about:

  CITED       every answer carries a [n] citation            (FR 3.3.3, Explainability)
  GROUNDED    no citation points at a block never sent       (phantom = groundedness failure)
  CORRECT     the expected GR is among the cited sources     (retrieval + use, end to end)
  ABSTAINS    out-of-corpus questions are refused, not answered (FR 3.3.5)

plus latency against the SRS's <10 s target, and a DEGENERATE check for replies
that are only a citation with no sentence — a real failure mode of small models
that every other metric here would happily score as a pass.

Gold entries with in_corpus=false are the out-of-corpus questions; they are
scored on abstention and excluded from the other metrics.

Run the API first (uvicorn app.api:app --port 8000), then:
    python scripts/eval_answers.py
    python scripts/eval_answers.py --url http://localhost:8000 --language mr
"""

import argparse
import json
import re
import statistics
import time
import urllib.request

CITATION_RE = re.compile(r"\[\d+\]")
# "Not covered", in either language. The model phrases this freely, so match the
# INTENT — an earlier, narrower pattern scored a perfectly good refusal ("does
# not provide information on...") as a hallucination and undercounted abstention.
ABSTAIN_RE = re.compile(
    r"do(es)? not (appear to |seem to )?(cover|contain|provide|include|mention|have)|"
    r"not covered|no (relevant )?information|nothing (about|on)|not available|"
    r"could not find|unable to (find|answer)|"
    r"उपलब्ध नाही|माहिती नाही|समाविष्ट नाही|आढळत नाही|नमूद नाही",
    re.IGNORECASE)


def _ask(url, question, language, timeout):
    body = json.dumps({"question": question, "language": language}).encode()
    req = urllib.request.Request(f"{url}/ask", data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.time() - t


def _cites_expected(sources, expected):
    """True if any cited source names an expected GR. Same field-bag match as
    eval_retrieval._matches, over the resolved citation rather than the hit."""
    for s in sources:
        hay = " ".join(str(s.get(k, "")) for k in ("source_file", "gr_number", "title"))
        if any(e and e in hay for e in expected):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/gold/gold.json")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--language", default="auto")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    gold = json.load(open(args.gold, encoding="utf-8"))["questions"]
    in_corpus = [q for q in gold if q.get("in_corpus")]
    out_corpus = [q for q in gold if not q.get("in_corpus")]

    cited = grounded = correct = degenerate = 0
    abstained = 0
    latencies = []
    # Per-stage seconds reported by /ask (rag.answer's `timings`). Without this
    # the 10 s budget can only be attacked by guesswork: retrieval and
    # generation are tuned by completely different knobs.
    stages = {"rewrite": [], "retrieval": [], "generation": []}
    prompt_tokens, completion_tokens = [], []
    truncated = 0
    failures = []

    for q in gold:
        question, expected = q["q"], q.get("expected") or []
        res, dt = _ask(args.url, question, args.language, args.timeout)
        latencies.append(dt)
        for k in stages:
            if k in (res.get("timings") or {}):
                stages[k].append(res["timings"][k])
        truncated += bool(res.get("truncated"))
        usage = res.get("usage") or {}
        if usage.get("prompt_tokens"):
            prompt_tokens.append(usage["prompt_tokens"])
            completion_tokens.append(usage.get("completion_tokens", 0))
        text = res["answer"]
        prose = CITATION_RE.sub("", text).strip()

        if q.get("in_corpus"):
            has_cite = bool(res["sources"])
            cited += has_cite
            grounded += not res["phantom_citations"]
            hit = _cites_expected(res["sources"], expected)
            correct += hit
            # A reply that is only "[1]" satisfies every citation metric while
            # telling the officer nothing.
            # A PHANTOM citation — a [n] pointing at a block never sent — is the
            # groundedness alarm, and it used to be counted but never REPORTED:
            # the failure list showed only WRONG-SOURCE/DEGENERATE/UNCITED, so a
            # GROUNDED regression appeared as a number with no way to see which
            # question caused it. It is listed FIRST because it outranks the
            # others: citing a document that was never retrieved is worse than
            # citing the wrong one that was.
            if res["phantom_citations"]:
                failures.append(("PHANTOM", question,
                                 f"cited {res['phantom_citations']} but only "
                                 f"{len(res.get('sources') or [])} source(s) resolved :: {text[:90]}"))
            if len(prose) < 15:
                degenerate += 1
                failures.append(("DEGENERATE", question, text))
            elif not hit:
                failures.append(("WRONG-SOURCE", question, text[:120]))
            elif not has_cite:
                failures.append(("UNCITED", question, text[:120]))
        else:
            ok = bool(ABSTAIN_RE.search(text))
            abstained += ok
            if not ok:
                failures.append(("NO-ABSTAIN", question, text[:120]))

        if args.verbose:
            print(f"  {dt:5.1f}s  {question[:60]}\n         {text[:100]}")

    n_in, n_out = len(in_corpus), len(out_corpus)
    print(f"\nAnswer quality over {len(gold)} gold questions "
          f"({n_in} in-corpus, {n_out} out-of-corpus)")
    print(f"  CITED       {cited}/{n_in}   answers carrying a [n] citation")
    print(f"  GROUNDED    {grounded}/{n_in}   answers with no phantom citation")
    print(f"  CORRECT     {correct}/{n_in}   answers citing the expected GR")
    print(f"  DEGENERATE  {degenerate}/{n_in}   replies that are only a citation")
    print(f"  ABSTAINS    {abstained}/{n_out}   out-of-corpus questions refused")
    print(f"\nLatency  p50 {statistics.median(latencies):.1f}s   "
          f"max {max(latencies):.1f}s   "
          f"over-10s {sum(1 for l in latencies if l > 10)}/{len(latencies)}"
          f"   (SRS target <10s)")
    for name, vals in stages.items():
        if vals:
            print(f"    {name:11s} p50 {statistics.median(vals):5.2f}s   "
                  f"max {max(vals):5.2f}s   "
                  f"share of p50 {100 * statistics.median(vals) / statistics.median(latencies):3.0f}%")
    if prompt_tokens:
        print(f"    prompt tokens p50 {statistics.median(prompt_tokens):.0f}   "
              f"completion p50 {statistics.median(completion_tokens):.0f} "
              f"max {max(completion_tokens)}   "
              f"answers cut off by the output cap: {truncated}/{len(gold)}")

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for kind, question, text in failures:
            print(f"  [{kind}] {question[:70]}\n      -> {text}")


if __name__ == "__main__":
    main()
