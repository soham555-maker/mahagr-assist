"""
eval_retrieval.py — measure retrieval quality and CALIBRATE the thresholds.

The thresholds in RetrievalConfig (text/table cosine + rerank logit) are
model- and corpus-specific and currently placeholders for bge-m3. This harness
turns a gold set (data/gold/gold.json) into the two things you need to set them
honestly:

  1. QUALITY   — hit@1, hit@5, MRR over the in-corpus questions.
  2. SEPARATION — the score of the top RELEVANT hit (scores we must KEEP) vs the
     top IRRELEVANT / out-of-corpus hit (scores we must REJECT). A good
     threshold sits in the gap between those two distributions.

It retrieves with the thresholds turned OFF (permissive config) so it sees the
RAW ranked scores — otherwise the very filtering you're trying to tune would
hide the data. Run with the deployed reranker (default) to calibrate
`rerank_threshold`; add --no-rerank to calibrate the cosine thresholds instead.

Usage:
    python scripts/eval_retrieval.py                 # data/gold/gold.json, with reranker
    python scripts/eval_retrieval.py --no-rerank
    python scripts/eval_retrieval.py --gold data/gold/gold.json --k 5
"""

import argparse
import json
import statistics


def _pct(values, p):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def _matches(hit, expected):
    m = hit["metadata"]
    hay = " ".join(str(m.get(k, "")) for k in ("source_file", "order_id", "gr_number", "title"))
    return any(e and e in hay for e in expected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/gold/gold.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    gold = json.load(open(args.gold, encoding="utf-8"))["questions"]

    # Permissive config: thresholds OFF so we observe raw scores, not survivors.
    from engine.retrieval import RetrievalConfig, load_default_retriever
    cfg = RetrievalConfig(text_threshold=-1.0, table_threshold=-1.0,
                          rerank_threshold=-1e9, max_final_k=max(args.k, 10))
    reranker = None
    if not args.no_rerank:
        from engine.reranker import Reranker
        reranker = Reranker()
    print("Loading index + models...")
    retriever = load_default_retriever(config=cfg, reranker=reranker)
    score_kind = "cosine" if args.no_rerank else "rerank-logit"

    in_corpus = [q for q in gold if q.get("in_corpus", True)]
    ooc = [q for q in gold if not q.get("in_corpus", True)]

    hit1 = hit5 = 0
    rr = 0.0
    rel_scores, irr_scores, ooc_scores = [], [], []

    print(f"\n== IN-CORPUS ({len(in_corpus)} questions) ==")
    for q in in_corpus:
        hits = retriever.retrieve(q["q"])["chunks"]
        ranks = [i for i, h in enumerate(hits, 1) if _matches(h, q["expected"])]
        top = ranks[0] if ranks else None
        hit1 += top == 1
        hit5 += bool(ranks and ranks[0] <= 5)
        rr += (1.0 / top) if top else 0.0
        if ranks:
            rel_scores.append(hits[ranks[0] - 1]["score"])
        irr = next((h for h in hits if not _matches(h, q["expected"])), None)
        if irr:
            irr_scores.append(irr["score"])
        flag = f"#{top}" if top else "MISS"
        print(f"  [{flag:>4}] {q['q'][:64]}")

    print(f"\n== OUT-OF-CORPUS ({len(ooc)} questions — must be rejected) ==")
    for q in ooc:
        hits = retriever.retrieve(q["q"])["chunks"]
        if hits:
            ooc_scores.append(hits[0]["score"])
        print(f"  top score {hits[0]['score']:.3f}  {q['q'][:60]}" if hits else f"  (no hits)  {q['q'][:60]}")

    n = len(in_corpus) or 1
    print("\n" + "=" * 62)
    print(f"QUALITY   hit@1={hit1}/{len(in_corpus)}  hit@5={hit5}/{len(in_corpus)}  MRR={rr / n:.3f}")
    print(f"\nSEPARATION ({score_kind})")
    print(f"  KEEP  (relevant top hits)      p10={_fmt(_pct(rel_scores,10))} "
          f"median={_fmt(_pct(rel_scores,50))} p90={_fmt(_pct(rel_scores,90))}  n={len(rel_scores)}")
    reject = irr_scores + ooc_scores
    print(f"  REJECT(irrelevant + OOC tops)  p10={_fmt(_pct(reject,10))} "
          f"median={_fmt(_pct(reject,50))} p90={_fmt(_pct(reject,90))}  n={len(reject)}")

    lo, hi = _pct(rel_scores, 25), _pct(reject, 75)
    if lo is not None and hi is not None:
        mid = (lo + hi) / 2
        gap = lo - hi
        print(f"\nSUGGESTED threshold ~ {mid:.3f}  "
              f"(relevant p25={lo:.3f} vs reject p75={hi:.3f}; "
              f"{'clean +' + format(gap, '.3f') + ' gap' if gap > 0 else 'OVERLAP ' + format(gap, '.3f') + ' — scores do not separate well; lean toward recall'})")
        print(f"  -> set {'rerank_threshold' if not args.no_rerank else 'text_threshold/table_threshold'} "
              f"in engine/retrieval.py near this value (round toward recall).")
    else:
        print("\nNot enough data to suggest a threshold — add more gold questions.")


def _fmt(v):
    return f"{v:.3f}" if v is not None else "  -  "


if __name__ == "__main__":
    main()
