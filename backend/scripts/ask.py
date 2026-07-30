"""
ask.py — query the MahaGR index from the command line, with the FULL pipeline:
hybrid retrieval (dense bge-m3 + BM25) -> multilingual rerank
(bge-reranker-v2-m3) -> grounded, cited generation (Groq).

This is the one entry point that wires the reranker in (rag.py's own CLI runs
reranker-free). Ask in English or Marathi; the answer comes back in the
question's language with [n] citations to the source GRs.

Usage:
    python scripts/ask.py "What is the DTE admission fee structure?"
    python scripts/ask.py                 # interactive REPL
Requires GROQ_API_KEY in .env (see .env.example).
"""

import sys

from engine import rag
from engine.reranker import Reranker
from engine.retrieval import load_default_retriever


def main():
    client = rag.make_client()             # fail fast on a missing GROQ_API_KEY
    print("Loading index, embedding model (bge-m3) and reranker (bge-reranker-v2-m3)...")
    reranker = Reranker()                  # multilingual cross-encoder
    retriever = load_default_retriever(reranker=reranker)

    if len(sys.argv) > 1:
        rag.print_result(rag.answer(" ".join(sys.argv[1:]), retriever, client))
        return

    print('Ready. Ask in English or Marathi (blank line or Ctrl-D to quit).')
    history = []
    while True:
        try:
            q = input("\n?> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        try:
            result = rag.answer(q, retriever, client, history=history)
            rag.print_result(result)
            # keep a short conversational memory so follow-ups resolve (rag
            # rewrites the next question against this history before retrieval)
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": result["answer"]})
        except Exception as e:
            print(f"  error: {e}")


if __name__ == "__main__":
    main()
