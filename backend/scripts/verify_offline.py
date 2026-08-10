"""
verify_offline.py — PROVE the on-premise claim instead of asserting it.

The SRS requires on-premise/NIC deployment for sensitive documents, and the
whole pitch is "no Government document ever leaves the machine". A health
endpoint reporting `llm_provider: ollama` is not proof of that — it only says
which code path was chosen. This script makes the claim falsifiable:

  1. GROQ_API_KEY is set to "" before any engine import, so no hosted
     credential exists in the process. (It is SET, not deleted — config.py's
     load_dotenv() would otherwise put the key back from backend/.env, since
     dotenv does not override a variable that is already present.)
  2. socket.connect is monkey-patched to RAISE on any non-loopback address.
     Every stage — embedding, reranking, generation — must therefore run either
     in-process or against localhost, or the run fails loudly.

Then it answers one Marathi and one English question end to end. If they come
back cited, the pipeline demonstrably touched nothing outside this machine.

Run with the API server stopped (it holds the GPU — see HANDOFF §5):
    python scripts/verify_offline.py
"""

import argparse
import os
import socket
import sys

QUESTIONS = [
    "२०२४-२५ या वर्षासाठी खुल्या प्रवर्गाचे सुधारित वार्षिक शुल्क किती आहे?",
    "What is the annual fee for OBC students for the first-year diploma in 2023-24?",
]


def block_remote_sockets():
    """Allow loopback only. Ollama is localhost:11434; the models are on disk."""
    real_connect = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else ""
        if not (str(host).startswith("127.") or host in ("::1", "localhost")):
            raise RuntimeError(f"BLOCKED outbound connection to {address!r} — "
                               f"something in the pipeline is not local")
        return real_connect(self, address)

    socket.socket.connect = guarded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=None,
                    help="index directory (default: config.INDEX_DIR)")
    args = ap.parse_args()

    os.environ["GROQ_API_KEY"] = ""
    # Keep the model libraries from phoning home for a revision check too.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    block_remote_sockets()

    from engine import rag
    from engine.reranker import Reranker
    from engine.retrieval import load_default_retriever

    cfg = rag.GenerationConfig()
    if cfg.provider != "ollama":
        sys.exit(f"LLM_PROVIDER is {cfg.provider!r}, not 'ollama' — this run would "
                 f"need the network. Set LLM_PROVIDER=ollama in backend/.env.")

    print(f"provider={cfg.provider}  model={cfg.ollama_model}  "
          f"GROQ_API_KEY set: {bool(os.environ.get('GROQ_API_KEY'))}")

    retriever = load_default_retriever(index_dir=args.index, reranker=Reranker())
    client = rag.make_client(cfg)
    print(f"client={type(client).__name__}  vectors={len(retriever.store)}\n")

    failures = 0
    for question in QUESTIONS:
        result = rag.answer(question, retriever, client, config=cfg)
        cited = len(result["sources"])
        failures += cited == 0
        print(f"Q: {question}\nA: {result['answer'][:200]}\n   "
              f"sources={cited} model={result['model']}\n")

    if failures:
        sys.exit(f"{failures} answer(s) came back with no citation — "
                 f"offline path works but grounding did not.")
    print("OFFLINE VERIFIED — no hosted credential, every connection loopback-only.")


if __name__ == "__main__":
    main()
