"""
verify_demo.py — prove each demo question retrieves the RIGHT GR (and, for
number questions, that the exact figure is in the retrieved text). Retrieval is
deterministic and needs no LLM/GROQ key, so this verifies the demo will work
before you present. The live answer text is produced by the LLM at demo time.

Usage:  python scripts/verify_demo.py
"""

from engine.officer import _hay
from engine.reranker import Reranker
from engine.retrieval import load_default_retriever

# (label, question, expected source substrings, expected number|None, out_of_corpus)
DEMO = [
    ("multilingual EN->Marathi",
     "What procedure must a university follow to approve new colleges and courses?",
     ["201710121514029708", "एनजीसी २०१७"], None, False),
    ("Marathi question",
     "ग्रंथालय संचालनालयाच्या आधुनिकीकरणाबाबतचा निर्णय काय आहे?",
     ["201711061646497708", "मराग्रं २५१७"], None, False),
    ("table+number EN->Marathi (OBC 2023 fee)",
     "What is the annual fee for OBC students for the first-year diploma in 2023-24?",
     ["२०२३", "GR-2023"], "6000", False),
    ("table+number Marathi (Open 2024 fee)",
     "२०२४-२५ या वर्षासाठी खुल्या प्रवर्गाचे सुधारित वार्षिक शुल्क किती आहे?",
     ["२०२४", "GR-2024"], "15000", False),
    ("table+number Marathi (OBC 2024 revised fee)",
     "सुधारित शुल्करचनेनुसार इतर मागासवर्ग (OBC) प्रवर्गाचे वार्षिक शुल्क किती आहे?",
     ["२०२४", "GR-2024"], "7500", False),
    ("out-of-corpus (must abstain)",
     "What is the annual fee for a PhD in aerospace engineering at IIT Bombay?",
     [], None, True),
]


def main():
    print("Loading index + bge-m3 + reranker ...")
    retriever = load_default_retriever(reranker=Reranker())
    passed = 0
    for label, q, sources, number, ooc in DEMO:
        res = retriever.retrieve(q)
        chunks = res["chunks"]
        top_hay = _hay(chunks[0]["metadata"]) if chunks else ""
        all_text = " ".join(c["text"] for c in chunks)
        if ooc:
            ok = res["low_confidence"]  # nothing cleared the bar -> abstain
            detail = f"low_confidence={res['low_confidence']} (top score {chunks[0]['score']:.3f})" if chunks else "no hits"
        else:
            src_ok = any(s in top_hay for s in sources)
            num_ok = number is None or number in all_text
            ok = src_ok and num_ok
            top_id = chunks[0]["metadata"].get("gr_number") or chunks[0]["metadata"].get("order_id")
            detail = f"top={top_id}  src_ok={src_ok}" + ("" if number is None else f"  '{number}'_found={num_ok}")
        passed += ok
        print(f"  [{'PASS' if ok else 'FAIL'}]  {label}\n          {detail}")
    print(f"\n{passed}/{len(DEMO)} demo questions verified.")


if __name__ == "__main__":
    main()
