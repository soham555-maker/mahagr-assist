"""
Test bootstrap. The engine's pure logic (tokenizing, metadata parsing, RRF,
citation handling, chunking, the supersession graph) needs no ML models, but
engine/ingest.py imports sentence_transformers at module top (which drags in
torch). We stub it here BEFORE any test imports an engine module, so the whole
suite runs fast and without torch installed — a genuinely model-free test run.
Model instantiation (SentenceTransformer(), CrossEncoder()) is never exercised
by these tests, so the stub is never actually called.
"""

import os
import sys
import types

# make the `app` package importable (editable install only exposes `engine`)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "sentence_transformers" not in sys.modules:
    _st = types.ModuleType("sentence_transformers")
    _st.SentenceTransformer = object
    _st.CrossEncoder = object
    sys.modules["sentence_transformers"] = _st
