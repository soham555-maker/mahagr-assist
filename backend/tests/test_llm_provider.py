"""The swappable LLM provider seam (Phase 7): groq (default) vs local ollama."""

from engine import rag


def test_ollama_response_parsing():
    # OpenAI-shaped payload from Ollama's /v1/chat/completions -> Groq-like object
    payload = {
        "choices": [{"message": {"content": "उत्तर"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }
    r = rag.OllamaClient._to_response(payload)
    assert r.choices[0].message.content == "उत्तर"
    assert r.choices[0].finish_reason == "stop"
    assert r.usage.prompt_tokens == 12 and r.usage.completion_tokens == 3


def test_make_client_ollama_needs_no_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cfg = rag.GenerationConfig(provider="ollama")
    client = rag.make_client(cfg)
    assert isinstance(client, rag.OllamaClient)


def test_call_llm_ollama_uses_local_model_not_groq_name():
    """On ollama, the per-role groq model name is ignored; the one local model
    (config.ollama_model) is used for every call."""
    captured = {}

    class FakeCompletions:
        def create(self, **kw):
            captured.update(kw)
            return rag.OllamaClient._to_response({"choices": [{"message": {"content": "ok"}}]})

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    cfg = rag.GenerationConfig(provider="ollama", ollama_model="llama3.1:8b")
    rag.call_llm([{"role": "user", "content": "hi"}], cfg, FakeClient(), model="llama-3.3-70b-versatile")
    assert captured["model"] == "llama3.1:8b"  # NOT the groq name passed in


def test_provider_resolves_env_then_config(monkeypatch):
    """LLM_PROVIDER in the environment wins; otherwise the value config.py read
    from backend/.env at import time is used. config is patched too so the test
    asserts the resolution order rather than whatever .env this machine has —
    on an on-prem box .env says ollama, on the demo box it says groq."""
    monkeypatch.setattr(rag.config, "LLM_PROVIDER", "groq")

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert rag.GenerationConfig().provider == "groq"

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert rag.GenerationConfig().provider == "ollama"
