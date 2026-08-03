"""Local SQLite persistence (Phase 3) — conversations, messages, feedback."""

import importlib


def _fresh_db(monkeypatch, tmp_path):
    monkeypatch.setenv("MAHAGR_DB", str(tmp_path / "t.db"))
    from app import db
    importlib.reload(db)   # re-read DB_PATH from the env
    db.init()
    return db


def test_conversation_and_messages_persist(monkeypatch, tmp_path):
    db = _fresh_db(monkeypatch, tmp_path)
    cid = db.create_conversation("What is the OBC fee?")
    db.add_message(cid, "user", "What is the OBC fee?")
    db.add_message(cid, "assistant", "6000 [1]", [{"n": 1, "gr_number": "GR/x"}], ["superseded"])

    convs = db.list_conversations()
    assert len(convs) == 1 and convs[0]["messages"] == 2
    msgs = db.get_messages(cid)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    # structured fields round-trip through JSON
    assert msgs[1]["sources"][0]["gr_number"] == "GR/x"
    assert msgs[1]["warnings"] == ["superseded"]


def test_feedback_and_delete(monkeypatch, tmp_path):
    db = _fresh_db(monkeypatch, tmp_path)
    cid = db.create_conversation("Q")
    mid = db.add_message(cid, "assistant", "ans")
    assert db.add_feedback(cid, mid, "up")
    db.delete_conversation(cid)
    assert db.list_conversations() == []
    assert db.get_messages(cid) == []
