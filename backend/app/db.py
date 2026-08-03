"""
db.py — local SQLite persistence for the portal (Phase 3): conversations,
messages, and officer feedback. Standard-library only (sqlite3) — nothing
external, so it fits the on-prem story: the DB is a single file on disk.

Two jobs:
  * Durability — chats and feedback survive a restart (they were in-memory
    before), and the stored history is loaded back into the query-rewrite step
    so follow-up questions resolve across sessions, not just within one tab.
  * Feedback capture — thumbs up/down (+ optional comment) per answer.

Path via MAHAGR_DB (default data/db/mahagr.db); the directory is created on init.
"""

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager

DB_PATH = os.environ.get("MAHAGR_DB", os.path.join("data", "db", "mahagr.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY, title TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY, conversation_id TEXT, role TEXT, content TEXT,
    sources TEXT, warnings TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY, conversation_id TEXT, message_id TEXT,
    rating TEXT, comment TEXT, created_at REAL
);
CREATE INDEX IF NOT EXISTS ix_messages_conv ON messages(conversation_id, created_at);
"""


@contextmanager
def _db():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _id():
    return uuid.uuid4().hex[:12]


def init():
    with _db() as c:
        c.executescript(_SCHEMA)


def create_conversation(title):
    cid = _id()
    with _db() as c:
        c.execute("INSERT INTO conversations VALUES (?,?,?)", (cid, (title or "New conversation")[:80], time.time()))
    return cid


def list_conversations():
    with _db() as c:
        rows = c.execute(
            """SELECT c.id, c.title, c.created_at, COUNT(m.id) AS messages
               FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id
               GROUP BY c.id ORDER BY c.created_at DESC""").fetchall()
    return [dict(r) for r in rows]


def get_messages(cid):
    with _db() as c:
        rows = c.execute(
            "SELECT id, role, content, sources, warnings FROM messages "
            "WHERE conversation_id=? ORDER BY created_at", (cid,)).fetchall()
    return [{"id": r["id"], "role": r["role"], "content": r["content"],
             "sources": json.loads(r["sources"] or "[]"),
             "warnings": json.loads(r["warnings"] or "[]")} for r in rows]


def add_message(cid, role, content, sources=None, warnings=None):
    mid = _id()
    with _db() as c:
        c.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?)",
                  (mid, cid, role, content,
                   json.dumps(sources or [], ensure_ascii=False),
                   json.dumps(warnings or [], ensure_ascii=False), time.time()))
    return mid


def delete_conversation(cid):
    with _db() as c:
        c.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
        c.execute("DELETE FROM feedback WHERE conversation_id=?", (cid,))
        c.execute("DELETE FROM conversations WHERE id=?", (cid,))


def add_feedback(conversation_id, message_id, rating, comment=None):
    fid = _id()
    with _db() as c:
        c.execute("INSERT INTO feedback VALUES (?,?,?,?,?,?)",
                  (fid, conversation_id, message_id, rating, comment, time.time()))
    return fid
