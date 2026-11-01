"""pgvector store tests against a fake connection. No database is contacted."""

import sys

import numpy as np
import pytest

from app.rag.deps import MissingDependencyError
from app.rag.models import DocumentChunk
from app.rag.pgvector_store import PgVectorStore
from app.rag.vector_store import VectorStoreError

DSN = "postgresql://postgres:super-secret-password@db.example.supabase.co:5432/postgres"


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        if self.conn.fail:
            raise self.conn.fail
        self.conn.executed.append((sql, params))

    def executemany(self, sql, rows):
        self.conn.executed.append((sql, list(rows)))

    def fetchall(self):
        return self.conn.rows

    def fetchone(self):
        return self.conn.rows[0]


class FakeConnection:
    def __init__(self, rows=None, fail=None):
        self.rows = rows or []
        self.fail = fail
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return FakeCursor(self)


def make_store(conn):
    return PgVectorStore(DSN, dimension=3, connect=lambda dsn: conn)


def test_add_inserts_vector_literals():
    conn = FakeConnection()
    chunk = DocumentChunk(
        chunk_id="doc_a:p1:c0",
        document_id="doc_a",
        filename="a.pdf",
        page_number=1,
        chunk_index=0,
        text="hello",
        start_char=0,
        end_char=5,
    )

    make_store(conn).add([chunk], np.array([[0.5, 0.25, 0]], dtype=np.float32))

    sql, rows = conn.executed[0]
    assert sql.startswith("INSERT INTO document_chunks")
    assert rows[0][0] == "doc_a:p1:c0" and rows[0][-1] == "[0.5,0.25,0]"


def test_search_maps_rows_and_filters_documents():
    row = ("doc_a:p2:c0", "doc_a", "a.pdf", 2, 0, "text", 0, 4, {"title": "T"}, 0.87)
    conn = FakeConnection(rows=[row])

    results = make_store(conn).search(np.array([1, 0, 0]), top_k=3, document_ids=["doc_a"])

    assert results[0].chunk.page_number == 2 and results[0].score == pytest.approx(0.87)
    assert results[0].chunk.metadata == {"title": "T"}
    sql, params = conn.executed[0]
    assert "document_id = ANY(%s)" in sql and "<=>" in sql
    assert params == ["[1,0,0]", ["doc_a"], "[1,0,0]", 3]


def test_database_errors_are_wrapped_without_leaking_credentials():
    conn = FakeConnection(fail=ConnectionError("connection refused"))

    with pytest.raises(VectorStoreError) as exc_info:
        make_store(conn).search(np.array([1, 0, 0]), top_k=3)

    assert "connection refused" in str(exc_info.value)
    assert "super-secret-password" not in str(exc_info.value)


def test_connect_failure_is_wrapped():
    def refuse(dsn):
        raise OSError("could not resolve host")

    store = PgVectorStore(DSN, dimension=3, connect=refuse)
    with pytest.raises(VectorStoreError, match="could not resolve host"):
        store.ensure_schema()


def test_schema_and_table_validation():
    conn = FakeConnection()
    make_store(conn).ensure_schema()

    statements = [sql for sql, _ in conn.executed]
    assert statements[0] == "CREATE EXTENSION IF NOT EXISTS vector"
    assert "vector(3)" in statements[1] and "hnsw" in statements[3]
    with pytest.raises(ValueError):
        PgVectorStore(DSN, 3, table="chunks; DROP TABLE users")


def test_missing_psycopg_has_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "psycopg", None)

    with pytest.raises(MissingDependencyError, match="psycopg"):
        PgVectorStore(DSN, dimension=3).document_ids()
