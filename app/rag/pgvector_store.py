"""Optional Supabase PostgreSQL + pgvector backend.

Configured with `VECTOR_STORE=pgvector` and `SUPABASE_DB_URL` (the Postgres connection
string from Supabase > Project Settings > Database). The URL is a secret: keep it in `.env`
or a secret manager, never in code or notebook output. Error messages never include it.
"""

import json
import re
from collections.abc import Callable, Collection, Sequence

import numpy as np

from app.rag.deps import MissingDependencyError, require
from app.rag.models import DocumentChunk, RetrievedChunk
from app.rag.vector_store import VectorStoreError, _as_matrix

_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def _vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in vector) + "]"


def schema_sql(table: str, dimension: int) -> list[str]:
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"""CREATE TABLE IF NOT EXISTS {table} (
            chunk_id text PRIMARY KEY,
            document_id text NOT NULL,
            filename text NOT NULL,
            page_number integer NOT NULL,
            chunk_index integer NOT NULL,
            content text NOT NULL,
            start_char integer NOT NULL,
            end_char integer NOT NULL,
            metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
            embedding vector({dimension}) NOT NULL
        )""",
        f"CREATE INDEX IF NOT EXISTS {table}_document_idx ON {table} (document_id)",
        f"CREATE INDEX IF NOT EXISTS {table}_embedding_idx ON {table} "
        "USING hnsw (embedding vector_cosine_ops)",
    ]


class PgVectorStore:
    def __init__(
        self,
        dsn: str,
        dimension: int,
        *,
        table: str = "document_chunks",
        connect: Callable | None = None,
    ) -> None:
        if not _IDENTIFIER_RE.match(table):
            raise ValueError(f"Invalid table name: {table!r}")
        self.dimension = dimension
        self.table = table
        self._dsn = dsn
        self._connect = connect

    def _connection(self):
        if self._connect is None:
            psycopg = require("psycopg", "the Supabase/pgvector vector store")
            self._connect = psycopg.connect
        return self._connect(self._dsn)

    def _run(self, action: str, fn):
        try:
            with self._connection() as conn, conn.cursor() as cur:
                return fn(cur)
        except (VectorStoreError, MissingDependencyError):
            raise
        except Exception as exc:
            # Report the error class and message only; never echo the connection string.
            raise VectorStoreError(
                f"pgvector {action} failed ({type(exc).__name__}): {exc}"
            ) from exc

    def ensure_schema(self) -> None:
        def create(cur):
            for statement in schema_sql(self.table, self.dimension):
                cur.execute(statement)

        self._run("schema setup", create)

    def add(self, chunks: Sequence[DocumentChunk], embeddings: np.ndarray) -> None:
        matrix = _as_matrix(embeddings, self.dimension)
        if len(chunks) != matrix.shape[0]:
            raise VectorStoreError(f"{len(chunks)} chunks but {matrix.shape[0]} embeddings")
        rows = [
            (
                c.chunk_id,
                c.document_id,
                c.filename,
                c.page_number,
                c.chunk_index,
                c.text,
                c.start_char,
                c.end_char,
                json.dumps(c.metadata),
                _vector_literal(vector),
            )
            for c, vector in zip(chunks, matrix, strict=True)
        ]
        sql = (
            f"INSERT INTO {self.table} (chunk_id, document_id, filename, page_number, "
            "chunk_index, content, start_char, end_char, metadata, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector) "
            "ON CONFLICT (chunk_id) DO UPDATE SET content = EXCLUDED.content, "
            "embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata"
        )
        if rows:
            self._run("insert", lambda cur: cur.executemany(sql, rows))

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        document_ids: Collection[str] | None = None,
    ) -> list[RetrievedChunk]:
        literal = _vector_literal(_as_matrix(query_embedding, self.dimension)[0])
        where, params = "", [literal]
        if document_ids is not None:
            where, params = "WHERE document_id = ANY(%s)", [literal, list(document_ids)]
        sql = (
            "SELECT chunk_id, document_id, filename, page_number, chunk_index, content, "
            "start_char, end_char, metadata, 1 - (embedding <=> %s::vector) AS score "
            f"FROM {self.table} {where} ORDER BY embedding <=> %s::vector LIMIT %s"
        )

        def query(cur):
            cur.execute(sql, [*params, literal, top_k])
            return cur.fetchall()

        results = []
        for rank, row in enumerate(self._run("search", query), start=1):
            metadata = row[8] if isinstance(row[8], dict) else json.loads(row[8] or "{}")
            chunk = DocumentChunk(
                chunk_id=row[0],
                document_id=row[1],
                filename=row[2],
                page_number=row[3],
                chunk_index=row[4],
                text=row[5],
                start_char=row[6],
                end_char=row[7],
                metadata=metadata,
            )
            results.append(RetrievedChunk(chunk=chunk, score=float(row[9]), rank=rank))
        return results

    def delete_document(self, document_id: str) -> None:
        sql = f"DELETE FROM {self.table} WHERE document_id = %s"
        self._run("delete", lambda cur: cur.execute(sql, [document_id]))

    def document_ids(self) -> set[str]:
        def query(cur):
            cur.execute(f"SELECT DISTINCT document_id FROM {self.table}")
            return {row[0] for row in cur.fetchall()}

        return self._run("list documents", query)

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        sql = (
            "SELECT chunk_id, document_id, filename, page_number, chunk_index, content, "
            f"start_char, end_char, metadata FROM {self.table} WHERE chunk_id = %s"
        )

        def query(cur):
            cur.execute(sql, [chunk_id])
            return cur.fetchall()

        rows = self._run("get chunk", query)
        if not rows:
            return None
        row = rows[0]
        metadata = row[8] if isinstance(row[8], dict) else json.loads(row[8] or "{}")
        return DocumentChunk(
            chunk_id=row[0],
            document_id=row[1],
            filename=row[2],
            page_number=row[3],
            chunk_index=row[4],
            text=row[5],
            start_char=row[6],
            end_char=row[7],
            metadata=metadata,
        )

    def __len__(self) -> int:
        def query(cur):
            cur.execute(f"SELECT count(*) FROM {self.table}")
            return int(cur.fetchone()[0])

        return self._run("count", query)
