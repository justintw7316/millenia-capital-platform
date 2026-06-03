"""
Vector store abstraction for hybrid investor matching.

Two implementations with identical interfaces:
- InMemoryVectorStore  — local dev/testing, no dependencies
- PostgresVectorStore  — production, requires pgvector extension

Switch by passing the appropriate store to HybridMatchingService:
    from matching.vector_store import PostgresVectorStore
    matcher = HybridMatchingService(vector_store=PostgresVectorStore())
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class VectorRecord:
    namespace: str
    record_id: str
    vector: List[float]
    metadata: Dict[str, Any]


# ── In-Memory (dev/test) ──────────────────────────────────────────────────────

class InMemoryVectorStore:
    """
    In-memory vector store. Fast, zero-dependency.
    Use for local runs and tests.
    All data is lost when the process exits.
    """

    def __init__(self):
        self._records: Dict[str, List[VectorRecord]] = {}

    def upsert(self, namespace: str, records: Iterable[VectorRecord]) -> None:
        bucket = self._records.setdefault(namespace, [])
        existing = {r.record_id: i for i, r in enumerate(bucket)}
        for rec in records:
            if rec.record_id in existing:
                bucket[existing[rec.record_id]] = rec
            else:
                bucket.append(rec)

    def query(
        self,
        namespace: str,
        vector: List[float],
        top_k: int = 50,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        recs = self._records.get(namespace, [])
        results = []
        for rec in recs:
            if metadata_filter and not _metadata_matches(rec.metadata, metadata_filter):
                continue
            score = _cosine(vector, rec.vector)
            results.append({"record_id": rec.record_id, "score": score, "metadata": rec.metadata})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


# ── Postgres / pgvector (production) ─────────────────────────────────────────

class PostgresVectorStore:
    """
    pgvector-backed vector store for production use.

    Requires:
    - PostgreSQL with pgvector extension (docker-compose.yml includes this)
    - DATABASE_URL env var: postgresql://user:pass@host:port/dbname
    - pip install psycopg2-binary pgvector

    Schema is created automatically on first use (see migrations/001_pgvector.sql
    for the full production schema with HNSW indexes).

    Usage:
        store = PostgresVectorStore()
        # or with explicit connection string:
        store = PostgresVectorStore("postgresql://localhost:5432/millenia")
    """

    # Namespace → column name mapping in the investors table
    _NS_TO_COL = {
        "investor_fund_thesis_chunks": "fund_thesis_embedding",
        "investor_partner_bios": "partner_bio_embedding",
        "investor_portfolio_company_chunks": "portfolio_embedding",
        "investor_public_content_chunks": "activity_embedding",
    }

    # Reverse: store namespace → short cache namespace (used to reconstruct record_id)
    _NS_TO_CACHE_NS = {
        "investor_fund_thesis_chunks": "fund_thesis",
        "investor_partner_bios": "partner_bio",
        "investor_portfolio_company_chunks": "portfolio",
        "investor_public_content_chunks": "activity",
    }

    def __init__(self, connection_string: Optional[str] = None):
        self._dsn = connection_string or os.getenv("DATABASE_URL", "")
        if not self._dsn:
            raise ValueError(
                "DATABASE_URL must be set to use PostgresVectorStore. "
                "Example: postgresql://postgres:password@localhost:5432/millenia"
            )
        self._conn = None
        self._ensure_schema()

    def upsert(self, namespace: str, records: Iterable[VectorRecord]) -> None:
        import psycopg2.extras

        col = self._NS_TO_COL.get(namespace)
        if not col:
            self._upsert_generic(namespace, records)
            return

        records_list = list(records)
        if not records_list:
            return

        conn = self._connect()
        rows = [
            (
                rec.record_id.split(":")[0],
                self._fmt_vec(rec.vector),
                psycopg2.extras.Json(rec.metadata),
            )
            for rec in records_list
        ]

        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                f"""
                INSERT INTO investors (investor_id, {col}, metadata)
                VALUES %s
                ON CONFLICT (investor_id) DO UPDATE
                SET {col} = EXCLUDED.{col}
                """,
                rows,
                template="(%s, %s::vector, %s)",
                page_size=500,
            )
        conn.commit()
        logger.info(f"[pgvector] Upserted {len(rows)} records → namespace={namespace!r} col={col!r}")

    def query(
        self,
        namespace: str,
        vector: List[float],
        top_k: int = 50,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        col = self._NS_TO_COL.get(namespace)
        if not col:
            return self._query_generic(namespace, vector, top_k, metadata_filter)

        conn = self._connect()
        filter_clause = self._build_filter(metadata_filter)
        cache_ns = self._NS_TO_CACHE_NS.get(namespace, namespace)
        vec_str = self._fmt_vec(vector)

        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT investor_id,
                       1 - ({col} <=> %s::vector) AS score,
                       metadata
                FROM investors
                WHERE {col} IS NOT NULL
                  AND fund_status = 'active'
                  {filter_clause}
                ORDER BY {col} <=> %s::vector
                LIMIT %s
                """,
                (vec_str, vec_str, top_k),
            )
            return [
                {
                    "record_id": f"{row[0]}:{cache_ns}",
                    "score": max(0.0, min(1.0, float(row[1]))),
                    "metadata": row[2] or {},
                }
                for row in cur.fetchall()
            ]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _connect(self):
        import psycopg2
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self._dsn)
            # Register pgvector type
            try:
                from pgvector.psycopg2 import register_vector
                register_vector(self._conn)
            except ImportError:
                pass  # pgvector Python package not installed — vectors passed as lists
        return self._conn

    def _ensure_schema(self) -> None:
        """Create the minimum schema needed for vector search."""
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            # Minimal vector_records table (generic fallback for non-investor namespaces)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS vector_records (
                    namespace   TEXT NOT NULL,
                    record_id   TEXT NOT NULL,
                    embedding   vector,
                    metadata    JSONB,
                    PRIMARY KEY (namespace, record_id)
                )
            """)
            # Investors table — full schema in migrations/001_pgvector.sql
            cur.execute("""
                CREATE TABLE IF NOT EXISTS investors (
                    investor_id              TEXT PRIMARY KEY,
                    fund_status              TEXT DEFAULT 'active',
                    metadata                 JSONB,
                    fund_thesis_embedding    vector(1024),
                    partner_bio_embedding    vector(1024),
                    portfolio_embedding      vector(1024),
                    activity_embedding       vector(1024)
                )
            """)
        conn.commit()
        logger.info("[pgvector] Schema verified/created")

    def _upsert_generic(self, namespace: str, records: Iterable[VectorRecord]) -> None:
        import psycopg2.extras
        records_list = list(records)
        if not records_list:
            return
        conn = self._connect()
        rows = [
            (namespace, rec.record_id, self._fmt_vec(rec.vector), psycopg2.extras.Json(rec.metadata))
            for rec in records_list
        ]
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO vector_records (namespace, record_id, embedding, metadata)
                VALUES %s
                ON CONFLICT (namespace, record_id) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    metadata  = EXCLUDED.metadata
                """,
                rows,
                template="(%s, %s, %s::vector, %s)",
                page_size=500,
            )
        conn.commit()

    def _query_generic(
        self,
        namespace: str,
        vector: List[float],
        top_k: int,
        metadata_filter: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        conn = self._connect()
        vec_str = self._fmt_vec(vector)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT record_id,
                       1 - (embedding <=> %s::vector) AS score,
                       metadata
                FROM vector_records
                WHERE namespace = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vec_str, namespace, vec_str, top_k),
            )
            return [
                {"record_id": row[0], "score": float(row[1]), "metadata": row[2] or {}}
                for row in cur.fetchall()
            ]

    @staticmethod
    def _fmt_vec(vec: List[float]) -> str:
        """Format a Python list as pgvector literal '[x1,x2,...]' without needing the pgvector adapter."""
        return "[" + ",".join(repr(float(x)) for x in vec) + "]"

    @staticmethod
    def _build_filter(metadata_filter: Optional[Dict[str, Any]]) -> str:
        if not metadata_filter:
            return ""
        clauses = []
        for key, val in metadata_filter.items():
            if isinstance(val, list):
                # metadata stores JSON arrays; use @> containment per value (OR logic).
                # metadata->>'key' IN (...) was wrong — it extracts the whole array as
                # a raw string which never matches individual elements.
                subclauses = [
                    f"metadata->'{key}' @> '[\"{v}\"]'::jsonb"
                    for v in val
                ]
                clauses.append("(" + " OR ".join(subclauses) + ")")
            else:
                clauses.append(f"metadata->>'{key}' = '{val}'")
        return "AND " + " AND ".join(clauses) if clauses else ""


# ── Shared helpers ────────────────────────────────────────────────────────────

def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b))))


def _metadata_matches(metadata: Dict[str, Any], filt: Dict[str, Any]) -> bool:
    for key, val in filt.items():
        cur = metadata.get(key)
        if isinstance(val, list):
            if isinstance(cur, list):
                if not any(v in cur for v in val):
                    return False
            else:
                if cur not in val:
                    return False
        else:
            if cur != val:
                return False
    return True
