"""
Disk-based embedding cache for investor vectors.

Prevents re-embedding all investors on every run_match() call.
Cache is automatically invalidated when the embedding model changes.

Content-hash cache invalidation (P2-F-hash):
    Each namespace entry stores both the vector AND an MD5 hash of the
    text that was embedded.  On subsequent precompute runs only investors
    whose text actually changed for a given namespace are re-embedded,
    cutting the typical incremental run from ~2 hours to seconds/minutes.

On-disk format (v2 — with text hashes):
    {
        "model": "sentence-transformer-BAAI/bge-m3",
        "investor_count": 35277,
        "namespaces": ["fund_thesis", "partner_bio", "portfolio", "activity"],
        "embeddings": {
            "pb_inv123": {
                "fund_thesis": {"v": [0.12, -0.45, ...], "h": "a3f7b2..."},
                "partner_bio": {"v": [...],               "h": "c9d1e4..."},
                "portfolio":   {"v": [...],               "h": "00f812..."},
                "activity":    {"v": [...],               "h": "7a2b5c..."}
            }
        }
    }

Backward compatibility (v1 format — no hashes):
    Old entries stored the raw vector list directly:
        "fund_thesis": [0.12, -0.45, ...]
    On load these are silently migrated to {"v": [...], "h": ""}.
    An empty hash ("") is treated as "no baseline yet — re-embed on the next
    precompute run." This is deliberately conservative: without the original
    text hash, we cannot prove the old vector matches the current extractor.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

NAMESPACES = ["fund_thesis", "partner_bio", "portfolio", "activity"]

# Internal entry type after load/migration
_Entry = Dict[str, Union[List[float], str]]  # {"v": [...], "h": "md5hex"}


class EmbeddingCache:
    """
    Persists investor embeddings to disk so they survive across pipeline runs.

    Cache is invalidated automatically when the embedding model changes.
    Individual namespace entries are re-embedded only when their source text
    changes (detected via MD5 content hash).
    """

    def __init__(self, model_name: str, cache_path: Path, preload: bool = True):
        self._model_name = model_name
        self._cache_path = Path(cache_path)
        self._embeddings: Dict[str, Dict[str, _Entry]] = {}
        self._dirty = False
        if preload:
            self._load()

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def compute_hash(text: str) -> str:
        """Return a stable MD5 hex digest of *text*.

        Used to detect when the text that was embedded for a given
        (investor_id, namespace) pair has changed since the last precompute run.
        """
        return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()

    # ── Public interface ──────────────────────────────────────────────────────

    def get(self, investor_id: str, namespace: str) -> Optional[List[float]]:
        """Return cached vector or None if not cached."""
        entry = self._embeddings.get(investor_id, {}).get(namespace)
        if entry is None:
            return None
        # Handle both v2 dict format and any stray v1 list still in memory
        if isinstance(entry, list):
            return entry
        return entry.get("v")  # type: ignore[return-value]

    def set(self, investor_id: str, namespace: str, vector: List[float]) -> None:
        """Store a vector without changing the text hash.

        Used by HybridMatchingService._index_candidates() (no hash context
        available there).  Preserves any existing hash so backfilled entries
        are not accidentally wiped.
        """
        if investor_id not in self._embeddings:
            self._embeddings[investor_id] = {}
        existing = self._embeddings[investor_id].get(namespace, {})
        existing_h = existing.get("h", "") if isinstance(existing, dict) else ""
        self._embeddings[investor_id][namespace] = {"v": vector, "h": existing_h}
        self._dirty = True

    def set_with_hash(
        self, investor_id: str, namespace: str, vector: List[float], text_hash: str
    ) -> None:
        """Store a vector AND its text hash.

        Use this in the precompute script so future runs can detect text
        changes and skip re-embedding unchanged investors.
        """
        if investor_id not in self._embeddings:
            self._embeddings[investor_id] = {}
        self._embeddings[investor_id][namespace] = {"v": vector, "h": text_hash}
        self._dirty = True

    def get_text_hash(self, investor_id: str, namespace: str) -> Optional[str]:
        """Return the stored text hash for (investor_id, namespace).

        Returns:
            None  — no cache entry at all (investor never embedded)
            ""    — v1/old entry: vector exists but no hash was stored
            "..." — MD5 hex of the text that was last embedded
        """
        entry = self._embeddings.get(investor_id, {}).get(namespace)
        if entry is None:
            return None
        if isinstance(entry, list):
            return ""  # v1 stray in memory
        return entry.get("h", "")

    def text_changed(self, investor_id: str, namespace: str, new_hash: str) -> bool:
        """Return True if this namespace needs re-embedding.

        Logic:
            • No entry (None) → True  (never embedded — must embed)
            • Stored hash ""  → True  (v1 entry, no baseline; re-embed)
            • Stored hash == new_hash → False (text unchanged — skip)
            • Stored hash != new_hash → True  (text changed — re-embed)
        """
        stored = self.get_text_hash(investor_id, namespace)
        if stored is None:
            return True   # no vector at all
        if stored == "":
            return True    # v1 entry — no trustworthy baseline
        return stored != new_hash

    def has_all_namespaces(self, investor_id: str) -> bool:
        """Return True if all 4 namespace vectors are cached for this investor."""
        return all(self.get(investor_id, ns) is not None for ns in NAMESPACES)

    def save(self) -> None:
        """Persist the cache to disk. No-op if nothing changed."""
        if not self._dirty:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model_name,
            "investor_count": len(self._embeddings),
            "namespaces": NAMESPACES,
            "embeddings": self._embeddings,
        }
        with open(self._cache_path, "w") as f:
            json.dump(payload, f)
        logger.info(
            "[EmbeddingCache] Saved %d investor embeddings → %s",
            len(self._embeddings),
            self._cache_path,
        )
        self._dirty = False

    def invalidate(self) -> None:
        """Wipe all cached embeddings (e.g. after a model upgrade or --force)."""
        self._embeddings = {}
        self._dirty = True
        logger.info("[EmbeddingCache] Cache invalidated — all embeddings cleared")

    @property
    def size(self) -> int:
        return len(self._embeddings)

    @property
    def model_name(self) -> str:
        return self._model_name

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._cache_path.exists():
            logger.info("[EmbeddingCache] No cache at %s — starting fresh", self._cache_path)
            return
        try:
            with open(self._cache_path) as f:
                data = json.load(f)
            cached_model = data.get("model", "")
            if cached_model != self._model_name:
                logger.warning(
                    "[EmbeddingCache] Model mismatch: cache=%r, current=%r — cache invalidated",
                    cached_model,
                    self._model_name,
                )
                return
            raw = data.get("embeddings", {})
            # Migrate v1 entries (raw list) → v2 dict format {"v": [...], "h": ""}
            migrated = 0
            for inv_id, ns_data in raw.items():
                for ns, val in list(ns_data.items()):
                    if isinstance(val, list):
                        ns_data[ns] = {"v": val, "h": ""}
                        migrated += 1
            self._embeddings = raw
            if migrated:
                logger.info(
                    "[EmbeddingCache] Migrated %d v1 entries to v2 (re-embed required)",
                    migrated,
                )
            logger.info(
                "[EmbeddingCache] Loaded %d cached embeddings (model=%r)",
                len(self._embeddings),
                self._model_name,
            )
        except Exception as e:
            logger.warning("[EmbeddingCache] Failed to load cache: %s — starting fresh", e)
