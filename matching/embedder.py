"""
Embedding abstraction with pluggable embedders.

Two implementations:
- LocalHashEmbedder: deterministic hash-based (no dependencies, fast, low quality)
- SentenceTransformerEmbedder: real AI embeddings via sentence-transformers (high quality)

Recommended free models (all open-weight, Apache 2.0 / MIT):
- BAAI/bge-m3          1024-dim, 8192-token context, hybrid dense+sparse  [DEFAULT]
- mixedbread-ai/mxbai-embed-large-v1  1024-dim, 512-token context, highest MTEB (64.7)
- nomic-ai/nomic-embed-text-v1.5      768-dim,  8192-token context, Matryoshka flexible dims
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from typing import List

# Free, open-weight model presets — all runnable locally via sentence-transformers
BGE_M3 = "BAAI/bge-m3"                              # 1024-dim, 8192-token context  [recommended]
MXBAI_LARGE = "mixedbread-ai/mxbai-embed-large-v1"  # 1024-dim, 512-token context
NOMIC_V1_5 = "nomic-ai/nomic-embed-text-v1.5"       # 768-dim,  8192-token context, Matryoshka

# Dimension map for the presets (used for zero-vector fallback)
_MODEL_DIMS = {
    BGE_M3: 1024,
    MXBAI_LARGE: 1024,
    NOMIC_V1_5: 768,
}


class LocalHashEmbedder:
    """Deterministic hash-based embedder. Fast but low semantic quality."""

    def __init__(self, dim: int = 96):
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"[a-zA-Z0-9_#+.-]+", (text or "").lower())
        if not tokens:
            return vec
        for tok in tokens:
            h = hashlib.sha256(tok.encode("utf-8")).hexdigest()
            idx = int(h[:8], 16) % self.dim
            sign = -1.0 if int(h[8:10], 16) % 2 else 1.0
            weight = 1.0 + (len(tok) / 20.0)
            vec[idx] += sign * weight
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]

    @property
    def model_name(self) -> str:
        return f"local-hash-embedder-{self.dim}d"


class SentenceTransformerEmbedder:
    """Real AI embeddings via sentence-transformers.

    Defaults to BAAI/bge-m3 (1024-dim, 8192-token context).
    Pass model_id=MXBAI_LARGE or model_id=NOMIC_V1_5 for alternatives.
    """

    def __init__(
        self,
        model_id: str = BGE_M3,
        allow_download: bool | None = None,
        fallback_dim: int | None = None,
    ):
        self._model_id = model_id
        if allow_download is None:
            allow_download = os.getenv("MATCHING_ALLOW_MODEL_DOWNLOAD", "").lower() in {
                "1", "true", "yes", "on"
            }
        self._allow_download = allow_download
        self._dim = fallback_dim or _MODEL_DIMS.get(model_id, 1024)
        self._fallback = LocalHashEmbedder(dim=self._dim)
        self._fallback_active = False
        self._model = None  # lazy-loaded

    def _load_model(self):
        if self._fallback_active:
            return None
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            kwargs = {}
            if not self._allow_download:
                # Default to local-files-only to keep CLI/tests deterministic in offline environments.
                kwargs["local_files_only"] = True
            if self._model_id == NOMIC_V1_5:
                kwargs["trust_remote_code"] = True
            try:
                self._model = SentenceTransformer(self._model_id, **kwargs)
            except Exception:
                self._fallback_active = True
                self._model = None
        return self._model

    def embed(self, text: str) -> List[float]:
        model = self._load_model()
        if model is None:
            return self._fallback.embed(text)
        if not text or not text.strip():
            return [0.0] * self._dim
        try:
            vec = model.encode(text, normalize_embeddings=True)
            return vec.tolist()
        except Exception:
            self._fallback_active = True
            return self._fallback.embed(text)

    @property
    def model_name(self) -> str:
        if self._fallback_active:
            return f"{self._fallback.model_name}(fallback)"
        return f"sentence-transformer-{self._model_id}"
