"""
matching package — Hybrid investor-founder matching (local-first implementation).

This package provides a pluggable architecture aligned to a future
Postgres/pgvector/OpenSearch deployment, while running today with local JSON data
and in-memory vector search.
"""

from matching.hybrid_matcher import HybridMatchingService
from matching.profile_builder import build_company_profile_from_deal, build_company_profile_artifacts

__all__ = [
    "HybridMatchingService",
    "build_company_profile_from_deal",
    "build_company_profile_artifacts",
]
