"""
Company profile normalization and embedding artifact generation (Step 02 support).
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Any

from core.deal import Deal
from matching.embedder import LocalHashEmbedder
from matching.schemas import CompanyProfile


def build_company_profile_from_deal(deal: Deal) -> CompanyProfile:
    stage = _normalize_stage(deal.stage_label) or _infer_stage_from_raise(deal.raise_amount)
    geography = _split_geography(deal.geography) or ["United States"]
    problem = deal.problem or f"{deal.company_name} addresses a core problem in {deal.industry}."
    solution = deal.solution or ""
    market = deal.market_size or f"{deal.industry} market opportunity."
    traction = deal.traction_metrics or "Traction metrics pending detailed data room uploads; use placeholders until validated."
    raise_thesis_parts = [
        f"Raising ${deal.raise_amount:,.0f} over a 90-day process to fund growth milestones, investor outreach, and execution."
    ]
    if deal.use_of_funds_breakdown:
        raise_thesis_parts.append(f"Use of funds: {deal.use_of_funds_breakdown}")
    if deal.valuation:
        raise_thesis_parts.append(f"Valuation: ${deal.valuation:,.0f}")

    text_fields = {
        "company_core": f"{deal.company_name} is a {deal.industry} company raising ${deal.raise_amount:,.0f}.",
        "problem": problem,
        "solution": solution,
        "industry_market": f"{deal.industry} market opportunity. {market}",
        "traction": traction,
        "raise_thesis": " ".join(raise_thesis_parts),
        "founder_story": (
            f"Founder {deal.founder_name} is leading {deal.company_name}. "
            f"Contact: {deal.founder_email}. LinkedIn: {deal.founder_linkedin}. "
            f"Team: {deal.team_bios or 'Team bios pending.'}"
        ),
        "full_profile": deal.company_profile_text(),
    }
    return CompanyProfile(
        deal_id=deal.deal_id,
        company_name=deal.company_name,
        industry=deal.industry,
        company_website=deal.company_website,
        raise_amount=deal.raise_amount,
        stage=stage,
        geography=geography,
        subindustry=None,
        business_model="TBD",
        traction_metrics={"raw": deal.traction_metrics} if deal.traction_metrics else {},
        text_fields=text_fields,
        generated_at=datetime.utcnow().isoformat(),
    )


def build_company_profile_artifacts(deal: Deal, embedder: LocalHashEmbedder | None = None) -> Dict[str, Any]:
    embedder = embedder or LocalHashEmbedder()
    profile = build_company_profile_from_deal(deal)
    embeddings = {
        "company_core_embedding": embedder.embed(profile.text_fields.get("company_core", "")),
        "problem_embedding": embedder.embed(profile.text_fields.get("problem", "")),
        "industry_market_embedding": embedder.embed(profile.text_fields.get("industry_market", "")),
        "traction_embedding": embedder.embed(profile.text_fields.get("traction", "")),
        "raise_thesis_embedding": embedder.embed(profile.text_fields.get("raise_thesis", "")),
    }
    vector_metadata = {
        "collections": [
            "company_deal_profiles",
            "company_doc_chunks",
        ],
        "fields": {
            "entity_id": deal.deal_id,
            "entity_type": "company_deal_profile",
            "industry_tags": [deal.industry],
            "stage_tags": [profile.stage],
            "geos": profile.geography,
            "date_published": datetime.utcnow().strftime("%Y-%m-%d"),
            "freshness_bucket": "current",
            "confidence": 0.75,
            "visibility": "internal",
        },
    }
    return {
        "company_profile": profile.to_dict(),
        "embeddings": embeddings,
        "vector_db_design": vector_metadata,
        "embedding_model": embedder.model_name,
        "generated_at": datetime.utcnow().isoformat(),
    }


def _infer_stage_from_raise(raise_amount: float) -> str:
    if raise_amount <= 1_500_000:
        return "seed"
    if raise_amount <= 5_000_000:
        return "series_a"
    if raise_amount <= 15_000_000:
        return "series_b"
    return "growth"


def _normalize_stage(stage_label: str | None) -> str | None:
    if not stage_label:
        return None
    normalized = stage_label.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "preseed": "pre_seed",
        "pre_seed": "pre_seed",
        "seed": "seed",
        "series_a": "series_a",
        "series_b": "series_b",
        "series_c": "series_c",
        "growth": "growth",
        "buyout": "buyout",
    }
    return aliases.get(normalized)


def _split_geography(geography: str | None) -> list[str]:
    if not geography:
        return []
    return [part.strip() for part in geography.replace(";", ",").split(",") if part.strip()]
