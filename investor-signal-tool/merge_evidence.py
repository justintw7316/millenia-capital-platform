"""
Merge spreadsheet fields with official-site and search evidence before scoring.

Spreadsheet, official-site text, and reputable news/press text are all treated as
core evidence and are always combined when present. DDGS snippets remain a
noise-controlled supplement that is only added when it strengthens target-term
coverage or when higher-trust evidence is still thin.
"""
from typing import Dict, List

from config import MERGE_THIN_THRESHOLDS, SECTOR_KEYWORDS, STAGE_KEYWORDS, TARGET_GEOGRAPHY, TARGET_SECTOR, TARGET_STAGE
from utils import dedupe_preserve_order, field_blob, to_clean_text


def _target_sector_terms() -> List[str]:
    t = TARGET_SECTOR.lower()
    return list(SECTOR_KEYWORDS.get(t, [t]))


def _target_stage_terms() -> List[str]:
    t = TARGET_STAGE.lower()
    return list(STAGE_KEYWORDS.get(t, [t]))


def _geo_tokens() -> List[str]:
    g = TARGET_GEOGRAPHY.lower()
    out = [g]
    if g in {"u.s.", "us", "usa", "united states"}:
        out.extend(["united states", "u.s.", "usa", "texas", "california", "new york"])
    return out


def _covers_target(text: str, target_terms: List[str]) -> int:
    return sum(1 for t in target_terms if t in text)


def _merge_text_field_tiered(
    sheet_blob: str,
    official_high: str,
    news_medium: str,
    snippet_low: str,
    target_terms: List[str],
    thin_threshold: int,
) -> str:
    """
    Combine spreadsheet, official-site, and news/press text as a single evidence
    blob. Each higher-trust source is always included when present. DDGS snippets
    are added only when they strengthen target-term coverage or higher-trust
    sources are still thin.
    """
    s = (sheet_blob or "").strip().lower()
    o = (official_high or "").strip().lower()
    n = (news_medium or "").strip().lower()
    q = (snippet_low or "").strip().lower()

    parts: List[str] = []
    if s:
        parts.append(s)
    if o:
        parts.append(o)
    if n:
        parts.append(n)

    merged_so_far = " ".join(parts)
    merged_covers = _covers_target(merged_so_far, target_terms)

    if q:
        if (
            not merged_so_far.strip()
            or len(merged_so_far) < thin_threshold
            or merged_covers == 0
            or _covers_target(q, target_terms) > merged_covers
        ):
            parts.append(q)

    return " ".join(parts).strip()


def build_merged_evidence(inv: Dict, website_evidence: Dict, search_signals: Dict) -> Dict:
    """
    Return an investor-shaped dict plus merged text dimensions.
    Scoring should read text dimensions from this dict instead of re-merging ad hoc.
    """
    off_site = (website_evidence.get("combined_text") or "").lower()
    off_ddgs = (search_signals.get("_ddgs_official_combined") or "").lower()
    official_high = f"{off_site} {off_ddgs}".strip()
    news_medium = (search_signals.get("_ddgs_news_combined") or "").lower()
    snippet_low = " ".join(
        (search_signals.get(k) or "") for k in ("recent_deal_text", "new_fund_text", "hiring_text", "public_signal_text")
    ).lower()

    cats: Dict[str, str] = dict(website_evidence.get("text_by_category") or {})
    ddgs_cats = search_signals.get("_ddgs_official_by_category")
    if isinstance(ddgs_cats, dict):
        for k, v in ddgs_cats.items():
            cats[k] = f"{cats.get(k, '')} {v or ''}".strip().lower()

    team_t = cats.get("team", "")
    contact_t = cats.get("contact", "")
    portfolio_t = cats.get("portfolio", "")
    careers_t = cats.get("careers", "")
    news_t = cats.get("news", "")
    general_t = cats.get("general", "")

    sector_sheet = field_blob(
        inv,
        [
            "description",
            "preferred_industry",
            "preferred_verticals",
            "all_industries",
            "keywords",
            "primary_industry_sector",
            "primary_industry_group",
        ],
    )
    sector_terms = _target_sector_terms()
    merged_sector = _merge_text_field_tiered(
        sector_sheet,
        official_high,
        news_medium,
        snippet_low,
        sector_terms,
        thin_threshold=MERGE_THIN_THRESHOLDS["default"],
    )

    stage_sheet = field_blob(inv, ["primary_type", "other_types", "preferred_investment_types", "description"])
    merged_stage = _merge_text_field_tiered(
        stage_sheet,
        official_high,
        news_medium,
        snippet_low,
        _target_stage_terms(),
        thin_threshold=MERGE_THIN_THRESHOLDS["default"],
    )

    geo_sheet = field_blob(inv, ["hq_location", "hq_city", "hq_state", "hq_country", "preferred_geography"])
    merged_geo = _merge_text_field_tiered(
        geo_sheet,
        official_high,
        news_medium,
        snippet_low,
        _geo_tokens(),
        thin_threshold=MERGE_THIN_THRESHOLDS["geography"],
    )

    strategic_sheet = field_blob(
        inv,
        [
            "description",
            "preferred_industry",
            "preferred_verticals",
            "keywords",
            "latest_note",
            "last_investment_company",
        ],
    )
    merged_strategic = _merge_text_field_tiered(
        strategic_sheet,
        official_high,
        news_medium,
        snippet_low,
        sector_terms,
        thin_threshold=MERGE_THIN_THRESHOLDS["default"],
    )

    sheet_portfolio = field_blob(
        inv,
        ["last_investment_company", "preferred_verticals", "all_industries", "preferred_industry", "latest_note"],
    )
    merged_portfolio_context = " ".join(
        x for x in [sheet_portfolio, portfolio_t, news_t, general_t, news_medium] if x
    ).lower()

    site_urls = list(website_evidence.get("source_urls") or [])
    ddgs_off_urls = list(search_signals.get("_ddgs_official_urls") or [])
    warm_source_urls = dedupe_preserve_order(site_urls + ddgs_off_urls)

    merged = dict(inv)
    merged["_merged_sector_text"] = merged_sector
    merged["_merged_stage_text"] = merged_stage
    merged["_merged_geo_text"] = merged_geo
    merged["_merged_strategic_text"] = merged_strategic
    merged["_merged_recent_context"] = " ".join(
        x for x in [official_high, news_medium, snippet_low] if x
    ).lower()
    merged["_merged_fund_context"] = " ".join(x for x in [official_high, news_medium, snippet_low] if x).lower()
    merged["_merged_hiring_context"] = " ".join(
        x for x in [official_high, news_medium, search_signals.get("hiring_text", ""), careers_t] if x
    ).lower()
    merged["_merged_public_context"] = " ".join(
        x for x in [to_clean_text(inv.get("latest_note", "")), official_high, news_medium, snippet_low] if x
    ).lower()
    merged["_merge_debug"] = {
        "official_source_urls": warm_source_urls,
        "news_source_urls": list(search_signals.get("_ddgs_news_urls") or []),
        "source_lengths": {
            "sheet_sector": len(sector_sheet),
            "official_high": len(official_high),
            "news_medium": len(news_medium),
            "snippet_low": len(snippet_low),
        },
        "sources_present": {
            "spreadsheet": bool(sector_sheet.strip()),
            "official_site": bool(official_high.strip()),
            "news_press": bool(news_medium.strip()),
            "search_snippets": bool(snippet_low.strip()),
        },
    }
    return merged
