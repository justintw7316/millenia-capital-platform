from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from config import (
    FIT_SCORE_CAP,
    GENERIC_SECTOR_HINTS,
    INTENT_SCORE_CAP,
    INTENT_ACTIVE_THRESHOLD,
    RECENT_DEAL_DAYS,
    SECTOR_KEYWORDS,
    STAGE_KEYWORDS,
    TARGET_CHECK_SIZE,
    TARGET_GEOGRAPHY,
    TARGET_SECTOR,
    TARGET_STAGE,
)
from utils import field_blob, parse_date, parse_dates_in_text, parse_target_check_size_to_usd, to_clean_text, to_float

FINTECH_EXPLICIT_TERMS = {
    "fintech",
    "payments",
    "payment",
    "banking",
    "insurtech",
    "lending",
    "financial services",
    "credit",
    "credit card",
    "embedded finance",
    "wealthtech",
    "debt financing",
}

FINTECH_WEAK_TERMS = {
    "financial",
    "finance",
    "financing",
    "insurance",
    "capital markets",
}

INSTITUTIONAL_ENTITY_MARKERS = (
    "endowment",
    "foundation",
    "university",
    "nonprofit",
    "philanthrop",
    "charitable",
    "pension",
)

VENTURE_ENTITY_MARKERS = (
    "venture",
    "vc",
    "seed",
    "early stage",
    "angel",
    "accelerator",
    "fund",
    "startup",
)

RECENT_ACTIVITY_TERMS = ("investment", "invested", "funding", "round", "led", "backed", "portfolio")
RECENT_PUBLIC_SIGNAL_STRONG = (
    "announced investment",
    "led round",
    "portfolio announcement",
    "investment thesis",
    "funding announcement",
    "invested in",
    "led the round",
)
RECENT_PUBLIC_SIGNAL_MEDIUM = (
    "press release",
    "portfolio",
    "newsroom",
    "investment update",
    "backed by",
    "announced",
)


def _count_term_hits(text: str, terms) -> int:
    return sum(1 for term in terms if term in text)


def _is_institutional_nonventure_entity(merged: Dict) -> bool:
    """Flag entities that look more institutional allocators than startup investors."""
    blob = field_blob(
        merged,
        [
            "name",
            "description",
            "primary_type",
            "other_types",
            "investor_status",
            "preferred_investment_types",
        ],
    )
    has_institutional_marker = any(marker in blob for marker in INSTITUTIONAL_ENTITY_MARKERS)
    has_venture_marker = any(marker in blob for marker in VENTURE_ENTITY_MARKERS)
    return has_institutional_marker and not has_venture_marker


def _recent_dates_score(dates: List[datetime], today: datetime) -> Tuple[int, str]:
    if not dates:
        return 0, ""
    most_recent = max(dates)
    age_days = (today - most_recent).days
    if age_days <= RECENT_DEAL_DAYS:
        return 3, "dated evidence within target recency window"
    if age_days <= 180:
        return 2, "dated evidence within 6 months"
    if age_days <= 365:
        return 1, "dated evidence within 12 months"
    return 0, "dated evidence is stale"


def fit_sector_score(merged: Dict) -> Tuple[int, str]:
    """Score how clearly the investor matches the target sector."""
    target = TARGET_SECTOR.lower()
    target_terms = SECTOR_KEYWORDS.get(target, [target])
    sheet_blob = field_blob(
        merged,
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
    merged_blob = merged.get("_merged_sector_text", "")
    sheet_hits = [k for k in target_terms if k in sheet_blob]
    merged_hits = [k for k in target_terms if k in merged_blob]
    supp_hits = [k for k in target_terms if k in merged_blob and k not in sheet_blob]
    institutional = _is_institutional_nonventure_entity(merged)

    if target == "fintech":
        explicit_sheet_hits = _count_term_hits(sheet_blob, FINTECH_EXPLICIT_TERMS)
        explicit_merged_hits = _count_term_hits(merged_blob, FINTECH_EXPLICIT_TERMS)
        weak_finance_hits = _count_term_hits(merged_blob, FINTECH_WEAK_TERMS)

        if explicit_sheet_hits >= 2 or (explicit_sheet_hits >= 1 and explicit_merged_hits >= 3):
            if institutional:
                return 2, "merged: explicit fintech evidence, but entity appears institutional/non-venture"
            return 3, "merged: strong explicit fintech match (spreadsheet + public sources)"
        if explicit_sheet_hits >= 1:
            if institutional:
                return 1, "merged: some explicit fintech evidence, but entity appears institutional/non-venture"
            return 2, "spreadsheet: explicit fintech evidence"
        if explicit_merged_hits >= 2:
            return 1, "public sources: supplemental fintech evidence (no spreadsheet anchor)"
        if explicit_merged_hits >= 1:
            return 1, "public sources: limited fintech mention only"
        if weak_finance_hits >= 2 and not institutional:
            return 1, "merged: finance-oriented evidence, but not clearly fintech"
        if any(g in merged_blob for g in GENERIC_SECTOR_HINTS):
            return 0, "generic tech evidence only; not enough for fintech"
        return 0, "no clear fintech evidence"

    if len(sheet_hits) >= 2 or (len(sheet_hits) >= 1 and len(set(merged_hits)) >= 3):
        if institutional:
            return 2, "merged: strong sector match, but entity appears institutional/non-venture"
        return 3, "merged: strong sector match (spreadsheet anchor + public sources)"
    if len(sheet_hits) >= 1:
        if institutional:
            return 1, "merged: sector match, but entity appears institutional/non-venture"
        return 2, "spreadsheet: sector match"
    if len(set(merged_hits)) >= 2:
        return 1, "public sources: supplemental sector match (no spreadsheet anchor)"
    if supp_hits:
        return 1, "public sources: limited sector mention only"
    if any(g in merged_blob for g in GENERIC_SECTOR_HINTS):
        return 1, "merged: generic tech alignment only"
    return 0, "no clear sector evidence"


def fit_stage_score(merged: Dict) -> Tuple[int, str]:
    """Score stage fit using spreadsheet-first type and description evidence."""
    target = TARGET_STAGE.lower()
    keys = STAGE_KEYWORDS.get(target, [target])
    sheet_only = field_blob(merged, ["primary_type", "other_types", "preferred_investment_types", "description"])
    merged_text = merged.get("_merged_stage_text", "")
    institutional = _is_institutional_nonventure_entity(merged)
    if any(k in merged_text for k in keys):
        if institutional:
            return 1, "merged: some stage evidence, but entity appears institutional/non-venture"
        src = "spreadsheet" if any(k in sheet_only for k in keys) else "official site/search (sheet thin)"
        return 2, f"merged: direct stage/type evidence ({src})"
    stage_hints = {
        "seed": ["venture", "angel", "early"],
        "pre-seed": ["angel", "incubator", "accelerator"],
        "series a": ["venture", "growth equity"],
        "growth": ["growth", "late stage", "private equity"],
    }
    if any(k in merged_text for k in stage_hints.get(target, [])):
        if institutional:
            return 0, "entity appears institutional/non-venture despite partial stage hints"
        return 1, "merged: partial stage alignment"
    return 0, "no clear stage evidence"


def fit_check_size_score(merged: Dict) -> Tuple[int, str]:
    """Score check-size fit using structured spreadsheet fields only."""
    target_usd = parse_target_check_size_to_usd(TARGET_CHECK_SIZE)
    if not target_usd:
        return 0, "target check size not parseable"

    min_candidates = [
        to_float(merged.get("preferred_investment_amount_min")),
        to_float(merged.get("preferred_deal_size_min")),
    ]
    max_candidates = [
        to_float(merged.get("preferred_investment_amount_max")),
        to_float(merged.get("preferred_deal_size_max")),
    ]
    midpoint_candidates = [
        to_float(merged.get("preferred_investment_amount")),
        to_float(merged.get("preferred_deal_size")),
        to_float(merged.get("last_investment_size")),
    ]
    min_vals = [v for v in min_candidates if v is not None]
    max_vals = [v for v in max_candidates if v is not None]
    mids = [v for v in midpoint_candidates if v is not None]

    evidence: List[str] = []
    if min_vals or max_vals:
        low = min(min_vals) if min_vals else None
        high = max(max_vals) if max_vals else None
        if low is not None and high is not None:
            if low <= target_usd <= high:
                return 2, "spreadsheet: target check inside preferred min/max range"
            if target_usd < low * 0.35 or target_usd > high * 3.0:
                return 0, "spreadsheet: target check outside preferred range"
            return 1, "spreadsheet: target check near preferred range"
        if low is not None:
            ratio = target_usd / low if low else 0
            if 0.35 <= ratio <= 1.5:
                return 1, "spreadsheet: target check is directionally close to stated minimum"
            if ratio > 1.5:
                return 0, "spreadsheet: target check appears well above stated minimum"
        if high is not None:
            ratio = high / target_usd if target_usd else 0
            if 0.5 <= ratio <= 3.0:
                return 1, "spreadsheet: target check is directionally close to stated maximum"
            if ratio < 0.5:
                return 0, "spreadsheet: target check appears well above stated maximum"
        evidence.append("partial range")

    if mids:
        closest = min(mids, key=lambda x: abs(x - target_usd))
        ratio = closest / target_usd if target_usd else 0
        if 0.6 <= ratio <= 1.6:
            return 2, "spreadsheet: preferred/last check size close to target"
        if 0.25 <= ratio <= 4.0:
            return 1, "spreadsheet: some check-size compatibility"
        return 0, "spreadsheet: check-size evidence indicates mismatch"

    if evidence:
        return 1, "spreadsheet: limited check-size evidence"
    return 0, "no check-size evidence in spreadsheet"


def fit_geography_score(merged: Dict) -> Tuple[int, str]:
    """Score basic geography compatibility."""
    text = merged.get("_merged_geo_text", "")
    target = TARGET_GEOGRAPHY.lower()
    if target in text:
        return 1, "merged: target geography explicitly listed (sheet and/or site/search)"
    if target in {"u.s.", "us", "usa", "united states"} and any(
        t in text for t in ["united states", "u.s.", "us", "usa", "texas", "california", "new york"]
    ):
        return 1, "merged: U.S. geography inferred from HQ/preference or public pages"
    return 0, "no geography match evidence"


def fit_strategic_score(merged: Dict) -> Tuple[int, str]:
    """Score broader strategic alignment beyond literal sector labels."""
    target = TARGET_SECTOR.lower()
    terms = SECTOR_KEYWORDS.get(target, [target])
    sheet_blob = field_blob(
        merged,
        [
            "description",
            "preferred_industry",
            "preferred_verticals",
            "keywords",
            "latest_note",
            "last_investment_company",
        ],
    )
    merged_blob = merged.get("_merged_strategic_text", "")
    sheet_hits = sum(1 for t in terms if t in sheet_blob)
    supp_hits = sum(1 for t in terms if t in merged_blob and t not in sheet_blob)
    merged_hits = sum(1 for t in terms if t in merged_blob)
    institutional = _is_institutional_nonventure_entity(merged)

    if sheet_hits >= 2:
        if institutional:
            return 1, "merged: strategic evidence exists, but entity appears institutional/non-venture"
        return 2, "merged: strategic notes/portfolio align with target sector (spreadsheet)"
    if sheet_hits == 1 or merged_hits >= 2:
        if institutional:
            return 0, "entity appears institutional/non-venture despite partial strategic overlap"
        return 1, "merged: partial strategic alignment (sheet + official/search)"
    if supp_hits >= 1:
        if institutional:
            return 0, "light public strategic signal only; insufficient for institutional/non-venture entity"
        return 1, "merged: light strategic signal from public sources"
    return 0, "no strategic alignment evidence"


def compute_fit_score(merged: Dict) -> Tuple[int, Dict, List[str]]:
    """Compute the capped Fit score and its evidence trail."""
    sector, sector_ev = fit_sector_score(merged)
    stage, stage_ev = fit_stage_score(merged)
    check_size, check_ev = fit_check_size_score(merged)
    geo, geo_ev = fit_geography_score(merged)
    strat, strat_ev = fit_strategic_score(merged)
    total = min(FIT_SCORE_CAP, sector + stage + check_size + geo + strat)
    breakdown = {
        "Sector Match": sector,
        "Stage Match": stage,
        "Check Size Match": check_size,
        "Geography Match": geo,
        "Strategic Alignment": strat,
    }
    evidence = [
        f"sector[{sector}/3]: {sector_ev}",
        f"stage[{stage}/2]: {stage_ev}",
        f"check_size[{check_size}/2]: {check_ev}",
        f"geography[{geo}/1]: {geo_ev}",
        f"strategic[{strat}/2]: {strat_ev}",
    ]
    return total, breakdown, evidence


def intent_recent_investment(merged: Dict) -> Tuple[int, str]:
    """
    Score how recently the investor has deployed capital using both the
    structured spreadsheet activity fields and public/news text. Each source
    contributes independently and the higher signal wins, with a small boost
    when both corroborate.
    """
    n7 = to_float(merged.get("investments_7d")) or 0
    n6m = to_float(merged.get("investments_6m")) or 0
    n12m = to_float(merged.get("investments_12m")) or 0
    last_date = parse_date(merged.get("last_investment_date"))
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    sheet_score = 0
    sheet_reason = ""
    if n7 >= 1:
        sheet_score, sheet_reason = 3, "spreadsheet: investments in last 7 days"
    elif n6m >= 3:
        sheet_score, sheet_reason = 3, "spreadsheet: strong 6-month investment activity"
    elif n6m >= 1 or n12m >= 4:
        sheet_score, sheet_reason = 2, "spreadsheet: moderate recent investment activity"
    elif last_date:
        days = (today - last_date).days
        if days <= RECENT_DEAL_DAYS:
            sheet_score, sheet_reason = 3, "spreadsheet: very recent last investment date"
        elif days <= 180:
            sheet_score, sheet_reason = 2, "spreadsheet: last investment within 6 months"
        elif days <= 365:
            sheet_score, sheet_reason = 1, "spreadsheet: last investment within 12 months"

    web = merged.get("_merged_recent_context", "")
    web_score = 0
    web_reason = ""
    if web and any(t in web for t in RECENT_ACTIVITY_TERMS):
        dates = parse_dates_in_text(web)
        if dates:
            most_recent = max(dates)
            if most_recent >= today - timedelta(days=RECENT_DEAL_DAYS):
                web_score, web_reason = 2, "public/news: dated deal mention within target window"
            elif most_recent >= today - timedelta(days=180):
                web_score, web_reason = 1, "public/news: dated deal mention within 6 months"
            elif most_recent >= today - timedelta(days=365):
                web_score, web_reason = 1, "public/news: dated investment mention within 12 months"
        if web_score == 0:
            strong_ctx_hits = _count_term_hits(
                web, ("led", "funding", "round", "invested in", "announced investment")
            )
            if strong_ctx_hits >= 2:
                web_score, web_reason = 1, "public/news: repeated investment context"

    if sheet_score == 0 and web_score == 0:
        return 0, "no recent investment evidence in spreadsheet or public sources"

    combined = max(sheet_score, web_score)
    if sheet_score >= 2 and web_score >= 2:
        combined = 3

    if sheet_score and web_score:
        return combined, f"combined: {sheet_reason}; corroborated by {web_reason}"
    if sheet_score:
        return combined, sheet_reason
    return combined, web_reason


def intent_new_fund(merged: Dict) -> Tuple[int, str]:
    """
    Score whether the investor appears to have fresh fund capacity using both
    structured spreadsheet fields and public/news text. The higher signal wins
    and corroboration across both sources is reflected in the reason text.
    """
    fund_name = to_clean_text(merged.get("last_closed_fund_name"))
    fund_size = to_float(merged.get("last_closed_fund_size"))
    fund_date = parse_date(merged.get("last_closed_fund_close_date"))
    likely = to_clean_text(merged.get("most_likely_fundraising")).lower()
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    sheet_score = 0
    sheet_reason = ""
    if fund_name and fund_date:
        days = (today - fund_date).days
        if days <= 730:
            sheet_score, sheet_reason = 2, "spreadsheet: recently closed named fund"
        else:
            sheet_score, sheet_reason = 1, "spreadsheet: closed fund exists but older"
    elif fund_name and fund_size:
        sheet_score, sheet_reason = 1, "spreadsheet: fund name/size present without close date"
    elif likely in {"yes", "high", "likely"}:
        sheet_score, sheet_reason = 1, "spreadsheet: most likely fundraising flag"

    web = merged.get("_merged_fund_context", "")
    web_score = 0
    web_reason = ""
    fund_topic_match = any(term in web for term in ("new fund", "closed fund", "fundraise", "fundraising", "fund close"))
    fund_action_match = any(term in web for term in ("announced", "closed", "launch", "raise", "raises", "closes"))
    if fund_topic_match and fund_action_match:
        web_score, web_reason = 1, "public/news: explicit new fund / fundraising signal"

    if sheet_score == 0 and web_score == 0:
        return 0, "no new fund evidence in spreadsheet or public sources"

    combined = max(sheet_score, web_score)

    if sheet_score and web_score:
        return combined, f"combined: {sheet_reason}; corroborated by {web_reason}"
    if sheet_score:
        return combined, sheet_reason
    return combined, web_reason


def intent_hiring(merged: Dict) -> Tuple[int, str]:
    """Score explicit hiring activity that suggests platform momentum."""
    text = merged.get("_merged_hiring_context", "")
    has_hiring_signal = any(
        k in text
        for k in ["we are hiring", "hiring", "open role", "open position", "join our team", "careers", "job opening"]
    )
    has_investment_role = any(
        k in text
        for k in ["investment team", "venture associate", "investment associate", "principal", "analyst", "investor relations"]
    )
    if has_hiring_signal and has_investment_role:
        return 1, "merged public text: explicit hiring for investment-related role"
    return 0, "no explicit investment-team hiring evidence"


def intent_public_signals(merged: Dict) -> Tuple[int, str]:
    """Score public activity signals from notes and web evidence."""
    note = to_clean_text(merged.get("latest_note")).lower()
    web = merged.get("_merged_public_context", "")
    combined = f"{note} {web}".strip()
    strong_hits = _count_term_hits(combined, RECENT_PUBLIC_SIGNAL_STRONG)
    medium_hits = _count_term_hits(combined, RECENT_PUBLIC_SIGNAL_MEDIUM)
    dates = parse_dates_in_text(combined)
    dated_score, _ = _recent_dates_score(dates, datetime.now().replace(hour=0, minute=0, second=0, microsecond=0))

    if dated_score >= 2 and (strong_hits >= 1 or medium_hits >= 2):
        return 2, "merged: recent dated public investment signal"
    if dated_score >= 1 and strong_hits >= 1:
        return 1, "merged: dated public investment signal within past year"
    if strong_hits >= 2:
        return 1, "merged: undated but repeated public investment signal"
    if strong_hits >= 1 and medium_hits >= 1:
        return 1, "merged: moderate public investment signal"
    return 0, "no clear public investment signal"


def detect_warm_intro_signals(merged: Dict) -> List[str]:
    """
    Heuristic warm-intro signal detector aligned with the internal definition:
    shared network overlap, same geography, same industry, similar portfolio,
    public activity in the same space, and existing database / history hints.

    Intentionally LinkedIn-free: relies on dataset, official-site, and
    news/press evidence already merged upstream.
    """
    signals: List[str] = []

    geo_text = merged.get("_merged_geo_text", "")
    target_geo = TARGET_GEOGRAPHY.lower()
    geo_tokens = [target_geo]
    if target_geo in {"u.s.", "us", "usa", "united states"}:
        geo_tokens.extend(["united states", "u.s.", "usa", "texas", "california", "new york"])
    if any(t in geo_text for t in geo_tokens):
        signals.append("shared geography")

    sector_text = merged.get("_merged_sector_text", "")
    target_terms = SECTOR_KEYWORDS.get(TARGET_SECTOR.lower(), [TARGET_SECTOR.lower()])
    if any(t in sector_text for t in target_terms):
        signals.append("same industry")

    portfolio_ctx = merged.get("_merged_public_context", "")
    recent_ctx = merged.get("_merged_recent_context", "")
    combined_ctx = f"{portfolio_ctx} {recent_ctx}"
    portfolio_markers = ("portfolio", "invested in", "backed", "led round", "co-invest")
    if any(m in combined_ctx for m in portfolio_markers) and any(t in combined_ctx for t in target_terms):
        signals.append("similar portfolio")

    public_activity = ("investment thesis", "blog", "press", "announcement", "conference")
    if any(t in portfolio_ctx for t in public_activity) and any(t in portfolio_ctx for t in target_terms):
        signals.append("public activity in same space")

    if bool(to_clean_text(merged.get("description", ""))):
        signals.append("existing database / history")

    return signals


def intent_warm_intro(merged: Dict) -> Tuple[int, str]:
    """
    Warm Intro (0-2): relationship proximity + relevance score.
    Uses the shared `detect_warm_intro_signals` heuristic so the underlying
    Intent score and the business-facing Warm Intro Signal field always agree.
    """
    signals = detect_warm_intro_signals(merged)
    if len(signals) >= 3:
        return 2, f"strong proximity: {', '.join(signals)}"
    if len(signals) >= 1:
        return 1, f"moderate proximity: {', '.join(signals)}"
    return 0, "no relationship proximity signals"


def compute_intent_score(merged: Dict) -> Tuple[int, Dict, List[str]]:
    """Compute the capped Intent score and its evidence trail."""
    recent, recent_ev = intent_recent_investment(merged)
    fund, fund_ev = intent_new_fund(merged)
    hiring, hiring_ev = intent_hiring(merged)
    public, public_ev = intent_public_signals(merged)
    intro, intro_ev = intent_warm_intro(merged)

    total = min(INTENT_SCORE_CAP, recent + fund + hiring + public + intro)
    breakdown = {
        "Recent Investment": recent,
        "New Fund": fund,
        "Hiring": hiring,
        "Public Signals": public,
        "Warm Intro": intro,
    }
    evidence = [
        f"recent_investment[{recent}/3]: {recent_ev}",
        f"new_fund[{fund}/2]: {fund_ev}",
        f"hiring[{hiring}/1]: {hiring_ev}",
        f"public_signals[{public}/2]: {public_ev}",
        f"warm_intro[{intro}/2]: {intro_ev}",
    ]
    return total, breakdown, evidence


def classify_active(intent_score: int) -> str:
    return "Active" if intent_score >= INTENT_ACTIVE_THRESHOLD else "Inactive"


def classify_signal_type(intent_breakdown: Dict, fit_breakdown: Dict, merged: Dict) -> str:
    recent = intent_breakdown.get("Recent Investment", 0)
    new_fund = intent_breakdown.get("New Fund", 0)
    public = intent_breakdown.get("Public Signals", 0)
    sector = fit_breakdown.get("Sector Match", 0)

    if recent >= 2:
        return "Active Investing"
    if new_fund >= 1:
        likely = to_clean_text(merged.get("most_likely_fundraising", "")).lower()
        if likely in {"yes", "high", "likely"}:
            return "Fundraising"
        return "New Fund"
    if recent >= 1 and public >= 1:
        return "Active Investing"
    if public >= 2:
        return "Public Activity"
    if public >= 1:
        return "Similar Portfolio Activity"
    if recent >= 1:
        return "Active Investing"
    if sector >= 2:
        return "Sector Match Only"
    return "Weak / No Recent Signal"


def classify_sector_relevance(sector_score: int) -> str:
    if sector_score >= 3:
        return "High"
    if sector_score >= 1:
        return "Medium"
    return "Low"


def classify_recency(merged: Dict) -> str:
    """
    Aggregate every available recency signal (spreadsheet activity counts,
    structured dates, and parsed dates from official + news/press text) and
    pick the freshest bucket.
    """
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    candidate_days: List[int] = []

    n7 = to_float(merged.get("investments_7d")) or 0
    n6m = to_float(merged.get("investments_6m")) or 0
    n12m = to_float(merged.get("investments_12m")) or 0
    if n7 >= 1:
        candidate_days.append(7)
    if n6m >= 3:
        candidate_days.append(30)
    elif n6m >= 1:
        candidate_days.append(60)
    if n12m >= 4:
        candidate_days.append(60)
    elif n12m >= 1:
        candidate_days.append(90)

    last_date = parse_date(merged.get("last_investment_date"))
    if last_date:
        candidate_days.append((today - last_date).days)

    fund_date = parse_date(merged.get("last_closed_fund_close_date"))
    if fund_date:
        fund_days = (today - fund_date).days
        if fund_days <= 365:
            candidate_days.append(min(fund_days, 90))

    public_text = " ".join(
        piece
        for piece in (
            merged.get("_merged_recent_context", ""),
            merged.get("_merged_public_context", ""),
        )
        if piece
    )
    public_dates = parse_dates_in_text(public_text)
    if public_dates:
        candidate_days.append((today - max(public_dates)).days)

    if candidate_days:
        most_recent_days = min(d for d in candidate_days if d >= 0) if any(d >= 0 for d in candidate_days) else None
        if most_recent_days is not None:
            if most_recent_days <= 30:
                return "Last 30 days"
            if most_recent_days <= 60:
                return "Last 30-60 days"
            if most_recent_days <= 180:
                return "Last 60-90 days"

    activity_terms = ("investment", "invested", "funding", "round", "led", "backed", "portfolio", "announced")
    if public_text and sum(1 for t in activity_terms if t in public_text) >= 3:
        return "Last 60-90 days"

    return "Older / unclear"


def classify_confidence(has_website_evidence: bool, has_search_evidence: bool, total_score: int, recency: str) -> str:
    """
    Confidence reflects how well the score is supported by independent evidence.
    Two corroborating sources (official site + news/press) yield higher
    confidence; weak or single-source evidence is conservatively labeled.
    """
    evidence_sources = int(has_website_evidence) + int(has_search_evidence)
    recent = recency in ("Last 30 days", "Last 30-60 days")

    if evidence_sources == 0:
        return "Low"
    if evidence_sources >= 2 and total_score >= 10 and recent:
        return "High"
    if evidence_sources >= 2 and total_score >= 8:
        return "Medium"
    if evidence_sources >= 1 and total_score >= 6 and recent:
        return "Medium"
    return "Low"


def _activity_clause(merged: Dict, sector_label: str, sector_match: int, today: datetime) -> str:
    """Concrete recent-investment phrase (e.g. 'Invested in 3 fintech companies in the last 6 months, most recently Acme')."""
    n7 = int(to_float(merged.get("investments_7d")) or 0)
    n6m = int(to_float(merged.get("investments_6m")) or 0)
    n12m = int(to_float(merged.get("investments_12m")) or 0)
    last_co = to_clean_text(merged.get("last_investment_company"))
    last_inv_date = parse_date(merged.get("last_investment_date"))

    sector_qualifier = f"{sector_label} " if sector_match >= 2 else ""

    clause = ""
    if n7 >= 1:
        plural = "company" if n7 == 1 else "companies"
        clause = f"Invested in {n7} {sector_qualifier}{plural} in the last 7 days"
    elif n6m >= 1:
        plural = "company" if n6m == 1 else "companies"
        clause = f"Invested in {n6m} {sector_qualifier}{plural} in the last 6 months"
    elif n12m >= 1:
        plural = "company" if n12m == 1 else "companies"
        clause = f"Invested in {n12m} {plural} in the last 12 months"
    elif last_inv_date and (today - last_inv_date).days <= 365:
        co_part = f" ({last_co})" if last_co else ""
        clause = f"Last known investment in {last_inv_date.strftime('%b %Y')}{co_part}"

    if clause and last_co and ("companies" in clause or " company " in f" {clause} "):
        clause = f"{clause}, most recently {last_co}"
    return clause


def _fund_clause(merged: Dict, today: datetime) -> str:
    """Concrete fund-status phrase (e.g. 'recently closed Fund III' or 'actively fundraising')."""
    fund_name = to_clean_text(merged.get("last_closed_fund_name"))
    fund_date = parse_date(merged.get("last_closed_fund_close_date"))
    likely_raise = to_clean_text(merged.get("most_likely_fundraising")).lower()

    if fund_name and fund_date:
        days = (today - fund_date).days
        if days <= 730:
            return f"recently closed {fund_name}"
        if days <= 1825:
            return f"closed {fund_name} in {fund_date.strftime('%b %Y')}"
    if fund_name:
        return f"named fund: {fund_name}"
    if likely_raise in {"yes", "high", "likely"}:
        return "actively fundraising"
    return ""


def _fit_phrase(sector: int, stage: int, check: int, geo: int) -> str:
    if sector >= 3:
        if stage >= 2:
            return f"Strong {TARGET_SECTOR} and stage fit"
        if geo >= 1:
            return f"Strong {TARGET_SECTOR} fit in target geography"
        return f"Strong {TARGET_SECTOR} fit"
    if sector >= 2:
        if stage >= 2:
            return f"{TARGET_SECTOR.capitalize()} and stage match"
        if geo >= 1:
            return f"{TARGET_SECTOR.capitalize()} match in target geography"
        return f"{TARGET_SECTOR.capitalize()} sector match"
    if sector >= 1:
        if stage >= 2:
            return "Partial sector overlap and good stage fit"
        return "Some sector overlap"
    if stage >= 2:
        return "Good stage fit"
    if geo >= 1:
        return "Geographic match"
    return "Some alignment"


def _intent_phrase(recent: int, fund: int, hiring: int, public: int) -> str:
    if recent >= 3:
        if fund >= 1:
            return "very recent deals and fund activity"
        if hiring >= 1:
            return "very recent deals and team growth"
        if public >= 1:
            return "very recent deals and public signals"
        return "very recent deal activity"
    if recent >= 2:
        if public >= 1:
            return "recent deals and public signals"
        if fund >= 1:
            return "recent deals and fund activity"
        if hiring >= 1:
            return "recent deals and hiring signals"
        return "recent investing activity"
    if fund >= 2:
        return "a recently closed fund"
    if fund >= 1 and recent >= 1:
        return "fund activity and some deals"
    if fund >= 1:
        return "new fund or fundraising signals"
    if recent >= 1 and public >= 1:
        return "some deal activity and public signals"
    if recent >= 1:
        return "some recent deal activity"
    if public >= 1:
        return "public portfolio signals"
    if hiring >= 1:
        return "hiring for investment roles"
    return "some activity"


def _join_fit_intent(fit_desc: str, intent_desc: str) -> str:
    """Pick a clean separator so we never produce '...with X with Y...' phrasing."""
    if " with " in fit_desc.lower():
        return f"{fit_desc}; {intent_desc}."
    return f"{fit_desc} with {intent_desc}."


def generate_reason(
    fit_score: int,
    fit_breakdown: Dict,
    intent_score: int,
    intent_breakdown: Dict,
    merged: Dict = None,
) -> str:
    """
    Concrete 1-2 line reason grounded in actual data points (counts, last
    portfolio company, named fund, fund close date) when they are available,
    falling back to a fit/intent summary when they are not.
    """
    sector = fit_breakdown.get("Sector Match", 0)
    stage = fit_breakdown.get("Stage Match", 0)
    check = fit_breakdown.get("Check Size Match", 0)
    geo = fit_breakdown.get("Geography Match", 0)
    recent = intent_breakdown.get("Recent Investment", 0)
    fund = intent_breakdown.get("New Fund", 0)
    hiring = intent_breakdown.get("Hiring", 0)
    public = intent_breakdown.get("Public Signals", 0)
    total = fit_score + intent_score
    sector_label = TARGET_SECTOR

    if merged is not None:
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        activity_clause = _activity_clause(merged, sector_label, sector, today)
        fund_clause = _fund_clause(merged, today)

        if activity_clause and fund_clause:
            return f"{activity_clause} and {fund_clause}."
        if activity_clause:
            tags: List[str] = []
            if sector >= 2:
                tags.append(f"strong {sector_label} alignment")
            elif sector >= 1:
                tags.append(f"partial {sector_label} alignment")
            if check >= 2:
                tags.append("check-size aligned")
            if tags:
                return f"{activity_clause}; {', '.join(tags)}."
            return f"{activity_clause}."
        if fund_clause:
            sector_tag = f"strong {sector_label} alignment" if sector >= 2 else (
                f"partial {sector_label} alignment" if sector >= 1 else f"limited {sector_label} alignment"
            )
            return f"{fund_clause.capitalize()}; {sector_tag}."

    fit_desc = _fit_phrase(sector, stage, check, geo)
    intent_desc = _intent_phrase(recent, fund, hiring, public)
    has_fit = fit_score >= 4
    has_intent = intent_score >= 3

    if has_fit and has_intent:
        return _join_fit_intent(fit_desc, intent_desc)
    if has_fit:
        return f"{fit_desc}, but limited recent activity."
    if has_intent:
        return f"Shows {intent_desc}, but sector alignment is limited."
    if total >= 4:
        if sector >= 1:
            return "Some sector overlap, but signals are mixed."
        if recent >= 1 or fund >= 1:
            return "Some activity signals, but overall fit is unclear."
        return "Partial signals, but no strong fit or activity."
    return "No significant signals detected."


def normalize_score_to_100(total_score: int) -> int:
    """
    Translate the underlying 0-20 Fit+Intent score into a 0-100 score.
    Linear, explainable, and preserves the existing rubric's relative ordering.
    """
    cap = FIT_SCORE_CAP + INTENT_SCORE_CAP
    if cap <= 0:
        return 0
    return max(0, min(100, round((total_score / cap) * 100)))


def estimate_investment_probability(score_100: int, confidence: str) -> int:
    """
    Heuristic, intentionally conservative probability estimate (%) derived
    from the 0-100 score and confidence level. Not a calibrated statistical
    model; it's an explainable mapping the business team can sanity-check.

    Anchor points (before confidence adjustment):
        100 -> 80 %, 75 -> 55 %, 50 -> 30 %, 25 -> 12 %, 0 -> 2 %
    """
    score_100 = max(0, min(100, int(score_100)))
    base = round(2 + score_100 * 0.78)
    if confidence == "Low":
        base = round(base * 0.7)
    elif confidence == "High":
        base = round(min(85, base * 1.05))
    return max(0, min(85, int(base)))


def classify_check_size_fit(check_size_score: int, evidence_text: str = "") -> str:
    """
    Translate the 0-2 Check Size Match sub-score into a business label.
    Score 2 -> Perfect fit, 1 -> Slight mismatch, 0 -> Poor fit.
    """
    if check_size_score >= 2:
        return "Perfect fit"
    if check_size_score == 1:
        return "Slight mismatch"
    return "Poor fit"


def classify_warm_intro_signal(signals: List[str]) -> str:
    """
    Map heuristic warm-intro signal count to a business label.

    Signal count is driven by `detect_warm_intro_signals`, which uses the
    internal definition (shared network overlap, same geography, same
    industry, similar portfolio, public activity in same space, and existing
    database / history hints).
    """
    n = len(signals or [])
    if n >= 4:
        return "Direct connection"
    if n == 3:
        return "1st degree mutual"
    if n >= 1:
        return "2nd degree"
    return "None"


def classify_signal_bucket(score_100: int, recency: str, intent_score: int) -> str:
    """
    Four-tier business prioritization bucket:
        Immediate Target  - top score and fresh evidence
        High Priority     - solid score or strong intent, but timing is mixed
        Secondary         - directional fit, no urgency
        Ignore            - not enough signal to act on
    """
    fresh = recency in ("Last 30 days", "Last 30-60 days")
    if score_100 >= 75 and fresh:
        return "Immediate Target"
    if score_100 >= 70 and intent_score >= 5:
        return "Immediate Target"
    if score_100 >= 60:
        return "High Priority"
    if score_100 >= 50 and (fresh or intent_score >= 4):
        return "High Priority"
    if score_100 >= 35:
        return "Secondary"
    return "Ignore"


def classify_recommended_action(
    score_100: int,
    warm_intro_label: str,
    recency: str,
    intent_score: int,
) -> str:
    """Translate score + warmth + recency into a single suggested next step."""
    fresh = recency in ("Last 30 days", "Last 30-60 days")
    has_warm = warm_intro_label in ("Direct connection", "1st degree mutual")

    if score_100 >= 75 and fresh:
        if has_warm:
            return "Prioritize with warm intro"
        return "Reach out immediately"
    if score_100 >= 60 and has_warm:
        return "Prioritize with warm intro"
    if score_100 >= 60 and fresh:
        return "Reach out immediately"
    if score_100 >= 60 and intent_score >= 4:
        return "Reach out immediately"
    if score_100 >= 40:
        return "Monitor for now"
    return "Low priority"


def classify_urgency(score_100: int, recency: str, intent_score: int) -> str:
    """
    Time-boxed urgency hint:
        Act within 14 days - very recent + healthy score
        Act within 30 days - recent + decent score
        No urgency        - neither
    """
    if recency == "Last 30 days" and (score_100 >= 65 or intent_score >= 5):
        return "Act within 14 days"
    if recency in ("Last 30 days", "Last 30-60 days") and score_100 >= 50:
        return "Act within 30 days"
    if score_100 >= 60 and intent_score >= 4:
        return "Act within 30 days"
    return "No urgency"


def primary_data_sources(merged: Dict) -> str:
    """
    Concise human-readable description of the evidence sources actually
    used for this investor's score, derived from the merge debug record.
    """
    debug = merged.get("_merge_debug") or {}
    sources_present = debug.get("sources_present") or {}
    parts: List[str] = []
    if sources_present.get("spreadsheet"):
        parts.append("Investor database")
    if sources_present.get("official_site"):
        parts.append("Official site")
    if sources_present.get("news_press"):
        parts.append("News/press")
    if sources_present.get("search_snippets") and not sources_present.get("news_press"):
        parts.append("Public search")
    return ", ".join(parts) if parts else "Limited evidence"


def most_recent_evidence_date(merged: Dict) -> str:
    """
    Most recent evidence date (YYYY-MM-DD) backing this investor's score,
    drawing from structured spreadsheet dates and dates parsed from public/
    news text. Returns an empty string when no dated evidence is available.
    """
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    candidates: List[datetime] = []

    last_inv = parse_date(merged.get("last_investment_date"))
    if last_inv:
        candidates.append(last_inv)
    fund_date = parse_date(merged.get("last_closed_fund_close_date"))
    if fund_date:
        candidates.append(fund_date)

    public_text = " ".join(
        merged.get(key, "") or ""
        for key in ("_merged_recent_context", "_merged_public_context", "_merged_fund_context")
    )
    public_dates = parse_dates_in_text(public_text)
    if public_dates:
        candidates.append(max(public_dates))

    candidates = [d for d in candidates if d <= today]
    if not candidates:
        return ""
    return max(candidates).strftime("%Y-%m-%d")
