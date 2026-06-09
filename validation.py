from typing import Dict, List

from config import REQUIRED_INVESTOR_FIELDS
from utils import to_clean_text, website_domain


def validate_investor_input(inv: Dict) -> List[str]:
    """Return lightweight validation warnings for one investor record."""
    warnings: List[str] = []

    for field in REQUIRED_INVESTOR_FIELDS:
        if not to_clean_text(inv.get(field)):
            warnings.append(f"missing required field: {field}")

    website = to_clean_text(inv.get("website"))
    if website and not website_domain(website):
        warnings.append("website field is present but malformed")

    return warnings


def validate_score_consistency(
    fit_score: int,
    fit_breakdown: Dict[str, int],
    intent_score: int,
    intent_breakdown: Dict[str, int],
    total_score: int,
) -> List[str]:
    """Ensure breakdown arithmetic matches exported totals."""
    issues: List[str] = []

    if sum(fit_breakdown.values()) != fit_score:
        issues.append("fit breakdown does not sum to Fit Score")
    if sum(intent_breakdown.values()) != intent_score:
        issues.append("intent breakdown does not sum to Intent Score")
    if fit_score + intent_score != total_score:
        issues.append("Fit Score + Intent Score does not equal Total Score")

    return issues


def validate_result_row(row: Dict) -> List[str]:
    """Check the exported CSV row shape for obvious consistency problems."""
    required_fields = (
        "Investor",
        "Signal Score",
        "Investment Probability (%)",
        "Signal Type",
        "Sector Relevance",
        "Check Size Fit",
        "Warm Intro Signal",
        "Recency",
        "Confidence",
        "Reason",
        "Recommended Action",
        "Signal Bucket",
        "Urgency / Timing",
        "Data Source",
        "Last Updated",
    )
    issues: List[str] = []
    for field in required_fields:
        if field not in row:
            issues.append(f"missing result field: {field}")

    if row.get("Investor") in {"", None}:
        issues.append("result row is missing investor name")

    score = row.get("Signal Score")
    if isinstance(score, (int, float)) and not (0 <= score <= 100):
        issues.append("Signal Score outside 0-100 range")

    probability = row.get("Investment Probability (%)")
    if isinstance(probability, (int, float)) and not (0 <= probability <= 100):
        issues.append("Investment Probability outside 0-100 range")

    return issues
