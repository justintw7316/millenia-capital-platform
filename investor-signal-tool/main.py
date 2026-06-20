from config import (
    EXCEL_PATH,
    MAX_INVESTORS,
    MIN_SEARCH_SUMMARY_CHARS,
    MIN_WEBSITE_SUMMARY_CHARS,
    OUTPUT_CSV,
    OUTPUT_DEBUG_JSONL,
    OUTPUT_LOG,
    OUTPUT_RUN_SUMMARY_JSON,
    SEARCH_SIGNAL_KEYS,
    TARGET_CHECK_SIZE,
    TARGET_GEOGRAPHY,
    TARGET_SECTOR,
    TARGET_STAGE,
)
from data_loader import load_investors
from merge_evidence import build_merged_evidence
from output_writer import print_console_report, summarize_evidence_record, write_csv_report, write_debug_jsonl, write_run_summary
from scoring import (
    classify_check_size_fit,
    classify_confidence,
    classify_recency,
    classify_recommended_action,
    classify_sector_relevance,
    classify_signal_bucket,
    classify_signal_type,
    classify_urgency,
    classify_warm_intro_signal,
    compute_fit_score,
    compute_intent_score,
    detect_warm_intro_signals,
    estimate_investment_probability,
    generate_reason,
    most_recent_evidence_date,
    normalize_score_to_100,
    primary_data_sources,
)
from utils import setup_logging, summarize_text
from validation import validate_investor_input, validate_result_row, validate_score_consistency
from web_evidence import (
    discover_official_website,
    empty_search_signals,
    evaluate_search_strategy,
    fetch_website_evidence,
    search_investor_signals,
)


LOGGER = setup_logging(OUTPUT_LOG)

FIT_KEYS = (
    "Sector Match",
    "Stage Match",
    "Check Size Match",
    "Geography Match",
    "Strategic Alignment",
)
INTENT_KEYS = (
    "Recent Investment",
    "New Fund",
    "Hiring",
    "Public Signals",
    "Warm Intro",
)


def _empty_fit_breakdown():
    return {key: 0 for key in FIT_KEYS}


def _empty_intent_breakdown():
    return {key: 0 for key in INTENT_KEYS}


def _score_investor(merged):
    fit_score, fit_breakdown, fit_evidence = compute_fit_score(merged)
    intent_score, intent_breakdown, intent_evidence = compute_intent_score(merged)
    total_score = fit_score + intent_score
    return fit_score, fit_breakdown, fit_evidence, intent_score, intent_breakdown, intent_evidence, total_score


def _build_result_row(
    name,
    score_100,
    investment_probability,
    signal_type,
    sector_relevance,
    check_size_fit,
    warm_intro_signal,
    recency,
    confidence,
    reason,
    recommended_action,
    signal_bucket,
    urgency,
    data_source,
    last_updated,
):
    return {
        "Investor": name,
        "Signal Score": score_100,
        "Investment Probability (%)": investment_probability,
        "Signal Type": signal_type,
        "Sector Relevance": sector_relevance,
        "Check Size Fit": check_size_fit,
        "Warm Intro Signal": warm_intro_signal,
        "Recency": recency,
        "Confidence": confidence,
        "Reason": reason,
        "Recommended Action": recommended_action,
        "Signal Bucket": signal_bucket,
        "Urgency / Timing": urgency,
        "Data Source": data_source,
        "Last Updated": last_updated,
    }


def _build_debug_record(
    inv,
    status,
    failure_reason,
    failure_reasons,
    warning_reasons,
    processing_notes,
    website_evidence,
    search_signals,
    merged,
    fit_score,
    fit_breakdown,
    fit_evidence,
    intent_score,
    intent_breakdown,
    intent_evidence,
    total_score,
    result_row,
    score_100=None,
    investment_probability=None,
    warm_intro_signals=None,
):
    return {
        "investor": inv.get("name", ""),
        "website": inv.get("website", ""),
        "status": status,
        "failure_reason": failure_reason,
        "failure_reasons": failure_reasons,
        "warning_reasons": warning_reasons,
        "processing_notes": processing_notes,
        "business_output": dict(result_row),
        "raw_input": {
            "name": inv.get("name", ""),
            "website": inv.get("website", ""),
            "description": summarize_text(inv.get("description", "")),
            "preferred_industry": summarize_text(inv.get("preferred_industry", "")),
            "preferred_geography": summarize_text(inv.get("preferred_geography", "")),
        },
        "website_evidence": summarize_evidence_record(website_evidence),
        "search_evidence": {
            "source_urls": list(search_signals.get("_ddgs_official_urls") or [])
            + [url for url in list(search_signals.get("_ddgs_news_urls") or []) if url not in list(search_signals.get("_ddgs_official_urls") or [])],
            "snippet_summary": {
                key: summarize_text(search_signals.get(key, ""))
                for key in (
                    "recent_deal_text",
                    "new_fund_text",
                    "hiring_text",
                    "public_signal_text",
                    "outreach_text",
                )
                if search_signals.get(key)
            },
            "official_summary": summarize_text(search_signals.get("_ddgs_official_combined", "")),
            "news_summary": summarize_text(search_signals.get("_ddgs_news_combined", "")),
            "meta": search_signals.get("_meta") or {},
        },
        "merged_evidence": {
            "sector_summary": summarize_text(merged.get("_merged_sector_text", "")),
            "stage_summary": summarize_text(merged.get("_merged_stage_text", "")),
            "geo_summary": summarize_text(merged.get("_merged_geo_text", "")),
            "strategic_summary": summarize_text(merged.get("_merged_strategic_text", "")),
            "recent_context_summary": summarize_text(merged.get("_merged_recent_context", "")),
            "merge_debug": merged.get("_merge_debug") or {},
        },
        "scores": {
            "fit": fit_score,
            "intent": intent_score,
            "total": total_score,
            "score_100": score_100,
            "investment_probability": investment_probability,
            "warm_intro_signals": list(warm_intro_signals or []),
            "fit_breakdown": fit_breakdown,
            "intent_breakdown": intent_breakdown,
            "fit_evidence": fit_evidence,
            "intent_evidence": intent_evidence,
        },
    }


def _has_search_snippet_evidence(search_signals):
    return any(bool((search_signals.get(key) or "").strip()) for key in SEARCH_SIGNAL_KEYS)


def _classify_investor_reliability(inv, website_evidence, search_signals, processing_notes):
    website_meta = website_evidence.get("_meta") or {}
    search_meta = search_signals.get("_meta") or {}

    website_errors = list(website_meta.get("errors") or [])
    website_warnings = list(website_meta.get("warnings") or [])
    query_errors = list(search_meta.get("query_errors") or [])
    query_warnings = list(search_meta.get("query_warnings") or [])
    scrape_errors = list(search_meta.get("scrape_errors") or [])
    scrape_warnings = list(search_meta.get("scrape_warnings") or [])

    website_urls = list(website_evidence.get("source_urls") or [])
    search_urls = list(search_signals.get("_ddgs_official_urls") or []) + list(search_signals.get("_ddgs_news_urls") or [])
    has_search_snippets = _has_search_snippet_evidence(search_signals)

    has_website_evidence = bool(website_urls) or len((website_evidence.get("combined_text") or "").strip()) >= MIN_WEBSITE_SUMMARY_CHARS
    has_search_evidence = bool(search_urls) or has_search_snippets or len((search_signals.get("_ddgs_news_combined") or "").strip()) >= MIN_SEARCH_SUMMARY_CHARS
    has_any_web_evidence = has_website_evidence or has_search_evidence

    warning_reasons = []
    if not inv.get("website"):
        warning_reasons.append("website_missing_input")
    if website_warnings:
        warning_reasons.append("website_fetch_degraded")
    if query_warnings:
        warning_reasons.append("ddgs_query_fallback_used")
    if scrape_warnings:
        warning_reasons.append("ddgs_optional_scrape_blocked")

    failure_reasons = []
    if website_errors and not has_website_evidence and not has_search_evidence:
        failure_reasons.append("website_unreachable")
    elif website_errors and not has_website_evidence:
        warning_reasons.append("website_unreachable_but_search_available")

    if query_errors and not has_search_evidence:
        failure_reasons.append("ddgs_query_failed")
    elif query_errors:
        warning_reasons.append("ddgs_query_degraded_but_partial_results_available")

    if scrape_errors and not has_any_web_evidence:
        failure_reasons.append("search_scrape_failed")
    elif scrape_errors:
        warning_reasons.append("search_scrape_partially_failed")

    if inv.get("website") and not has_website_evidence and not website_errors:
        warning_reasons.append("website_present_but_weak_evidence")

    if not has_any_web_evidence:
        failure_reasons.append("no_web_evidence")

    if any("merge failed" in note for note in processing_notes):
        failure_reasons.append("merge_failed")
    if any("scoring failed" in note for note in processing_notes):
        failure_reasons.append("scoring_failed")
    if any("does not sum" in note or "missing result field" in note for note in processing_notes):
        failure_reasons.append("validation_failed")

    warning_reasons = sorted(set(warning_reasons))
    failure_reasons = sorted(set(failure_reasons))
    status = "partial_failure" if failure_reasons else "success"
    failure_reason = failure_reasons[0] if failure_reasons else ""

    return {
        "status": status,
        "failure_reason": failure_reason,
        "failure_reasons": failure_reasons,
        "warning_reasons": warning_reasons,
        "has_website_evidence": has_website_evidence,
        "has_search_evidence": has_search_evidence,
        "has_any_web_evidence": has_any_web_evidence,
    }


def process_investor(inv, index, total_investors):
    name = inv.get("name", f"Investor {index}")
    LOGGER.info("[%s/%s] Processing %s", index, total_investors, name)

    processing_notes = validate_investor_input(inv)
    website_evidence = {"combined_text": "", "source_urls": [], "text_by_category": {}}
    search_signals = empty_search_signals()
    merged = dict(inv)
    input_website = inv.get("website", "")
    effective_website = input_website

    if not effective_website:
        try:
            discovery = discover_official_website(name)
            if discovery.get("website"):
                effective_website = discovery["website"]
                processing_notes.append(f"website discovered via search: {effective_website}")
            else:
                processing_notes.append("website missing and no confident official website discovered")
        except Exception as exc:
            processing_notes.append(f"website discovery failed: {exc}")
            LOGGER.exception("Website discovery failed for %s", name)

    try:
        website_evidence = fetch_website_evidence(effective_website, name)
        website_evidence.setdefault("_meta", {})
        website_evidence["_meta"]["input_website"] = input_website
        website_evidence["_meta"]["effective_website"] = effective_website
    except Exception as exc:
        processing_notes.append(f"website evidence failed: {exc}")
        LOGGER.exception("Website evidence failed for %s", name)
        website_evidence = {"combined_text": "", "source_urls": [], "text_by_category": {}, "_meta": {"errors": [str(exc)]}}

    search_strategy = evaluate_search_strategy(website_evidence)
    if not effective_website:
        search_strategy = {
            "run_search": True,
            "query_keys": list(SEARCH_SIGNAL_KEYS),
            "reasons": ["official_website_missing_or_not_discovered"],
        }
    processing_notes.append(f"search strategy: {', '.join(search_strategy['reasons'])}")

    try:
        skip = set(website_evidence.get("source_urls") or [])
        search_signals = search_investor_signals(
            name,
            effective_website,
            skip_urls=skip,
            query_keys=search_strategy["query_keys"],
        )
        search_signals.setdefault("_meta", {})
        search_signals["_meta"]["input_website"] = input_website
        search_signals["_meta"]["effective_website"] = effective_website
        search_signals["_meta"]["strategy_reasons"] = list(search_strategy["reasons"])
    except Exception as exc:
        processing_notes.append(f"search evidence failed: {exc}")
        LOGGER.exception("Search evidence failed for %s", name)
        search_signals = {
            "_ddgs_official_combined": "",
            "_ddgs_news_combined": "",
            "_ddgs_official_by_category": {},
            "_ddgs_official_urls": [],
            "_ddgs_news_urls": [],
            "_meta": {
                "query_errors": [str(exc)],
                "scrape_errors": [],
                "input_website": input_website,
                "effective_website": effective_website,
                "strategy_reasons": list(search_strategy["reasons"]),
            },
        }
    try:
        merged = build_merged_evidence(inv, website_evidence, search_signals)
    except Exception as exc:
        processing_notes.append(f"merge failed: {exc}")
        LOGGER.exception("Merge failed for %s", name)
        merged = build_merged_evidence(
            inv,
            {"combined_text": "", "source_urls": [], "text_by_category": {}},
            {},
        )

    fit_score = 0
    intent_score = 0
    total_score = 0
    fit_breakdown = _empty_fit_breakdown()
    intent_breakdown = _empty_intent_breakdown()
    fit_evidence = ["fit[0]: scoring unavailable"]
    intent_evidence = ["intent[0]: scoring unavailable"]

    try:
        (
            fit_score,
            fit_breakdown,
            fit_evidence,
            intent_score,
            intent_breakdown,
            intent_evidence,
            total_score,
        ) = _score_investor(merged)
    except Exception as exc:
        processing_notes.append(f"scoring failed: {exc}")
        LOGGER.exception("Scoring failed for %s", name)

    consistency_issues = validate_score_consistency(
        fit_score,
        fit_breakdown,
        intent_score,
        intent_breakdown,
        total_score,
    )
    processing_notes.extend(consistency_issues)

    reliability = _classify_investor_reliability(inv, website_evidence, search_signals, processing_notes)

    signal_type = classify_signal_type(intent_breakdown, fit_breakdown, merged)
    sector_relevance = classify_sector_relevance(fit_breakdown.get("Sector Match", 0))
    recency = classify_recency(merged)
    confidence = classify_confidence(
        reliability["has_website_evidence"],
        reliability["has_search_evidence"],
        total_score,
        recency,
    )
    score_100 = normalize_score_to_100(total_score)
    investment_probability = estimate_investment_probability(score_100, confidence)
    check_size_fit = classify_check_size_fit(fit_breakdown.get("Check Size Match", 0))
    warm_intro_signals = detect_warm_intro_signals(merged)
    warm_intro_signal = classify_warm_intro_signal(warm_intro_signals)
    reason = generate_reason(fit_score, fit_breakdown, intent_score, intent_breakdown, merged)
    signal_bucket = classify_signal_bucket(score_100, recency, intent_score)
    recommended_action = classify_recommended_action(score_100, warm_intro_signal, recency, intent_score)
    urgency = classify_urgency(score_100, recency, intent_score)
    data_source = primary_data_sources(merged)
    last_updated = most_recent_evidence_date(merged)

    result_row = _build_result_row(
        name,
        score_100,
        investment_probability,
        signal_type,
        sector_relevance,
        check_size_fit,
        warm_intro_signal,
        recency,
        confidence,
        reason,
        recommended_action,
        signal_bucket,
        urgency,
        data_source,
        last_updated,
    )
    processing_notes.extend(validate_result_row(result_row))
    if reliability["warning_reasons"]:
        processing_notes.extend(f"warning: {r}" for r in reliability["warning_reasons"])
    if reliability["failure_reasons"]:
        processing_notes.extend(f"failure: {r}" for r in reliability["failure_reasons"])

    debug_record = _build_debug_record(
        inv,
        reliability["status"],
        reliability["failure_reason"],
        reliability["failure_reasons"],
        reliability["warning_reasons"],
        processing_notes,
        website_evidence,
        search_signals,
        merged,
        fit_score,
        fit_breakdown,
        fit_evidence,
        intent_score,
        intent_breakdown,
        intent_evidence,
        total_score,
        result_row,
        score_100=score_100,
        investment_probability=investment_probability,
        warm_intro_signals=warm_intro_signals,
    )
    return result_row, debug_record, reliability["status"], reliability["has_website_evidence"]


def main():
    LOGGER.info("Loading investors from %s", EXCEL_PATH)
    investors = load_investors(EXCEL_PATH, MAX_INVESTORS)
    LOGGER.info("Loaded %s investors", len(investors))
    LOGGER.info(
        "Deal criteria | Sector=%s | Stage=%s | Check Size=%s | Geography=%s",
        TARGET_SECTOR,
        TARGET_STAGE,
        TARGET_CHECK_SIZE,
        TARGET_GEOGRAPHY,
    )

    results = []
    debug_records = []
    succeeded = 0
    partial_failed = 0
    no_website_evidence = 0
    failure_reason_counts = {}

    for idx, inv in enumerate(investors, start=1):
        result_row, debug_record, status, has_website_evidence = process_investor(inv, idx, len(investors))
        results.append(result_row)
        debug_records.append(debug_record)

        if status == "success":
            succeeded += 1
        else:
            partial_failed += 1
        if not has_website_evidence:
            no_website_evidence += 1
        if debug_record.get("failure_reason"):
            failure_reason_counts[debug_record["failure_reason"]] = failure_reason_counts.get(debug_record["failure_reason"], 0) + 1

    results.sort(key=lambda row: row["Signal Score"], reverse=True)
    debug_records.sort(key=lambda record: record["scores"]["total"], reverse=True)

    print_console_report(results)
    write_csv_report(results, OUTPUT_CSV)
    write_debug_jsonl(debug_records, OUTPUT_DEBUG_JSONL)

    run_summary = {
        "investors_processed": len(investors),
        "succeeded": succeeded,
        "partially_failed": partial_failed,
        "no_website_evidence": no_website_evidence,
        "partial_failure_reason_counts": failure_reason_counts,
        "output_files": {
            "csv": str(OUTPUT_CSV),
            "debug_jsonl": str(OUTPUT_DEBUG_JSONL),
            "run_summary_json": str(OUTPUT_RUN_SUMMARY_JSON),
            "log": str(OUTPUT_LOG),
        },
    }
    write_run_summary(run_summary, OUTPUT_RUN_SUMMARY_JSON)

    print("\nRun summary")
    print(f"- investors processed: {run_summary['investors_processed']}")
    print(f"- succeeded: {run_summary['succeeded']}")
    print(f"- partially failed: {run_summary['partially_failed']}")
    print(f"- no website evidence: {run_summary['no_website_evidence']}")
    print(f"- CSV output: {OUTPUT_CSV}")
    print(f"- Debug JSONL output: {OUTPUT_DEBUG_JSONL}")
    print(f"- Run summary JSON: {OUTPUT_RUN_SUMMARY_JSON}")
    print(f"- Log file: {OUTPUT_LOG}")
    LOGGER.info("Pipeline complete")
    return results


if __name__ == "__main__":
    main()
