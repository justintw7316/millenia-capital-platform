import csv
import json
from pathlib import Path
from typing import Dict, List

from config import DEBUG_TEXT_SUMMARY_CHARS
from utils import ensure_parent_dir, summarize_text

CSV_COLUMNS = [
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
]


def print_console_report(results: List[Dict]) -> None:
    """Print the concise business-facing report to the console."""
    print("\n" + "=" * 90)
    print("INVESTOR SIGNAL REPORT (sorted by Signal Score)")
    print("=" * 90)
    for i, row in enumerate(results, 1):
        print(f"{i}. {row['Investor']}")
        print(
            f"   Signal Score: {row['Signal Score']}/100 | "
            f"Probability: {row['Investment Probability (%)']}% | "
            f"Bucket: {row['Signal Bucket']} | "
            f"Action: {row['Recommended Action']}"
        )
        print(
            f"   Signal Type: {row['Signal Type']} | "
            f"Sector: {row['Sector Relevance']} | "
            f"Check Size: {row['Check Size Fit']} | "
            f"Warm Intro: {row['Warm Intro Signal']}"
        )
        print(
            f"   Recency: {row['Recency']} | "
            f"Urgency: {row['Urgency / Timing']} | "
            f"Confidence: {row['Confidence']}"
        )
        print(f"   Reason: {row['Reason']}")
        print(
            f"   Data Source: {row['Data Source']} | "
            f"Last Updated: {row['Last Updated'] or 'n/a'}"
        )
        print("-" * 90)


def write_csv_report(results: List[Dict], path: Path) -> None:
    """Persist the business-facing CSV output."""
    ensure_parent_dir(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(results)


def write_debug_jsonl(records: List[Dict], path: Path) -> None:
    """Persist one structured debug record per investor."""
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def write_run_summary(summary: Dict, path: Path) -> None:
    """Persist a run summary for repeatable local runs."""
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")


def summarize_evidence_record(record: Dict) -> Dict:
    """Trim large evidence blobs into a stable debug-friendly shape."""
    by_category = record.get("text_by_category") or {}
    return {
        "source_urls": list(record.get("source_urls") or []),
        "summary": summarize_text(record.get("combined_text", ""), DEBUG_TEXT_SUMMARY_CHARS),
        "by_category_summary": {
            key: summarize_text(value, DEBUG_TEXT_SUMMARY_CHARS)
            for key, value in by_category.items()
            if value
        },
        "meta": record.get("_meta") or {},
    }
