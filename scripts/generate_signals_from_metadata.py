#!/usr/bin/env python3
"""Generate deterministic activity signals from existing PitchBook metadata.

This is intentionally offline and repeatable. It converts already-ingested
PitchBook fields into data/signals/{investor_id}.json files so the activity
embedding namespace has real deployment/fundraising evidence before pgvector is
rebuilt.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR
from matching.repository import LocalInvestorRepository
from matching.schemas import InvestorCandidate

logger = logging.getLogger("generate_signals")
TODAY = date.today().isoformat()
GENERATED_TYPES = {
    "pitchbook_activity",
    "pitchbook_activity_count",
    "fundraising_status",
}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _make_signals(inv: InvestorCandidate, today: str = TODAY) -> list[dict]:
    """Build metadata-derived activity signals for one PitchBook investor."""
    meta = inv.metadata or {}
    source_url = _clean(meta.get("pitchbook_url"))
    signals: list[dict] = []

    last_company = _clean(meta.get("last_investment_company"))
    last_date = _clean(meta.get("last_investment_date"))
    last_type = _clean(meta.get("last_investment_type"))
    last_size = _clean(meta.get("last_investment_size"))
    if last_company or last_date:
        parts = ["Most recent PitchBook investment"]
        if last_company:
            parts.append(f"company: {last_company}")
        if last_date:
            parts.append(f"date: {last_date}")
        if last_type:
            parts.append(f"type: {last_type}")
        if last_size:
            parts.append(f"size: {last_size}")
        signals.append({
            "source_type": "pitchbook_activity",
            "text": " | ".join(parts),
            "source_url": source_url,
            "date": last_date or today,
            "confidence": 0.85,
            "visibility": "public",
        })

    activity_parts = []
    for label, key in [
        ("total investments", "total_investments"),
        ("investments past 5 years", "investments_5y"),
        ("investments past 24 months", "investments_24m"),
        ("active portfolio companies", "active_portfolio_count"),
    ]:
        value = _clean(meta.get(key))
        if value:
            activity_parts.append(f"{label}: {value}")
    count_12m = inv.fund.recent_investment_count_12m
    if count_12m:
        activity_parts.append(f"investments past 12 months: {count_12m}")
    if activity_parts:
        signals.append({
            "source_type": "pitchbook_activity_count",
            "text": "PitchBook activity counts | " + " | ".join(activity_parts),
            "source_url": source_url,
            "date": today,
            "confidence": 0.80,
            "visibility": "public",
        })

    fundraising = _clean(meta.get("most_likely_fundraising"))
    if fundraising:
        signals.append({
            "source_type": "fundraising_status",
            "text": f"PitchBook fundraising outlook: {fundraising}",
            "source_url": source_url,
            "date": today,
            "confidence": 0.75,
            "visibility": "public",
        })

    return signals


def _merge_existing(existing_payload: Any, generated: list[dict]) -> list[dict]:
    """Replace prior generated metadata signals while preserving other sources."""
    if isinstance(existing_payload, dict):
        existing = existing_payload.get("signals", [])
    elif isinstance(existing_payload, list):
        existing = existing_payload
    else:
        existing = []
    preserved = [
        s for s in existing
        if isinstance(s, dict) and s.get("source_type") not in GENERATED_TYPES
    ]
    return preserved + generated


def generate_signals(
    output_dir: Path,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    repo = LocalInvestorRepository()
    investors = [
        inv for inv in repo.load_all()
        if inv.investor_id.startswith("pb_")
    ]
    if limit is not None:
        investors = investors[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "output_dir": str(output_dir),
        "dry_run": dry_run,
        "pitchbook_investors_seen": len(investors),
        "files_written": 0,
        "investors_with_generated_signals": 0,
        "investors_without_metadata_signals": 0,
        "signal_type_counts": {},
        "sample_files": [],
    }
    type_counts = Counter()

    for inv in investors:
        generated = _make_signals(inv)
        if not generated:
            summary["investors_without_metadata_signals"] += 1
            continue
        summary["investors_with_generated_signals"] += 1
        type_counts.update(s["source_type"] for s in generated)
        out_path = output_dir / f"{inv.investor_id}.json"
        if not dry_run:
            existing_payload = None
            if out_path.exists():
                try:
                    existing_payload = json.loads(out_path.read_text())
                except json.JSONDecodeError:
                    existing_payload = None
            merged = _merge_existing(existing_payload, generated)
            out_path.write_text(json.dumps({
                "investor_id": inv.investor_id,
                "generated_at": summary["generated_at"],
                "signals": merged,
            }, indent=2))
            summary["files_written"] += 1
            if len(summary["sample_files"]) < 10:
                summary["sample_files"].append(str(out_path))

    summary["signal_type_counts"] = dict(type_counts)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "signals")
    parser.add_argument("--summary", type=Path, default=Path("outputs/vector_audit/metadata_signals_summary.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    summary = generate_signals(args.output_dir, limit=args.limit, dry_run=args.dry_run)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2))
    logger.info("Summary written to %s", args.summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
