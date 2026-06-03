#!/usr/bin/env python3
"""
Precompute BGE-M3 embeddings for all investors and persist to disk cache.

Run once after loading PitchBook data (and again when a new city export is added):

    MATCHING_ALLOW_MODEL_DOWNLOAD=true python3 scripts/precompute_embeddings.py

Options:
    --model       Embedding model ID (default: BAAI/bge-m3)
    --batch-size  Internal sentence-transformers mini-batch size (default: 32)
    --chunk-size  Texts per encode() call (default: 1024)
    --force       Re-embed all investors even if already cached
    --pgvector    After caching, push vectors to PostgreSQL (requires DATABASE_URL)

The script:
1. Downloads the model on first run (cached by sentence-transformers in ~/.cache)
2. Loads all investors from every data/investors/*.json export
3. Uses content-hash cache invalidation (P2-F-hash):
   - Computes MD5 hash of the text for each (investor, namespace) pair
   - Skips pairs whose text hash matches the stored hash (text unchanged)
   - Re-embeds old v1 cache entries that lack hashes (conservative migration)
   - Only re-embeds pairs where text actually changed or is new
4. Saves the cache to data/investors/embeddings_cache.json
5. Optionally pushes vectors to PostgreSQL via PostgresVectorStore

On first run after deploying this update, OLD cache entries (without hashes)
are re-embedded because the prior text hash is unknown. Later runs can skip
unchanged pairs safely.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is on path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("MATCHING_ALLOW_MODEL_DOWNLOAD", "true")

from config import DATA_DIR
from matching.embedder import BGE_M3, SentenceTransformerEmbedder
from matching.embedding_cache import EmbeddingCache
from matching.repository import LocalInvestorRepository
from matching.schemas import InvestorCandidate
from matching.vector_store import InMemoryVectorStore, VectorRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("precompute")


# ── Namespace → text extractor ────────────────────────────────────────────────
# Each namespace is embedded separately so the vector store can do per-namespace
# similarity lookups (e.g. match on "fund thesis" vs "portfolio adjacency").
# Richer text = better BGE-M3 embeddings = more accurate matches.

def _text_for_namespace(inv: InvestorCandidate, ns: str) -> str:
    if ns == "fund_thesis":
        # Combine thesis + sector + stage focus for maximum semantic coverage.
        # Thesis alone is often too short or missing for standard_list investors.
        parts = []
        thesis = inv.fund.thesis_text or inv.investment_thesis
        if thesis:
            parts.append(thesis)
        if inv.fund.industry_focus:
            parts.append("Focus sectors: " + ", ".join(inv.fund.industry_focus))
        if inv.fund.preferred_verticals:
            parts.append("Verticals: " + ", ".join(inv.fund.preferred_verticals))
        if inv.fund.stage_focus:
            parts.append("Stages: " + ", ".join(inv.fund.stage_focus))
        if inv.fund.check_size_min or inv.fund.check_size_max:
            lo = f"${inv.fund.check_size_min:,}" if inv.fund.check_size_min else "?"
            hi = f"${inv.fund.check_size_max:,}" if inv.fund.check_size_max else "?"
            parts.append(f"Check size: {lo}–{hi}")
        return " | ".join(parts) if parts else ""

    if ns == "partner_bio":
        # Include location and thesis so BGE-M3 can encode geography + domain expertise.
        parts = []
        name_title = " ".join(p for p in [inv.full_name, inv.title, "at", inv.firm] if p)
        if name_title.strip():
            parts.append(name_title)
        if inv.hq_location:
            parts.append(f"Based in {inv.hq_location}")
        if inv.investor_type:
            parts.append(f"Investor type: {inv.investor_type}")
        if inv.investment_thesis and inv.investment_thesis not in (inv.fund.thesis_text or ""):
            parts.append(inv.investment_thesis)
        return " | ".join(parts) if parts else inv.full_name or ""

    if ns == "portfolio":
        # Portfolio companies define what a fund has conviction in.
        # Keep it as a space-joined list so each company name is a token.
        return " ".join(inv.portfolio_companies) if inv.portfolio_companies else ""

    if ns == "activity":
        # Activity signals: recency / deployment velocity indicators.
        # IMPORTANT: only use signals from genuine activity sources.
        # 'pitchbook_profile' and 'pitchbook_preferences' are STATIC description
        # text — the same content already encoded in fund_thesis — so including
        # them here would create a near-duplicate vector (the old bug).
        # Real activity sources: scraped news, press releases, deal announcements.
        _ACTIVITY_SOURCES = {
            "news_rss", "google_news", "news_mention",
            "linkedin_post", "twitter_post",
            "press_release", "portfolio_announcement", "crunchbase_deal",
            "pitchbook_activity", "pitchbook_activity_count", "fundraising_status",
            "pitchbook_deal",  # future: scraped deal data, not static profile
        }
        activity_signals = [s for s in inv.signals if s.source_type in _ACTIVITY_SOURCES]
        if activity_signals:
            return " | ".join(s.text for s in activity_signals[:8])

        # Build PitchBook-metadata activity text
        meta = inv.metadata or {}
        parts: list[str] = []

        last_date    = meta.get("last_investment_date")
        last_company = meta.get("last_investment_company")
        last_type    = meta.get("last_investment_type")
        total_inv    = meta.get("total_investments")
        active_port  = meta.get("active_portfolio_count")
        inv_5y       = meta.get("investments_5y")
        inv_24m      = meta.get("investments_24m")
        fundraising  = meta.get("most_likely_fundraising")

        # Most recent deal — highest-value recency signal
        if last_date and last_company:
            line = f"Most recent investment: {last_company} on {last_date}"
            if last_type:
                line += f" ({last_type})"
            parts.append(line)
        elif last_date:
            parts.append(f"Last investment date: {last_date}")
        elif last_company:
            parts.append(f"Most recent portfolio company: {last_company}")

        # Deal velocity
        if total_inv:
            parts.append(f"Total investments: {total_inv}")
        if inv_5y:
            parts.append(f"Investments past 5 years: {inv_5y}")
        if inv_24m:
            parts.append(f"Investments past 24 months: {inv_24m}")
        if inv.fund.recent_investment_count_12m:
            parts.append(f"Deals past 12 months: {inv.fund.recent_investment_count_12m}")
        if active_port:
            parts.append(f"Active portfolio companies: {active_port}")

        # Investor profile context
        if inv.investor_type:
            parts.append(f"Investor type: {inv.investor_type}")
        if inv.hq_location:
            parts.append(f"Based in: {inv.hq_location}")
        if fundraising:
            parts.append(f"Fundraising status: {fundraising}")

        # Append most recent portfolio companies as additional token signal
        if inv.portfolio_companies:
            parts.append("Recent portfolio: " + ", ".join(inv.portfolio_companies[-3:]))

        return " | ".join(parts) if parts else ""

    return ""


# ── Batch embedding helper ────────────────────────────────────────────────────

def _embed_batch(
    embedder: SentenceTransformerEmbedder,
    texts: list[str],
    model_batch_size: int,
) -> list[list[float]]:
    """Embed a batch of texts. Falls back to individual encode on error."""
    model = embedder._load_model()
    if model is None:
        return [embedder.embed(t) for t in texts]
    try:
        vecs = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=min(model_batch_size, len(texts)),
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]
    except Exception as e:
        logger.warning(f"Batch encode failed ({e}), falling back to per-item encode")
        return [embedder.embed(t) for t in texts]


# ── Main ─────────────────────────────────────────────────────────────────────

def _analyze_embedding_requirements(
    cache: EmbeddingCache,
    investors: list[InvestorCandidate],
    namespaces: list[str],
    force: bool = False,
) -> tuple[dict[str, list[tuple[str, str, str]]], dict]:
    """Return embed work items and a JSON-serializable cache analysis summary."""
    ns_needs_embed: dict[str, list[tuple[str, str, str]]] = {ns: [] for ns in namespaces}
    reason_counts = {
        ns: {
            "force": 0,
            "missing_vector": 0,
            "v1_no_hash": 0,
            "text_changed": 0,
            "up_to_date": 0,
        }
        for ns in namespaces
    }
    fully_cached_count = 0

    for inv in investors:
        inv_fully_cached = True
        for ns in namespaces:
            text = _text_for_namespace(inv, ns)
            text_hash = EmbeddingCache.compute_hash(text)
            vec = cache.get(inv.investor_id, ns)
            stored_h = cache.get_text_hash(inv.investor_id, ns)

            if force:
                ns_needs_embed[ns].append((inv.investor_id, text, text_hash))
                reason_counts[ns]["force"] += 1
                inv_fully_cached = False
            elif vec is None:
                ns_needs_embed[ns].append((inv.investor_id, text, text_hash))
                reason_counts[ns]["missing_vector"] += 1
                inv_fully_cached = False
            elif stored_h == "" or stored_h is None:
                ns_needs_embed[ns].append((inv.investor_id, text, text_hash))
                reason_counts[ns]["v1_no_hash"] += 1
                inv_fully_cached = False
            elif stored_h != text_hash:
                ns_needs_embed[ns].append((inv.investor_id, text, text_hash))
                reason_counts[ns]["text_changed"] += 1
                inv_fully_cached = False
            else:
                reason_counts[ns]["up_to_date"] += 1

        if inv_fully_cached:
            fully_cached_count += 1

    namespace_totals = {ns: len(items) for ns, items in ns_needs_embed.items()}
    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "repository_investor_count": len(investors),
        "cache_investor_count": cache.size,
        "force": force,
        "fully_cached_investor_count": fully_cached_count,
        "total_pairs_needing_embedding": sum(namespace_totals.values()),
        "namespace_pairs_needing_embedding": namespace_totals,
        "reason_counts_by_namespace": reason_counts,
    }
    return ns_needs_embed, summary


def _write_analysis_summary(summary: dict, path: Path | None = None) -> Path:
    if path is None:
        out_dir = Path("outputs") / "vector_audit"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"precompute_analysis_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    logger.info("Precompute analysis written → %s", path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Precompute investor embeddings")
    parser.add_argument("--model", default=BGE_M3, help="Embedding model ID")
    parser.add_argument("--batch-size", type=int, default=32, help="Internal encode mini-batch size")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Texts per encode() call")
    parser.add_argument("--force", action="store_true", help="Re-embed all investors")
    parser.add_argument("--pgvector", action="store_true", help="Push vectors to PostgreSQL")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Report cache/re-embed counts without loading the embedding model",
    )
    parser.add_argument("--analysis-output", type=Path, default=None, help="Path for --analyze-only JSON")
    args = parser.parse_args()

    expected_model_name = f"sentence-transformer-{args.model}"
    namespaces = ["fund_thesis", "partner_bio", "portfolio", "activity"]

    if args.analyze_only:
        cache_path = DATA_DIR / "investors" / "embeddings_cache.json"
        cache = EmbeddingCache(model_name=expected_model_name, cache_path=cache_path)
        repo = LocalInvestorRepository()
        all_investors = repo.load_all()
        _, analysis = _analyze_embedding_requirements(
            cache, all_investors, namespaces, force=args.force
        )
        _write_analysis_summary(analysis, args.analysis_output)
        print(json.dumps(analysis, indent=2))
        return

    # ── Load embedder ─────────────────────────────────────────────────────────
    logger.info(f"Loading embedding model: {args.model}")
    embedder = SentenceTransformerEmbedder(model_id=args.model, allow_download=True)
    # Force model load now so we see errors early
    model = embedder._load_model()
    if model is None or embedder.model_name != expected_model_name:
        logger.error(
            "Failed to load %s; refusing to precompute with fallback model %s",
            expected_model_name,
            embedder.model_name,
        )
        raise SystemExit(1)
    logger.info(f"Model ready: {embedder.model_name}")

    # ── Load cache ────────────────────────────────────────────────────────────
    cache_path = DATA_DIR / "investors" / "embeddings_cache.json"
    cache = EmbeddingCache(model_name=embedder.model_name, cache_path=cache_path)
    logger.info(f"Cache: {cache.size} investors already cached at {cache_path}")
    if args.force and cache.size:
        cache.invalidate()
        logger.info("Force mode enabled: cleared existing cache before recomputing")

    # ── Load all investors ────────────────────────────────────────────────────
    repo = LocalInvestorRepository()
    all_investors = repo.load_all()
    logger.info(f"Loaded {len(all_investors)} investors from disk")

    # ── Determine which (investor, namespace) pairs need (re-)embedding ──────
    # Content-hash invalidation: compare MD5 of the text to be embedded against
    # the stored hash. Only pairs whose text changed, have no vector, or come
    # from the old v1 cache format (h="") are added to the re-embed queue.
    ns_needs_embed, analysis = _analyze_embedding_requirements(
        cache, all_investors, namespaces, force=args.force
    )
    total_pairs = sum(len(v) for v in ns_needs_embed.values())
    logger.info(
        "Cache analysis: %d investors fully cached, %d v1 re-embeds, "
        "%d (investor, namespace) pairs need re-embedding",
        analysis["fully_cached_investor_count"],
        sum(ns_counts["v1_no_hash"] for ns_counts in analysis["reason_counts_by_namespace"].values()),
        total_pairs,
    )

    if total_pairs == 0:
        logger.info("Nothing to do — all investors are up to date. Use --force to re-embed.")
        if args.pgvector:
            _push_to_pgvector(cache, all_investors)
        return

    # ── Embed in batches per namespace ────────────────────────────────────────
    zero_vector = embedder.embed("")
    t0 = time.monotonic()
    total_vectors = 0

    for ns in namespaces:
        items = ns_needs_embed[ns]  # list of (inv_id, text, text_hash)
        if not items:
            logger.info(f"  Namespace {ns}: nothing to embed (all up to date)")
            continue

        logger.info(f"  Namespace: {ns} — {len(items)} investors")
        # Build a lookup so we can store the right hash after encoding
        hash_by_id = {inv_id: h for inv_id, _, h in items}
        id_text_pairs = [(inv_id, text) for inv_id, text, _ in items]

        # Process in batches
        for batch_start in range(0, len(id_text_pairs), args.chunk_size):
            batch_items = id_text_pairs[batch_start:batch_start + args.chunk_size]
            # Group similar-length texts inside each chunk to reduce padding waste
            # without front-loading the single worst outliers into the first chunk.
            batch_items.sort(key=lambda item: len(item[1]), reverse=True)
            empty_ids = []
            encode_ids = []
            encode_texts = []
            for inv_id, text in batch_items:
                if text and text.strip():
                    encode_ids.append(inv_id)
                    encode_texts.append(text)
                else:
                    empty_ids.append(inv_id)

            for inv_id in empty_ids:
                cache.set_with_hash(inv_id, ns, zero_vector, hash_by_id[inv_id])
                total_vectors += 1

            vectors = (
                _embed_batch(embedder, encode_texts, args.batch_size)
                if encode_texts else []
            )
            for inv_id, vec in zip(encode_ids, vectors):
                cache.set_with_hash(inv_id, ns, vec, hash_by_id[inv_id])
                total_vectors += 1

            done = min(batch_start + args.chunk_size, len(id_text_pairs))
            elapsed = time.monotonic() - t0
            rate = total_vectors / max(elapsed, 0.001)
            logger.info(
                f"    {done}/{len(id_text_pairs)} investors "
                f"({total_vectors} vectors total, {rate:.1f} vec/s)"
            )
            # Save after every chunk so a crash doesn't lose computed work.
            # On restart, investors with non-empty hashes are skipped, giving
            # crash-safe incremental progress over multi-hour runs.
            cache.save()
            logger.info(f"    [checkpoint] cache saved ({done} investors in {ns})")

    # ── Final save (no-op if last chunk already saved) ────────────────────────
    cache.save()
    elapsed = time.monotonic() - t0
    logger.info(
        f"Done. {total_vectors} vectors computed in {elapsed:.1f}s "
        f"({total_vectors/elapsed:.1f} vec/s). Cache: {cache.size} investors."
    )

    # ── Push to PostgreSQL (optional) ─────────────────────────────────────────
    if args.pgvector:
        _push_to_pgvector(cache, all_investors)


def _push_to_pgvector(cache: EmbeddingCache, investors: list[InvestorCandidate]) -> dict:
    """Push cached embeddings to PostgreSQL via PostgresVectorStore."""
    from matching.vector_store import PostgresVectorStore

    logger.info("Connecting to PostgreSQL...")
    try:
        store = PostgresVectorStore()
    except ValueError as e:
        logger.error(f"Cannot connect to Postgres: {e}")
        logger.error("Set DATABASE_URL env var and retry with --pgvector")
        return {"success": False, "error": str(e)}

    ns_map = {
        "fund_thesis": "investor_fund_thesis_chunks",
        "partner_bio": "investor_partner_bios",
        "portfolio": "investor_portfolio_company_chunks",
        "activity": "investor_public_content_chunks",
    }

    investor_by_id = {inv.investor_id: inv for inv in investors}
    summary = {
        "success": True,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "repository_investor_count": len(investor_by_id),
        "cache_investor_count": cache.size,
        "namespaces": {},
        "missing_investor_ids_count": 0,
        "missing_investor_ids_sample": [],
    }
    missing_investor_ids_seen: set[str] = set()

    for cache_ns, store_ns in ns_map.items():
        records = []
        missing_vector_count = 0
        for inv_id, ns_vecs in cache._embeddings.items():
            vec = cache.get(inv_id, cache_ns)
            if vec is None:
                missing_vector_count += 1
                continue
            inv = investor_by_id.get(inv_id)
            if inv is None:
                if inv_id not in missing_investor_ids_seen:
                    missing_investor_ids_seen.add(inv_id)
                    if len(summary["missing_investor_ids_sample"]) < 50:
                        summary["missing_investor_ids_sample"].append(inv_id)
                continue
            records.append(VectorRecord(
                namespace=store_ns,
                record_id=f"{inv_id}:{cache_ns}",
                vector=vec,
                metadata={
                    "entity_id": inv_id,
                    "entity_type": "investor_partner",
                    "fund_id": inv.fund.fund_id,
                    "industry_tags": inv.fund.industry_focus,
                    "stage_tags": inv.fund.stage_focus,
                    "geos": inv.fund.geography,
                    "confidence": inv.data_quality_confidence,
                    "visibility": "public",
                    "source_type": cache_ns,
                },
            ))

        summary["namespaces"][cache_ns] = {
            "store_namespace": store_ns,
            "records_upserted": len(records),
            "missing_vector_count": missing_vector_count,
        }
        if records:
            logger.info(f"Upserting {len(records)} records into namespace={store_ns!r}")
            store.upsert(store_ns, records)

    summary["missing_investor_ids_count"] = len(missing_investor_ids_seen)
    _write_pgvector_push_summary(summary)
    logger.info("pgvector push complete.")
    return summary


def _write_pgvector_push_summary(summary: dict) -> Path:
    """Write a lightweight pgvector push summary for auditability."""
    out_dir = Path("outputs") / "vector_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"pgvector_push_summary_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(summary, indent=2))
    logger.info("pgvector push summary written → %s", path)
    return path


if __name__ == "__main__":
    main()
