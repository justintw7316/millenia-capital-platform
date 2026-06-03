"""
Hybrid investor-founder matching service.

Pipeline:
1. Hard eligibility filtering
2. Hybrid candidate generation (vector + keyword + warm path boosts)
3. Feature computation
4. Weighted reranking
5. Explainability and approval queue payloads
"""
from __future__ import annotations

import logging
import math
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple

from core.deal import Deal
from config import DATA_DIR
from matching.embedder import LocalHashEmbedder, SentenceTransformerEmbedder
from matching.embedding_cache import EmbeddingCache
from matching.profile_builder import build_company_profile_from_deal
from matching.repository import LocalInvestorRepository
from matching.schemas import (
    CompanyProfile,
    InvestorCandidate,
    MatchExplanation,
    MatchFeatures,
    MatchResult,
    MatchRun,
)
from matching.vector_store import InMemoryVectorStore, VectorRecord

logger = logging.getLogger(__name__)


DEFAULT_WEIGHTS = {
    "semantic_fit": 0.35,
    "stage_and_check": 0.20,
    "recent_activity": 0.20,
    "portfolio_adjacency": 0.10,
    "geography_fit": 0.05,
    "warm_intro_score": 0.05,
    "data_quality_confidence": 0.05,
}


class HybridMatchingService:
    def __init__(
        self,
        repository: LocalInvestorRepository | None = None,
        embedder: LocalHashEmbedder | SentenceTransformerEmbedder | None = None,
        vector_store: InMemoryVectorStore | None = None,
        embedding_cache: EmbeddingCache | None = None,
    ):
        self.repository = repository or LocalInvestorRepository()
        self.embedder = embedder or SentenceTransformerEmbedder()

        if vector_store is not None:
            self.vector_store = vector_store
        elif os.getenv("DATABASE_URL"):
            from matching.vector_store import PostgresVectorStore
            try:
                self.vector_store = PostgresVectorStore()
                logger.info("HybridMatchingService: using PostgresVectorStore (DATABASE_URL set)")
            except Exception as e:
                logger.warning(
                    "PostgresVectorStore init failed (%s) — falling back to InMemoryVectorStore", e
                )
                self.vector_store = InMemoryVectorStore()
        else:
            self.vector_store = InMemoryVectorStore()

        # Embedding cache: persist investor vectors across run_match() calls.
        # Auto-invalidated when the embedding model changes.
        # When using PostgresVectorStore, skip loading the 1.9GB cache from disk —
        # investors are already indexed in pgvector and _index_candidates() is skipped.
        if embedding_cache is not None:
            self._cache = embedding_cache
        else:
            from matching.vector_store import PostgresVectorStore
            cache_path = DATA_DIR / "investors" / "embeddings_cache.json"
            preload = not isinstance(self.vector_store, PostgresVectorStore)
            self._cache = EmbeddingCache(
                model_name=self.embedder.model_name,
                cache_path=cache_path,
                preload=preload,
            )

    def run_match(self, deal: Deal, top_k: int = 20, candidate_target: int = 300) -> MatchRun:
        profile = build_company_profile_from_deal(deal)
        all_candidates = self.repository.load_candidates(deal)
        eligible, rejected = self._hard_filter_candidates(profile, all_candidates)

        if self._should_index_candidates():
            self._index_candidates(eligible)
        vector_scores = self._vector_candidate_scores(profile, eligible, top_k=min(max(candidate_target, top_k), len(eligible) or top_k))
        keyword_scores = self._keyword_candidate_scores(profile, eligible)
        warm_boosts = {c.investor_id: min(1.0, 0.4 + 0.3 * len(c.warm_intro_paths)) for c in eligible if c.warm_intro_paths}
        industry_scores = {c.investor_id: _industry_alignment_score(profile, c) for c in eligible}

        stage2_ids = set()
        stage2_ids.update([cid for cid, _ in sorted(vector_scores.items(), key=lambda x: x[1], reverse=True)[:max(top_k * 3, 20)]])
        stage2_ids.update([cid for cid, _ in sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:max(top_k * 3, 20)]])
        stage2_ids.update([cid for cid, _ in sorted(industry_scores.items(), key=lambda x: x[1], reverse=True)[:max(top_k * 3, 20)]])
        stage2_ids.update([cid for cid, boost in warm_boosts.items() if boost > 0])

        shortlisted = [c for c in eligible if c.investor_id in stage2_ids] or eligible

        results: List[MatchResult] = []
        for cand in shortlisted:
            features = self._compute_features(profile, cand, vector_scores.get(cand.investor_id, 0.0), keyword_scores.get(cand.investor_id, 0.0))
            final_score = self._final_score(features)
            explanation = self._explain(profile, cand, features, rejected_reason=None)
            results.append(MatchResult(
                investor=cand,
                features=features,
                final_score=final_score,
                explanation=explanation,
                score_weights=DEFAULT_WEIGHTS.copy(),
            ))

        results.sort(key=lambda r: r.final_score, reverse=True)

        # Deduplicate by canonical investor ID (strip __dupN suffix) so the same
        # person/firm doesn't occupy two slots in the top-20 output.
        seen_canonical: set = set()
        deduped: list = []
        for r in results:
            canonical = re.sub(r"__dup\d+$", "", r.investor.investor_id)
            if canonical not in seen_canonical:
                seen_canonical.add(canonical)
                deduped.append(r)

        for i, r in enumerate(deduped[:top_k], 1):
            r.rank = i

        top_results = deduped[:top_k]
        from matching.vector_store import PostgresVectorStore
        vector_store_version = (
            "postgres-pgvector-v1"
            if isinstance(self.vector_store, PostgresVectorStore)
            else "in-memory-vector-store-v1"
        )
        return MatchRun(
            deal_id=deal.deal_id,
            company_name=deal.company_name,
            query_inputs={
                "industry": profile.industry,
                "raise_amount": profile.raise_amount,
                "stage": profile.stage,
                "geography": profile.geography,
                "top_k": top_k,
                "candidate_target": candidate_target,
            },
            model_versions={
                "embedding_model": self.embedder.model_name,
                "reranker": "weighted-rules-v1",
                "vector_store": vector_store_version,
                "keyword_search": "token-overlap-v1",
            },
            score_weights=DEFAULT_WEIGHTS.copy(),
            candidate_counts={
                "raw_candidates": len(all_candidates),
                "eligible_after_filters": len(eligible),
                "rejected_by_filters": len(rejected),
                "stage2_shortlist": len(shortlisted),
                "final_ranked": len(top_results),
            },
            results=top_results,
            approval_queue={
                "status": "pending_phil_team_review",
                "human_in_loop_controls": [
                    "editable_exclusions",
                    "override_rank",
                    "pin_investor",
                    "mark_do_not_contact",
                    "save_approval_rationale",
                ],
                "recommended_actions": [
                    "Review match reasons and recent signals for top 20",
                    "Approve, reject, or pin investors before Step 07b outreach drafting",
                    "Tag warm-path contacts for Phil/GT follow-up",
                ],
            },
            generated_at=datetime.utcnow().isoformat(),
        )

    def results_to_outreach_records(self, match_run: MatchRun) -> List[dict]:
        out = []
        for r in match_run.results:
            inv = r.investor
            reasons = r.explanation.reasons[:4]
            recent_signals = [s.to_dict() for s in inv.signals[:3]]
            feature_dict = r.features.to_dict()
            fit_score_0_100 = int(round(r.final_score * 100))
            out.append({
                "investor_id": inv.investor_id,
                "rank": r.rank,
                "full_name": inv.full_name,
                "firm": inv.firm,
                "title": inv.title,
                "email": inv.email,
                "phone": inv.phone,
                "linkedin_url": inv.linkedin_url,
                "twitter_handle": inv.twitter_handle,
                "website": inv.website,
                "fit_score": fit_score_0_100,
                "investment_thesis": inv.investment_thesis,
                "why_good_fit": "; ".join(reasons) if reasons else "Hybrid matching score indicates strong fit.",
                "portfolio_companies": inv.portfolio_companies,
                "check_size_range": inv.check_size_range,
                "match_features": feature_dict,
                "match_explanation": r.explanation.to_dict(),
                "source_provenance": r.explanation.source_provenance,
                "recent_signals": recent_signals,
                "warm_intro_paths": inv.warm_intro_paths,
                "source_tags": inv.source_tags,
                "score_breakdown": {
                    "final_score": round(r.final_score, 4),
                    "weights": r.score_weights,
                    "features": feature_dict,
                },
                "match_segment": r.explanation.match_segment,
                "data_quality_confidence": inv.data_quality_confidence,
                "last_verified_at": inv.last_verified_at,
            })
        return out

    def _hard_filter_candidates(self, profile: CompanyProfile, candidates: List[InvestorCandidate]) -> Tuple[List[InvestorCandidate], List[Tuple[str, str]]]:
        eligible = []
        rejected = []
        for c in candidates:
            # Fund status
            if c.fund.status != "active":
                rejected.append((c.investor_id, "inactive_fund"))
                continue

            # Sector exclusions
            if any(ex.lower() in profile.industry.lower() for ex in c.fund.sector_exclusions):
                rejected.append((c.investor_id, "sector_exclusion"))
                continue

            # Stage mismatch (soft-block if no stage metadata)
            if c.fund.stage_focus and profile.stage not in c.fund.stage_focus:
                if not (profile.stage == "series_a" and "seed" in c.fund.stage_focus):
                    rejected.append((c.investor_id, "stage_mismatch"))
                    continue

            # Check size hard filter: reject only investors whose maximum check is
            # genuinely too small to be a meaningful participant (< $25K or < 0.25%
            # of the raise, whichever is larger).  We do NOT penalise large VCs here
            # — Khosla writing a $3M lead cheque into a $5M seed is perfectly valid.
            if c.fund.check_size_max > 0 and c.fund.check_size_max < max(25_000, profile.raise_amount * 0.0025):
                rejected.append((c.investor_id, "check_size_too_small"))
                continue

            # Geography restriction
            if c.fund.geography and not _geography_matches(profile.geography, c.fund.geography):
                rejected.append((c.investor_id, "geography_mismatch"))
                continue

            # Dormancy filter: reject funds with no investment activity in > 4 years
            _last_inv = c.metadata.get("last_investment_date") if c.metadata else None
            if _last_inv:
                try:
                    _last_dt = datetime.fromisoformat(_last_inv[:10])
                    if (datetime.utcnow() - _last_dt).days > 1460:
                        rejected.append((c.investor_id, "dormant_fund"))
                        continue
                except (ValueError, TypeError):
                    pass

            # Crypto mismatch: reject crypto-native investors for non-crypto deals.
            # BGE-M3 embeds crypto/web3 language close to fintech, causing false
            # positives (e.g. NFT companies appearing as fintech investors).
            _deal_terms = _industry_terms(profile.industry)
            if "crypto" not in _deal_terms:
                _inv_thesis_terms = _industry_terms(
                    " ".join([c.investment_thesis or "", c.fund.thesis_text or ""])
                )
                if "crypto" in _inv_thesis_terms and "fintech" not in _inv_thesis_terms:
                    rejected.append((c.investor_id, "crypto_sector_mismatch"))
                    continue

            eligible.append(c)
        return eligible, rejected

    def _index_candidates(self, candidates: List[InvestorCandidate]) -> None:
        """Embed candidates and load into the vector store.

        Vectors are read from the disk cache when available — only investors
        not yet cached (or with stale cache after a model change) are re-embedded.
        The cache is flushed to disk after any new embeddings are computed.
        """
        namespaces: Dict[str, List[VectorRecord]] = {
            "investor_fund_thesis_chunks": [],
            "investor_partner_bios": [],
            "investor_public_content_chunks": [],
            "investor_portfolio_company_chunks": [],
        }

        # namespace key → (cache key, text extractor)
        _NS_CFG = [
            ("investor_fund_thesis_chunks", "fund_thesis",
             lambda c: c.fund.thesis_text or c.investment_thesis or ""),
            ("investor_partner_bios", "partner_bio",
             lambda c: f"{c.full_name} {c.title} at {c.firm}. {c.investment_thesis}"),
            ("investor_portfolio_company_chunks", "portfolio",
             lambda c: " ".join(c.portfolio_companies)),
            ("investor_public_content_chunks", "activity",
             lambda c: " ".join(s.text for s in c.signals[:5]) or c.investment_thesis or ""),
        ]

        cache_dirty = False
        for c in candidates:
            base_meta = {
                "entity_id": c.investor_id,
                "entity_type": "investor_partner",
                "fund_id": c.fund.fund_id,
                "partner_id": c.investor_id,
                "industry_tags": c.fund.industry_focus,
                "stage_tags": c.fund.stage_focus,
                "geos": c.fund.geography,
                "confidence": c.data_quality_confidence,
                "visibility": "public",
            }
            for store_ns, cache_ns, text_fn in _NS_CFG:
                vec = self._cache.get(c.investor_id, cache_ns)
                if vec is None:
                    vec = self.embedder.embed(text_fn(c))
                    self._cache.set(c.investor_id, cache_ns, vec)
                    cache_dirty = True
                namespaces[store_ns].append(VectorRecord(
                    namespace=store_ns,
                    record_id=f"{c.investor_id}:{cache_ns}",
                    vector=vec,
                    metadata={**base_meta, "source_type": cache_ns},
                ))

        if cache_dirty:
            self._cache.save()

        for ns, recs in namespaces.items():
            self.vector_store.upsert(ns, recs)

    def _should_index_candidates(self) -> bool:
        """Only build an index for ephemeral stores.

        pgvector is populated by scripts/precompute_embeddings.py. Re-upserting
        the full eligible corpus inside every match/eval run makes production
        evaluation pathologically slow and can hold long INSERT transactions.
        """
        from matching.vector_store import PostgresVectorStore
        return not isinstance(self.vector_store, PostgresVectorStore)

    def _vector_candidate_scores(self, profile: CompanyProfile, candidates: List[InvestorCandidate], top_k: int) -> Dict[str, float]:
        company_vectors = {
            "fund": self.embedder.embed(profile.text_fields.get("industry_market", "")),
            "partner": self.embedder.embed(profile.text_fields.get("company_core", "")),
            "portfolio": self.embedder.embed(profile.text_fields.get("problem", "")),
            "activity": self.embedder.embed(profile.text_fields.get("raise_thesis", "")),
        }
        ns_map = {
            "fund": "investor_fund_thesis_chunks",
            "partner": "investor_partner_bios",
            "portfolio": "investor_portfolio_company_chunks",
            "activity": "investor_public_content_chunks",
        }
        partials: Dict[str, Dict[str, float]] = {}
        for key, ns in ns_map.items():
            for hit in self.vector_store.query(ns, company_vectors[key], top_k=top_k):
                inv_id = hit["record_id"].split(":")[0]
                partials.setdefault(inv_id, {})[key] = max(partials.get(inv_id, {}).get(key, 0.0), hit["score"])

        scores = {}
        for c in candidates:
            p = partials.get(c.investor_id, {})
            # late fusion
            scores[c.investor_id] = (
                0.35 * p.get("fund", 0.0) +
                0.30 * p.get("partner", 0.0) +
                0.15 * p.get("portfolio", 0.0) +
                0.20 * p.get("activity", 0.0)
            )
        return scores

    def _keyword_candidate_scores(self, profile: CompanyProfile, candidates: List[InvestorCandidate]) -> Dict[str, float]:
        query_tokens = _tokens(
            " ".join([
                profile.industry,
                profile.text_fields.get("problem", ""),
                profile.text_fields.get("industry_market", ""),
                profile.text_fields.get("raise_thesis", ""),
            ])
        )
        scores = {}
        for c in candidates:
            haystack = " ".join([
                c.investment_thesis,
                " ".join(c.fund.industry_focus),
                " ".join(c.portfolio_companies),
                " ".join(s.text for s in c.signals),
            ])
            cand_tokens = _tokens(haystack)
            if not cand_tokens:
                scores[c.investor_id] = 0.0
                continue
            overlap = len(query_tokens & cand_tokens)
            scores[c.investor_id] = min(1.0, overlap / max(6, len(query_tokens)))
        return scores

    def _compute_features(self, profile: CompanyProfile, cand: InvestorCandidate, vector_score: float, keyword_score: float) -> MatchFeatures:
        industry_alignment = _industry_alignment_score(profile, cand)
        vector_semantic = 0.8 * vector_score + 0.2 * keyword_score
        tag_semantic = 0.75 * industry_alignment + 0.25 * keyword_score
        # Use industry_alignment as a gate on semantic_fit so that a high vector score
        # from surface-level language overlap (e.g. healthcare investor mentioning "AI")
        # cannot override a genuine sector mismatch.  Floor at 0.15 to avoid completely
        # zeroing out broad/unknown investors with no structured focus data.
        industry_gate = max(0.15, industry_alignment)
        semantic_fit = min(1.0, industry_gate * max(vector_semantic, tag_semantic))
        stage_compat = 1.0 if profile.stage in cand.fund.stage_focus else (0.7 if "seed" in cand.fund.stage_focus else 0.3)
        # Check size compatibility — practical lead/follow model.
        # Prior formula assumed checks are always 0.5-5% of raise, which
        # penalises lead VCs who write $1-5M cheques into $5M rounds.
        # New model: compatible if investor CAN write a meaningful cheque
        # (>= $25K or 0.25% of raise) without needing to own the whole round
        # (min cheque <= total raise amount).
        if cand.fund.check_size_max <= 0:
            check_compat = 0.5   # no data → neutral
        else:
            min_meaningful = max(25_000, profile.raise_amount * 0.0025)
            if cand.fund.check_size_max < min_meaningful:
                check_compat = 0.0   # definitively too small
            elif cand.fund.check_size_min > profile.raise_amount:
                check_compat = 0.2   # minimum cheque exceeds the whole raise
            else:
                # Investor can participate; score slightly higher if their typical
                # cheque fits well (not much larger than the whole raise).
                typical = (cand.fund.check_size_min + cand.fund.check_size_max) / 2
                if typical <= profile.raise_amount:
                    check_compat = 1.0
                else:
                    check_compat = 0.8   # can lead generously but slightly oversized
        # Recency: prefer metadata.last_investment_date (set during ingestion), then
        # fall back to the most recent pitchbook_activity signal date.  This lifts the
        # 79 % of investors whose metadata date is missing but who have signal records.
        _meta_date = cand.metadata.get("last_investment_date") if cand.metadata else None
        if not _meta_date and cand.signals:
            _activity_dates = [
                s.date for s in cand.signals
                if s.source_type == "pitchbook_activity" and s.date
            ]
            if _activity_dates:
                # Use the most recent date among activity signals
                parsed = [_parse_investment_date(d) for d in _activity_dates]
                valid  = [d for d in parsed if d is not None]
                if valid:
                    _meta_date = max(valid).strftime("%Y-%m-%d")
        _rd = _recency_score(_meta_date)
        _count_term = min(0.4, 0.1 * cand.fund.recent_investment_count_12m)
        recent_activity = min(1.0, 0.6 * _rd + _count_term)
        industry_token = profile.industry.lower()
        portfolio_hits = sum(1 for p in cand.portfolio_companies if any(tok in p.lower() for tok in industry_token.split()))
        portfolio_adjacency = min(1.0, 0.3 + 0.25 * portfolio_hits) if cand.portfolio_companies else 0.2
        geography_fit = 1.0 if _geography_matches(profile.geography, cand.fund.geography) else 0.5
        warm_intro = min(1.0, 0.4 + 0.3 * len(cand.warm_intro_paths)) if cand.warm_intro_paths else 0.0
        return MatchFeatures(
            semantic_fit=semantic_fit,
            stage_compatibility=stage_compat,
            check_size_compatibility=check_compat,
            recent_activity=recent_activity,
            portfolio_adjacency=portfolio_adjacency,
            geography_fit=geography_fit,
            warm_intro_score=warm_intro,
            data_quality_confidence=max(0.0, min(1.0, cand.data_quality_confidence)),
            keyword_overlap=keyword_score,
        )

    def _final_score(self, f: MatchFeatures) -> float:
        stage_and_check = (f.stage_compatibility + f.check_size_compatibility) / 2
        score = (
            DEFAULT_WEIGHTS["semantic_fit"] * f.semantic_fit
            + DEFAULT_WEIGHTS["stage_and_check"] * stage_and_check
            + DEFAULT_WEIGHTS["recent_activity"] * f.recent_activity
            + DEFAULT_WEIGHTS["portfolio_adjacency"] * f.portfolio_adjacency
            + DEFAULT_WEIGHTS["geography_fit"] * f.geography_fit
            + DEFAULT_WEIGHTS["warm_intro_score"] * f.warm_intro_score
            + DEFAULT_WEIGHTS["data_quality_confidence"] * f.data_quality_confidence
        )
        return max(0.0, min(1.0, score))

    def _explain(self, profile: CompanyProfile, cand: InvestorCandidate, features: MatchFeatures, rejected_reason: str | None) -> MatchExplanation:
        reasons: list[str] = []
        warnings: list[str] = []
        industry_words = set(re.sub(r"[^a-z\s]", "", profile.industry.lower()).split())

        # ── 1. Thesis: find the investor-specific sentence that matches the deal ──
        if features.semantic_fit >= 0.5 and cand.investment_thesis:
            sentences = [s.strip() for s in re.split(r"[.!?]", cand.investment_thesis) if len(s.strip()) > 20]
            relevant = next(
                (s for s in sentences if any(w in s.lower() for w in industry_words)),
                None,
            )
            if relevant:
                reasons.append(f'Thesis: "{relevant[:130].strip()}"')
            elif cand.fund.industry_focus:
                reasons.append(f"Fund sector focus: {', '.join(cand.fund.industry_focus[:3])}")
            else:
                reasons.append(f"Semantic signal indicates strong {profile.industry} alignment")
        elif features.semantic_fit >= 0.25 and cand.fund.industry_focus:
            reasons.append(f"Sector overlap: {', '.join(cand.fund.industry_focus[:3])}")

        # ── 2. Stage: name exactly which stages they back ─────────────────────
        if features.stage_compatibility >= 0.9 and cand.fund.stage_focus:
            stage_list = ", ".join(s.replace("_", " ") for s in cand.fund.stage_focus[:3])
            reasons.append(f"Invests at {profile.stage.replace('_', ' ')} stage ({stage_list})")

        # ── 3. Check size: show the actual numbers ────────────────────────────
        if features.check_size_compatibility >= 0.9 and cand.fund.check_size_max > 0:
            def _fmt(n: float) -> str:
                return f"${n/1e6:.1f}M" if n >= 1e6 else f"${n/1e3:.0f}K"
            lo_str = _fmt(cand.fund.check_size_min) if cand.fund.check_size_min > 0 else "open"
            hi_str = _fmt(cand.fund.check_size_max)
            reasons.append(f"Cheque range {lo_str}–{hi_str} fits a ${_fmt(profile.raise_amount)} raise")

        # ── 4. Portfolio: name adjacent bets ──────────────────────────────────
        if features.portfolio_adjacency >= 0.55 and cand.portfolio_companies:
            hits = [p for p in cand.portfolio_companies if any(w in p.lower() for w in industry_words)]
            if hits:
                reasons.append(f"Portfolio includes adjacent bets: {', '.join(hits[:3])}")
            else:
                reasons.append(f"Portfolio of {len(cand.portfolio_companies)} companies shows sector exposure")

        # ── 5. Recent activity — show last investment if known ────────────────
        if features.recent_activity >= 0.7:
            if cand.fund.recent_investment_count_12m > 0:
                reasons.append(
                    f"Actively deploying: {cand.fund.recent_investment_count_12m} deals in last 12 months"
                )
            else:
                # Try to surface the last investment from signals
                activity = next(
                    (s for s in cand.signals if s.source_type == "pitchbook_activity" and s.date),
                    None,
                )
                if activity:
                    # Extract company name from signal text if present
                    company = ""
                    for part in activity.text.split("|"):
                        part = part.strip()
                        if part.startswith("company:"):
                            company = part.split(":", 1)[1].strip()
                            break
                    if company:
                        reasons.append(f"Recent investment: {company} ({activity.date[:7]})")
                    else:
                        reasons.append(f"Recent investment activity recorded ({activity.date[:7]})")

        # ── 6. Geography: call out the matched region ─────────────────────────
        if features.geography_fit >= 1.0 and cand.fund.geography and profile.geography:
            geo_match = next(
                (g for g in cand.fund.geography
                 if any(pg.lower() in g.lower() or g.lower() in pg.lower()
                        for pg in profile.geography)),
                cand.fund.geography[0],
            )
            reasons.append(f"Geographic focus includes {geo_match}")

        # ── 7. Warm intro ─────────────────────────────────────────────────────
        if cand.warm_intro_paths:
            reasons.append(cand.warm_intro_paths[0])

        if not reasons:
            reasons.append("Passed eligibility filters; borderline fit — review manually")

        # ── Warnings ──────────────────────────────────────────────────────────
        if features.data_quality_confidence < 0.75:
            warnings.append("Data confidence moderate — verify before outreach")
        if not cand.email:
            warnings.append("No email on file — research required")
        if not cand.linkedin_url:
            warnings.append("No LinkedIn URL — verify identity before contacting")

        segment = _segment_for_candidate(cand, profile)
        provenance = []
        for sig in cand.signals[:3]:
            provenance.append({
                "source_type": sig.source_type,
                "source_url": sig.source_url,
                "date": sig.date,
                "confidence": sig.confidence,
            })
        return MatchExplanation(
            reasons=reasons[:5] or ["Hybrid matching score indicates acceptable fit after eligibility filtering."],
            warnings=warnings,
            source_provenance=provenance,
            match_segment=segment,
        )


def _parse_investment_date(date_str: str | None) -> "datetime | None":
    """Parse an investment date string into a datetime, handling multiple formats.

    PitchBook exports use inconsistent formats:
      - ISO:         "2024-09-25"   (most common)
      - DD-Mon-YYYY: "03-Jun-2024"  (minority, but present)
    """
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(date_str[:11].strip(), fmt)
        except (ValueError, TypeError):
            continue
    # Last attempt: fromisoformat on the first 10 characters
    try:
        return datetime.fromisoformat(date_str[:10])
    except (ValueError, TypeError):
        return None


def _recency_score(last_date_str: str | None) -> float:
    """Map a last-investment-date string to a [0.05, 1.0] recency score.

    Decay tiers:
        < 90 days  → 1.0   (very active)
        < 365 days → 0.7   (active in past year)
        < 730 days → 0.4   (deployed in past 2 years)
        < 1095 days→ 0.2   (sporadic — 2–3 years ago)
        ≥ 1095 days→ 0.05  (stale — 3+ years)

    Returns 0.5 (neutral) when no date is available so missing data
    does not unfairly penalise or reward an investor.
    """
    last_date = _parse_investment_date(last_date_str)
    if last_date is None:
        return 0.5
    days_ago = (datetime.utcnow() - last_date).days
    if days_ago < 90:
        return 1.0
    if days_ago < 365:
        return 0.7
    if days_ago < 730:
        return 0.4
    if days_ago < 1095:
        return 0.2
    return 0.05


def _tokens(text: str) -> set:
    toks = re.findall(r"[a-zA-Z0-9_+#.-]+", (text or "").lower())
    stop = {"the", "and", "for", "with", "this", "that", "from", "into", "are", "is", "in", "of", "to", "a", "an"}
    return {t for t in toks if len(t) > 2 and t not in stop}


def _geography_matches(profile_geography: List[str], investor_geography: List[str]) -> bool:
    if not investor_geography:
        return True
    deal_tokens = _geo_token_set(profile_geography)
    investor_tokens = _geo_token_set(investor_geography)
    if not deal_tokens or not investor_tokens:
        return True
    broad = {"global", "north america", "americas", "united states", "usa", "us"}
    if investor_tokens & broad:
        return True
    return bool(deal_tokens & investor_tokens)


def _geo_token_set(values: List[str]) -> set[str]:
    aliases = {
        "u.s.": "united states",
        "u.s": "united states",
        "usa": "united states",
        "us": "united states",
        "southeast us": "southeast",
        "southeast united states": "southeast",
        "southwest us": "southwest",
        "northeast us": "northeast",
        "midwest us": "midwest",
    }
    out: set[str] = set()
    for raw in values or []:
        text = (raw or "").lower()
        parts = re.split(r"[,;/|]|\s+and\s+", text)
        for part in parts:
            token = re.sub(r"\s+", " ", part.strip())
            if not token:
                continue
            out.add(aliases.get(token, token))
            if "united states" in token:
                out.add("united states")
            if "texas" in token:
                out.add("texas")
            if "southeast" in token:
                out.add("southeast")
            if "north america" in token:
                out.add("north america")
    return out


def _industry_alignment_score(profile: CompanyProfile, cand: InvestorCandidate) -> float:
    profile_terms = _industry_terms(profile.industry)
    profile_terms.update(_industry_terms(profile.text_fields.get("industry_market", "")))
    if not profile_terms:
        return 0.0

    # Expand profile terms with closely related categories so that e.g. an AI company
    # also matches "software"-focused investors, and a fintech company also matches
    # "financial services" investors.
    if "ai" in profile_terms:
        profile_terms.add("enterprise_software")
    if "fintech" in profile_terms:
        profile_terms.add("enterprise_software")
    if "healthcare" in profile_terms:
        profile_terms.add("hardware")  # med-devices companies match hardware investors

    # --- Structured focus check (highest-reliability signal) ---
    # Only use the investor's explicit industry_focus + preferred_verticals list.
    # If this structured data exists AND has zero overlap with the deal sector,
    # return 0.0 immediately — thesis text mentioning "AI" as a tool should NOT
    # rescue a pharma-only or healthcare-only investor for a general AI company.
    focus_terms = _industry_terms(
        " ".join(cand.fund.industry_focus + cand.fund.preferred_verticals)
    )
    if focus_terms and not (focus_terms & profile_terms):
        return 0.0

    # --- Full text scoring ---
    investor_text = " ".join([
        " ".join(cand.fund.industry_focus),
        " ".join(cand.fund.preferred_verticals),
        cand.fund.thesis_text,
        cand.investment_thesis,
        cand.investor_type,
        " ".join(cand.source_tags),
    ])
    investor_terms = _industry_terms(investor_text)
    if not investor_terms:
        # No industry data at all — return neutral 0.5 so broad/unknown investors
        # are not unfairly penalised by the industry_gate in _compute_features.
        return 0.5
    overlap = profile_terms & investor_terms
    if not overlap:
        return 0.0
    coverage = len(overlap) / len(profile_terms)
    focus_ratio = len(overlap) / len(investor_terms)
    score = 0.20 + 0.60 * coverage + 0.20 * focus_ratio

    # Broad multi-sector firms should remain eligible, but they should not get
    # the same semantic score as a focused thesis match just because they list
    # nearly every sector.
    if len(investor_terms) > max(4, len(profile_terms) + 3):
        score = min(score, 0.65)

    return min(1.0, score)


def _industry_terms(text: str) -> set[str]:
    """Map free text (including PitchBook taxonomy strings) to canonical sector labels."""
    t = (text or "").lower()
    terms: set[str] = set()
    patterns = {
        # AI / ML
        "ai": [
            "artificial intelligence", "machine learning", " ai ", " ai-", "ai infrastructure",
            "genai", "deep learning", "large language", "llm", "nlp", "computer vision",
            "generative ai", "foundation model",
        ],
        # Fintech / financial services
        "fintech": [
            "fintech", "financial technology", "payments", "banking", "lending", "credit",
            "insurance", "insurtech", "wealth management", "capital markets",
            "financial services", "other financial services", "financial software",
            "payment processing", "neobank", "regtech",
        ],
        # Healthcare — includes full PitchBook taxonomy
        "healthcare": [
            "healthcare", "health tech", "healthtech", "clinical", "medical",
            "pharmaceutical", "pharmaceuticals", "biotechnology", "biotech", "biopharma",
            "life sciences", "health services", "surgical devices", "surgical",
            "diagnostic", "diagnostics", "patient", "hospital", "pharma", "therapeutics",
            "drug discovery", "medical devices", "medical supplies", "other healthcare",
            "digital health", "m-health", "consumer health",
        ],
        # Real estate / proptech
        "real_estate": [
            "real estate", "proptech", "property", "commercial real estate", "residential",
        ],
        # Manufacturing / industrial
        "manufacturing": [
            "manufacturing", "industrial", "factory", "supply chain", "logistics",
        ],
        # Energy / climate / deep-tech
        "energy": [
            "energy", "clean energy", "climate", "renewable", "battery", "nuclear",
            "solar", "wind", "decarbonization", "cleantech", "power generation",
            "chemicals and gases", "chemicals", "gases", "metals",
            "geothermal", "carbon capture",
        ],
        # Consumer
        "consumer": [
            "consumer", "restaurant", "retail", "cpg", "hospitality",
            "food and beverage", "direct to consumer", "e-commerce",
            "consumer products and services",
        ],
        # Robotics / automation / deep-tech
        "robotics": [
            "robotics", "drone", "drones", "automation", "agriculture", "agtech",
            "autonomous systems", "autonomous vehicles",
        ],
        # Enterprise software / B2B SaaS
        "enterprise_software": [
            "enterprise software", "b2b", "saas", "workflow", "software",
            "developer tools", "information technology", "it services",
            "other information technology", "devtools", "productivity software",
        ],
        # Defense / aerospace / dual-use
        "defense": [
            "aerospace", "defense", "defence", "dual-use", "military",
            "national security", "government tech", "govtech",
            "aerospace and defense", "communications and networking",
        ],
        # Crypto / web3 — distinct from fintech
        "crypto": [
            "crypto", "blockchain", "nft", "web3", "defi", "digital assets",
            "dao", "token", "decentralized finance", "cryptocurrency",
        ],
        # Hardware / semiconductor / instruments
        "hardware": [
            "hardware", "semiconductor", "chip", "electronic equipment",
            "electronic equipment and instruments", "instruments", "sensors",
            "devices", "medical devices", "diagnostic equipment",
            "other devices and supplies",
        ],
    }
    padded = f" {t} "
    for term, needles in patterns.items():
        if any(needle in padded for needle in needles):
            terms.add(term)
    return terms


def _segment_for_candidate(cand: InvestorCandidate, profile: CompanyProfile) -> str:
    thesis = (cand.investment_thesis or "").lower()
    if "infrastructure" in thesis or "developer" in thesis:
        return "ai_infra" if "ai" in profile.industry.lower() else "deep_tech"
    if "application" in thesis or "saas" in thesis:
        return "ai_apps" if "ai" in profile.industry.lower() else "software"
    if cand.warm_intro_paths:
        return "warm_network"
    return "general_early_stage"
