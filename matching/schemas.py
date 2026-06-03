from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class CompanyProfile:
    deal_id: str
    company_name: str
    industry: str
    company_website: str
    raise_amount: float
    stage: str
    geography: List[str] = field(default_factory=lambda: ["United States"])
    subindustry: Optional[str] = None
    business_model: Optional[str] = None
    traction_metrics: Dict[str, Any] = field(default_factory=dict)
    text_fields: Dict[str, str] = field(default_factory=dict)
    generated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deal_id": self.deal_id,
            "company_name": self.company_name,
            "industry": self.industry,
            "company_website": self.company_website,
            "raise_amount": self.raise_amount,
            "stage": self.stage,
            "geography": self.geography,
            "subindustry": self.subindustry,
            "business_model": self.business_model,
            "traction_metrics": self.traction_metrics,
            "text_fields": self.text_fields,
            "generated_at": self.generated_at,
        }


@dataclass
class InvestorSignal:
    source_type: str
    text: str
    source_url: str = ""
    date: str = ""
    confidence: float = 0.7
    visibility: str = "public"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "text": self.text,
            "source_url": self.source_url,
            "date": self.date,
            "confidence": self.confidence,
            "visibility": self.visibility,
        }


@dataclass
class FundProfile:
    fund_id: str
    name: str
    stage_focus: List[str]
    check_size_min: int
    check_size_max: int
    geography: List[str]
    industry_focus: List[str]
    sector_exclusions: List[str] = field(default_factory=list)
    lead_preference: str = "either"  # lead_only | follow_only | either
    status: str = "active"
    recent_investment_count_12m: int = 0
    thesis_text: str = ""
    aum_millions: Optional[float] = None
    dry_powder_millions: Optional[float] = None
    preferred_verticals: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fund_id": self.fund_id,
            "name": self.name,
            "stage_focus": self.stage_focus,
            "check_size_min": self.check_size_min,
            "check_size_max": self.check_size_max,
            "geography": self.geography,
            "industry_focus": self.industry_focus,
            "sector_exclusions": self.sector_exclusions,
            "lead_preference": self.lead_preference,
            "status": self.status,
            "recent_investment_count_12m": self.recent_investment_count_12m,
            "thesis_text": self.thesis_text,
            "aum_millions": self.aum_millions,
            "dry_powder_millions": self.dry_powder_millions,
            "preferred_verticals": self.preferred_verticals,
        }


@dataclass
class InvestorCandidate:
    investor_id: str
    full_name: str
    firm: str
    title: str
    email: str
    phone: str
    linkedin_url: str
    twitter_handle: str
    website: str
    fund: FundProfile
    investment_thesis: str
    portfolio_companies: List[str]
    check_size_range: str
    signals: List[InvestorSignal] = field(default_factory=list)
    source_tags: List[str] = field(default_factory=list)
    warm_intro_paths: List[str] = field(default_factory=list)
    data_quality_confidence: float = 0.8
    last_verified_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    investor_type: str = ""
    pitchbook_id: str = ""
    hq_location: str = ""
    hq_state: str = ""

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "investor_id": self.investor_id,
            "full_name": self.full_name,
            "firm": self.firm,
            "title": self.title,
            "email": self.email,
            "phone": self.phone,
            "linkedin_url": self.linkedin_url,
            "twitter_handle": self.twitter_handle,
            "website": self.website,
            "investment_thesis": self.investment_thesis,
            "portfolio_companies": self.portfolio_companies,
            "check_size_range": self.check_size_range,
            "source_tags": self.source_tags,
            "warm_intro_paths": self.warm_intro_paths,
            "data_quality_confidence": self.data_quality_confidence,
            "last_verified_at": self.last_verified_at,
            "fund": self.fund.to_dict(),
            "signals": [s.to_dict() for s in self.signals],
            "metadata": self.metadata,
            "investor_type": self.investor_type,
            "pitchbook_id": self.pitchbook_id,
            "hq_location": self.hq_location,
            "hq_state": self.hq_state,
        }


@dataclass
class MatchFeatures:
    semantic_fit: float
    stage_compatibility: float
    check_size_compatibility: float
    recent_activity: float
    portfolio_adjacency: float
    geography_fit: float
    warm_intro_score: float
    data_quality_confidence: float
    keyword_overlap: float = 0.0
    style_affinity: float = 0.0
    response_likelihood_prior: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()


@dataclass
class MatchExplanation:
    reasons: List[str]
    warnings: List[str] = field(default_factory=list)
    source_provenance: List[Dict[str, Any]] = field(default_factory=list)
    match_segment: str = "general"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reasons": self.reasons,
            "warnings": self.warnings,
            "source_provenance": self.source_provenance,
            "match_segment": self.match_segment,
        }


@dataclass
class MatchResult:
    investor: InvestorCandidate
    features: MatchFeatures
    final_score: float
    explanation: MatchExplanation
    score_weights: Dict[str, float]
    rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "final_score": self.final_score,
            "score_weights": self.score_weights,
            "features": self.features.to_dict(),
            "explanation": self.explanation.to_dict(),
            "investor": self.investor.to_public_dict(),
        }


@dataclass
class MatchRun:
    deal_id: str
    company_name: str
    query_inputs: Dict[str, Any]
    model_versions: Dict[str, str]
    score_weights: Dict[str, float]
    candidate_counts: Dict[str, int]
    results: List[MatchResult]
    approval_queue: Dict[str, Any]
    generated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deal_id": self.deal_id,
            "company_name": self.company_name,
            "query_inputs": self.query_inputs,
            "model_versions": self.model_versions,
            "score_weights": self.score_weights,
            "candidate_counts": self.candidate_counts,
            "results": [r.to_dict() for r in self.results],
            "approval_queue": self.approval_queue,
            "generated_at": self.generated_at,
        }
