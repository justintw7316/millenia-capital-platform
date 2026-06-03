"""
core/deal.py — Deal dataclass — the central state object passed through every step.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
import json
from datetime import datetime


class DealStage(Enum):
    ONBOARDED = "onboarded"
    DATA_ROOM_READY = "data_room_ready"
    ALT_FUNDING_IDENTIFIED = "alt_funding_identified"
    VIDEO_READY = "video_ready"
    OUTREACH_ACTIVE = "outreach_active"
    CAMPAIGNS_LIVE = "campaigns_live"
    INVESTOR_MEETINGS_SCHEDULED = "investor_meetings_scheduled"
    COMMITMENTS_RECEIVED = "commitments_received"
    CLOSED = "closed"


class DocumentStatus(Enum):
    MISSING = "missing"
    DRAFT = "draft"
    APPROVED = "approved"
    DISTRIBUTED = "distributed"


class AccreditedStatus(Enum):
    UNVERIFIED = "unverified"
    ACCREDITED = "accredited"
    DISQUALIFIED = "disqualified"


class WireStatus(Enum):
    PENDING = "pending"
    CLIENT_CONFIRMED = "client_confirmed"
    INVESTOR_CONFIRMED = "investor_confirmed"
    DOCK_WALLS_APPROVED = "dock_walls_approved"
    COMPLETED = "completed"


@dataclass
class WireVerification:
    investor_name: str
    firm: str
    wire_amount: float
    status: str = WireStatus.PENDING.value
    client_confirmed_at: Optional[str] = None
    investor_confirmed_at: Optional[str] = None
    dock_walls_approved_at: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "investor_name": self.investor_name,
            "firm": self.firm,
            "wire_amount": self.wire_amount,
            "status": self.status,
            "client_confirmed_at": self.client_confirmed_at,
            "investor_confirmed_at": self.investor_confirmed_at,
            "dock_walls_approved_at": self.dock_walls_approved_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WireVerification":
        return cls(
            investor_name=d.get("investor_name", ""),
            firm=d.get("firm", ""),
            wire_amount=float(d.get("wire_amount", 0)),
            status=d.get("status", WireStatus.PENDING.value),
            client_confirmed_at=d.get("client_confirmed_at"),
            investor_confirmed_at=d.get("investor_confirmed_at"),
            dock_walls_approved_at=d.get("dock_walls_approved_at"),
            notes=d.get("notes", ""),
        )

    @property
    def is_complete(self) -> bool:
        return self.status == WireStatus.COMPLETED.value

    @property
    def needs_dock_walls(self) -> bool:
        return self.status in (
            WireStatus.CLIENT_CONFIRMED.value,
            WireStatus.INVESTOR_CONFIRMED.value,
        )


@dataclass
class DealDocuments:
    tear_sheet: DocumentStatus = DocumentStatus.MISSING
    pitch_deck: DocumentStatus = DocumentStatus.MISSING
    financial_projections: DocumentStatus = DocumentStatus.MISSING
    nda: DocumentStatus = DocumentStatus.MISSING
    ppm: DocumentStatus = DocumentStatus.MISSING
    subscription_agreement: DocumentStatus = DocumentStatus.MISSING
    wiring_instructions: DocumentStatus = DocumentStatus.MISSING
    use_of_funds: DocumentStatus = DocumentStatus.MISSING

    def to_dict(self) -> Dict[str, str]:
        return {
            "tear_sheet": self.tear_sheet.value,
            "pitch_deck": self.pitch_deck.value,
            "financial_projections": self.financial_projections.value,
            "nda": self.nda.value,
            "ppm": self.ppm.value,
            "subscription_agreement": self.subscription_agreement.value,
            "wiring_instructions": self.wiring_instructions.value,
            "use_of_funds": self.use_of_funds.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "DealDocuments":
        return cls(
            tear_sheet=DocumentStatus(data.get("tear_sheet", "missing")),
            pitch_deck=DocumentStatus(data.get("pitch_deck", "missing")),
            financial_projections=DocumentStatus(data.get("financial_projections", "missing")),
            nda=DocumentStatus(data.get("nda", "missing")),
            ppm=DocumentStatus(data.get("ppm", "missing")),
            subscription_agreement=DocumentStatus(data.get("subscription_agreement", "missing")),
            wiring_instructions=DocumentStatus(data.get("wiring_instructions", "missing")),
            use_of_funds=DocumentStatus(data.get("use_of_funds", "missing")),
        )

    def missing_documents(self) -> List[str]:
        """Returns list of document names that are MISSING."""
        missing = []
        for field_name in [
            "tear_sheet", "pitch_deck", "financial_projections", "nda",
            "ppm", "subscription_agreement", "wiring_instructions", "use_of_funds"
        ]:
            if getattr(self, field_name) == DocumentStatus.MISSING:
                missing.append(field_name)
        return missing


@dataclass
class Deal:
    # ── Required identity fields ───────────────────────────────────────────────
    deal_id: str
    company_name: str
    company_website: str
    industry: str
    raise_amount: float  # in USD

    # ── Founder info ──────────────────────────────────────────────────────────
    founder_name: str
    founder_email: str
    founder_linkedin: str

    # ── Pipeline state ────────────────────────────────────────────────────────
    stage: DealStage = DealStage.ONBOARDED
    documents: DealDocuments = field(default_factory=DealDocuments)

    # ── Rich optional context — improves every Claude prompt ──────────────────
    # Collect these at intake; all downstream AI steps benefit from them.
    stage_label: Optional[str] = None           # "pre-seed" | "seed" | "series-a" | "series-b" | "growth"
    geography: Optional[str] = None             # e.g. "Dallas, TX" | "United States" | "Global"
    problem: Optional[str] = None               # problem statement (2-4 sentences)
    solution: Optional[str] = None              # solution / product description
    market_size: Optional[str] = None           # e.g. "$50B TAM / $5B SAM / $500M SOM"
    valuation: Optional[float] = None           # pre-money valuation in USD
    team_bios: Optional[str] = None             # key team member bios
    traction_metrics: Optional[str] = None      # MRR, ARR, users, growth rate, etc.
    competitors: Optional[str] = None           # key competitors (comma-separated or descriptive)
    use_of_funds_breakdown: Optional[str] = None  # how raise will be deployed

    # ── Investor tracking ─────────────────────────────────────────────────────
    investors_contacted: List[Dict] = field(default_factory=list)
    investors_responded: List[Dict] = field(default_factory=list)
    investors_committed: List[Dict] = field(default_factory=list)
    disqualified_investors: List[Dict] = field(default_factory=list)
    nda_signed_by: List[str] = field(default_factory=list)

    # ── Wire verification ─────────────────────────────────────────────────────
    wire_verifications: List[Dict] = field(default_factory=list)

    # ── Campaign state ────────────────────────────────────────────────────────
    outreach_week: int = 0
    campaign_active: bool = False
    meetalfred_approved: bool = False
    meetalfred_campaign_id: Optional[str] = None

    # ── Metadata ──────────────────────────────────────────────────────────────
    box_folder_id: Optional[str] = None
    created_at: Optional[str] = None
    last_updated: Optional[str] = None
    step_log: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    skipped_steps: List[str] = field(default_factory=list)

    def log_step(self, step: str, status: str, message: str, output: Any = None) -> None:
        """Append a step execution record to the audit trail."""
        self.step_log.append({
            "step": step,
            "status": status,
            "message": message,
            "output": output,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self.last_updated = datetime.utcnow().isoformat()

    def add_error(self, error: str) -> None:
        self.errors.append(error)

    def add_wire_verification(self, investor_name: str, firm: str, wire_amount: float) -> WireVerification:
        """Create a new pending wire verification record for a committed investor."""
        wv = WireVerification(
            investor_name=investor_name,
            firm=firm,
            wire_amount=wire_amount,
        )
        self.wire_verifications.append(wv.to_dict())
        return wv

    def pending_wire_verifications(self) -> List[WireVerification]:
        """Return all wire verifications not yet completed."""
        return [
            WireVerification.from_dict(w) for w in self.wire_verifications
            if w.get("status") != WireStatus.COMPLETED.value
        ]

    def to_dict(self) -> Dict:
        """Serialize Deal to a plain dict (JSON-safe)."""
        return {
            "deal_id": self.deal_id,
            "company_name": self.company_name,
            "company_website": self.company_website,
            "industry": self.industry,
            "raise_amount": self.raise_amount,
            "founder_name": self.founder_name,
            "founder_email": self.founder_email,
            "founder_linkedin": self.founder_linkedin,
            "stage": self.stage.value,
            "documents": self.documents.to_dict(),
            # Rich context
            "stage_label": self.stage_label,
            "geography": self.geography,
            "problem": self.problem,
            "solution": self.solution,
            "market_size": self.market_size,
            "valuation": self.valuation,
            "team_bios": self.team_bios,
            "traction_metrics": self.traction_metrics,
            "competitors": self.competitors,
            "use_of_funds_breakdown": self.use_of_funds_breakdown,
            # Investor tracking
            "investors_contacted": self.investors_contacted,
            "investors_responded": self.investors_responded,
            "investors_committed": self.investors_committed,
            "disqualified_investors": self.disqualified_investors,
            "nda_signed_by": self.nda_signed_by,
            # Wire verification
            "wire_verifications": self.wire_verifications,
            # Campaign state
            "outreach_week": self.outreach_week,
            "campaign_active": self.campaign_active,
            "meetalfred_approved": self.meetalfred_approved,
            "meetalfred_campaign_id": self.meetalfred_campaign_id,
            # Metadata
            "box_folder_id": self.box_folder_id,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "step_log": self.step_log,
            "errors": self.errors,
            "skipped_steps": self.skipped_steps,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Deal":
        """Deserialize Deal from a plain dict. All optional fields default gracefully."""
        deal = cls(
            deal_id=data["deal_id"],
            company_name=data["company_name"],
            company_website=data["company_website"],
            industry=data["industry"],
            raise_amount=float(data["raise_amount"]),
            founder_name=data["founder_name"],
            founder_email=data["founder_email"],
            founder_linkedin=data["founder_linkedin"],
        )
        deal.stage = DealStage(data.get("stage", "onboarded"))
        deal.documents = DealDocuments.from_dict(data.get("documents", {}))
        # Rich optional context
        deal.stage_label = data.get("stage_label")
        deal.geography = data.get("geography")
        deal.problem = data.get("problem")
        deal.solution = data.get("solution")
        deal.market_size = data.get("market_size")
        deal.valuation = data.get("valuation")
        deal.team_bios = data.get("team_bios")
        deal.traction_metrics = data.get("traction_metrics")
        deal.competitors = data.get("competitors")
        deal.use_of_funds_breakdown = data.get("use_of_funds_breakdown")
        # Investor tracking
        deal.investors_contacted = data.get("investors_contacted", [])
        deal.investors_responded = data.get("investors_responded", [])
        deal.investors_committed = data.get("investors_committed", [])
        deal.disqualified_investors = data.get("disqualified_investors", [])
        deal.nda_signed_by = data.get("nda_signed_by", [])
        # Wire verification
        deal.wire_verifications = data.get("wire_verifications", [])
        # Campaign state
        deal.outreach_week = data.get("outreach_week", 0)
        deal.campaign_active = data.get("campaign_active", False)
        deal.meetalfred_approved = data.get("meetalfred_approved", False)
        deal.meetalfred_campaign_id = data.get("meetalfred_campaign_id")
        # Metadata
        deal.box_folder_id = data.get("box_folder_id")
        deal.created_at = data.get("created_at")
        deal.last_updated = data.get("last_updated")
        deal.step_log = data.get("step_log", [])
        deal.errors = data.get("errors", [])
        deal.skipped_steps = data.get("skipped_steps", [])
        return deal

    @classmethod
    def from_json_file(cls, file_path: str) -> "Deal":
        """Load a Deal from a JSON file, with field validation."""
        with open(file_path, "r") as f:
            data = json.load(f)

        required_fields = [
            "deal_id", "company_name", "company_website", "industry",
            "raise_amount", "founder_name", "founder_email", "founder_linkedin",
        ]
        missing = [field for field in required_fields if field not in data]
        if missing:
            raise ValueError(f"Invalid deal JSON — missing required fields: {missing}")

        return cls.from_dict(data)

    def company_profile_text(self) -> str:
        """Return a rich plain-text company summary for use in AI prompts.

        Uses all available context fields so Claude produces specific, investor-grade output
        rather than generic placeholders.
        """
        lines = [
            f"Company: {self.company_name}",
            f"Website: {self.company_website}",
            f"Industry: {self.industry}",
            f"Raise Amount: ${self.raise_amount:,.0f}",
            f"Founder: {self.founder_name} ({self.founder_email})",
            f"LinkedIn: {self.founder_linkedin}",
        ]
        if self.stage_label:
            lines.append(f"Stage: {self.stage_label}")
        if self.geography:
            lines.append(f"Geography: {self.geography}")
        if self.valuation:
            lines.append(f"Pre-Money Valuation: ${self.valuation:,.0f}")

        if self.problem:
            lines += ["", "Problem:", self.problem]
        if self.solution:
            lines += ["", "Solution:", self.solution]
        if self.market_size:
            lines += ["", "Market Size:", self.market_size]
        if self.traction_metrics:
            lines += ["", "Traction:", self.traction_metrics]
        if self.competitors:
            lines += ["", "Key Competitors:", self.competitors]
        if self.team_bios:
            lines += ["", "Team:", self.team_bios]
        if self.use_of_funds_breakdown:
            lines += ["", "Use of Funds:", self.use_of_funds_breakdown]

        return "\n".join(lines)
