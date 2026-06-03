"""
config.py — All configuration: API keys (env vars), constants, deal stages.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / os.getenv("OUTPUT_DIR", "outputs")
DATA_DIR = BASE_DIR / "data"
TEMPLATES_DIR = DATA_DIR / "templates"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Anthropic ─────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-6"
CLAUDE_HAIKU_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_MAX_TOKENS = 8192

import sys
if not ANTHROPIC_API_KEY:
    print("[ERROR] ANTHROPIC_API_KEY not set in .env — cannot run pipeline.")
    sys.exit(1)

def check_env_warnings():
    stubs = {
        "APOLLO_API_KEY": "Apollo enrichment",
        "BOX_ACCESS_TOKEN": "Box data room",
        "GMAIL_CLIENT_ID": "Gmail outreach",
        "MEETALFRED_API_KEY": "MeetAlfred campaigns",
    }
    for var, name in stubs.items():
        if not os.getenv(var):
            print(f"[WARN] {var} not set — {name} will run in stub (mock) mode")

# ── Box.com ───────────────────────────────────────────────────────────────────
BOX_CLIENT_ID = os.getenv("BOX_CLIENT_ID", "")
BOX_CLIENT_SECRET = os.getenv("BOX_CLIENT_SECRET", "")
BOX_ACCESS_TOKEN = os.getenv("BOX_ACCESS_TOKEN", "")
BOX_ROOT_FOLDER_ID = os.getenv("BOX_FOLDER_ID", "331277504511")

# ── Apollo.io ─────────────────────────────────────────────────────────────────
APOLLO_API_KEY = os.getenv("APOLLO_API_KEY", "")

# ── MeetAlfred ────────────────────────────────────────────────────────────────
MEETALFRED_API_KEY = os.getenv("MEETALFRED_API_KEY", "")

# ── Gmail ─────────────────────────────────────────────────────────────────────
GMAIL_CLIENT_ID = os.getenv("GMAIL_CLIENT_ID", "")
GMAIL_CLIENT_SECRET = os.getenv("GMAIL_CLIENT_SECRET", "")

# ── PostgreSQL / pgvector ─────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "")

# ── Document processing ───────────────────────────────────────────────────────
DOCUMENT_SYNTHESIS_PROVIDER = os.getenv("DOCUMENT_SYNTHESIS_PROVIDER", "disabled").strip().lower()
DOCUMENT_SYNTHESIS_ALLOW_EXTERNAL = os.getenv("DOCUMENT_SYNTHESIS_ALLOW_EXTERNAL", "").lower() in {"1", "true", "yes"}

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Workflow Constants ────────────────────────────────────────────────────────
FUNDING_DEADLINE_DAYS = 90
WEEKLY_OUTREACH_TARGET = 100
MEETINGS_PER_WEEK_TARGET = 1
COMMITMENTS_PER_MONTH_TARGET = 2
RESPONSE_RATE_ALERT_THRESHOLD = 0.20   # 20%
MEETING_CONVERSION_ALERT_THRESHOLD = 0.05  # 5%
BROKER_DEALER_THRESHOLD = 5_000_000    # $5M

# Step execution order
STEP_ORDER = [
    "01", "02", "03", "04", "05", "06",
    "07a", "07b", "07c", "08", "09", "10",
    "11", "12", "13", "14",
]

# Steps that are critical gates (pipeline pauses if these fail critically)
CRITICAL_GATE_STEPS = {"02"}  # Step 2 must produce tear sheet before 7a

# Reporting interval
REPORTING_INTERVAL_DAYS = 7

# AI retry settings
AI_MAX_RETRIES = 3
AI_RETRY_BASE_DELAY = 2  # seconds (exponential backoff)
