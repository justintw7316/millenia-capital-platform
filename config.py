from pathlib import Path

TARGET_SECTOR = "fintech"
TARGET_STAGE = "seed"
TARGET_CHECK_SIZE = "$1M"
TARGET_GEOGRAPHY = "U.S."

BASE_DIR = Path(__file__).parent

INPUTS = {
    "excel_path": BASE_DIR / "Dallas Investors.xlsx",
    "header_row": 6,
    "max_investors": 15,
}

OUTPUTS = {
    "csv": BASE_DIR / "investor_signals_report.csv",
    "debug_jsonl": BASE_DIR / "investor_signals_debug.jsonl",
    "run_summary_json": BASE_DIR / "investor_signals_run_summary.json",
    "log": BASE_DIR / "investor_signals_pipeline.log",
}

NETWORK = {
    "request_timeout_sec": 8,
    "search_delay_sec": 1.0,
    "http_retry_total": 2,
    "http_retry_backoff_sec": 0.75,
    "http_retry_statuses": (429, 500, 502, 503, 504),
    "ddgs_retry_total": 2,
    "ddgs_retry_backoff_sec": 1.0,
    "user_agent": "Mozilla/5.0 (compatible; InvestorSignalBot/1.0; +https://example.com/bot)",
}


BENIGN_HTTP_STATUSES = (401, 403, 404, 410)

CRAWL_LIMITS = {
    "max_web_pages_per_investor": 4,
    "max_search_results": 5,
    "max_ddgs_scrape_official": 4,
    "max_ddgs_scrape_news": 2,
    "scrape_text_max_chars": 8000,
}

TRUST_TIERS = {
    "official": 120.0,
    "news": 55.0,
    "other": 8.0,
}

TRUST_DOMAIN_FRAGMENTS = {
    "news": (
        "bloomberg.com",
        "reuters.com",
        "wsj.com",
        "ft.com",
        "techcrunch.com",
        "axios.com",
        "forbes.com",
        "cnbc.com",
        "venturebeat.com",
        "prnewswire.com",
        "businesswire.com",
        "nytimes.com",
        "theinformation.com",
        "pitchbook.com",
        "sifted.eu",
    ),
}

URL_BUCKET_SCORES = {
    "portfolio": 14,
    "team": 14,
    "contact": 12,
    "careers": 10,
    "news": 9,
    "general": 2,
}

URL_RANKING_BONUSES = {
    "investor_name_phrase": 18.0,
    "official_domain_url": 25.0,
    "trusted_path_keyword": 5.0,
}

URL_RANKING_PENALTIES = {
    "pdf": 80.0,
    "social": 18.0,
    "profile": 25.0,
    "blocked_fragment": 10.0,
}

DISCOVERY_KEYWORDS = (
    "about",
    "team",
    "portfolio",
    "news",
    "press",
    "blog",
    "careers",
    "jobs",
    "contact",
    "pitch",
    "submit",
    "founders",
    "apply",
    "people",
    "leadership",
)

TRUSTED_PATH_KEYWORDS = (
    "portfolio",
    "invest",
    "investment",
    "team",
    "partner",
    "principal",
    "contact",
    "careers",
    "founder",
    "submit",
)

BLOCKED_URL_EXTENSIONS = (
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".zip",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
)

BLOCKED_URL_FRAGMENTS = (
    "/feed",
    "/tag/",
    "/category/",
    "/wp-content/",
    "/wp-json/",
    "utm_",
    "share=",
    "javascript:",
)

SOCIAL_DOMAIN_FRAGMENTS = (
    "linkedin.com",
    "twitter.com",
    "x.com",
    "facebook.com",
    "instagram.com",
    "youtube.com",
    "tiktok.com",
)

ENTITY_MATCH_SETTINGS = {
    "min_token_hits": 2,
    "min_token_ratio": 0.6,
    "short_name_compact_max_len": 6,
    "name_stopwords": (
        "capital",
        "ventures",
        "venture",
        "management",
        "group",
        "partners",
        "partner",
        "investments",
        "investment",
        "holdings",
        "fund",
        "funds",
        "equity",
        "vc",
        "lp",
        "llc",
        "co",
        "company",
    ),
}

MERGE_THIN_THRESHOLDS = {
    "default": 12,
    "geography": 8,
}

DDGS_QUERY_TEMPLATES = {
    "recent_deal_text": '"{name}" investment funding round led {year} {site_bias}',
    "new_fund_text": '"{name}" new fund raised closes {site_bias}',
    "hiring_text": '"{name}" careers hiring investment team associate {site_bias}',
    "public_signal_text": '"{name}" portfolio press announcement investment',
    "outreach_text": '"{name}" partner principal contact pitch founders submit deck {site_bias}',
}

SEARCH_SIGNAL_KEYS = tuple(DDGS_QUERY_TEMPLATES.keys())

RECENT_DEAL_DAYS = 90
INTENT_ACTIVE_THRESHOLD = 5
FIT_SCORE_CAP = 10
INTENT_SCORE_CAP = 10
DEBUG_TEXT_SUMMARY_CHARS = 400
RANKED_CANDIDATE_DEBUG_LIMIT = 12
REQUIRED_INVESTOR_FIELDS = ("name",)
MIN_WEBSITE_SUMMARY_CHARS = 120
MIN_SEARCH_SUMMARY_CHARS = 120

EXCEL_PATH = INPUTS["excel_path"]
HEADER_ROW = INPUTS["header_row"]
MAX_INVESTORS = INPUTS["max_investors"]

OUTPUT_CSV = OUTPUTS["csv"]
OUTPUT_DEBUG_JSONL = OUTPUTS["debug_jsonl"]
OUTPUT_RUN_SUMMARY_JSON = OUTPUTS["run_summary_json"]
OUTPUT_LOG = OUTPUTS["log"]

SEARCH_DELAY_SEC = NETWORK["search_delay_sec"]
REQUEST_TIMEOUT_SEC = NETWORK["request_timeout_sec"]
HTTP_RETRY_TOTAL = NETWORK["http_retry_total"]
HTTP_RETRY_BACKOFF_SEC = NETWORK["http_retry_backoff_sec"]
HTTP_RETRY_STATUSES = NETWORK["http_retry_statuses"]
DDGS_RETRY_TOTAL = NETWORK["ddgs_retry_total"]
DDGS_RETRY_BACKOFF_SEC = NETWORK["ddgs_retry_backoff_sec"]
USER_AGENT = NETWORK["user_agent"]


MAX_WEB_PAGES_PER_INVESTOR = CRAWL_LIMITS["max_web_pages_per_investor"]
MAX_SEARCH_RESULTS = CRAWL_LIMITS["max_search_results"]
MAX_DDGS_SCRAPE_OFFICIAL = CRAWL_LIMITS["max_ddgs_scrape_official"]
MAX_DDGS_SCRAPE_NEWS = CRAWL_LIMITS["max_ddgs_scrape_news"]
SCRAPE_TEXT_MAX_CHARS = CRAWL_LIMITS["scrape_text_max_chars"]

NEWS_TRUST_DOMAIN_FRAGMENTS = TRUST_DOMAIN_FRAGMENTS["news"]

SECTOR_KEYWORDS = {
    "ai": ["artificial intelligence", "ai", "machine learning", "ml", "data science", "software"],
    "fintech": ["fintech", "financial", "finance", "payments", "banking", "insurtech", "lending"],
    "real estate": ["real estate", "office", "industrial", "multifamily", "commercial", "retail", "proptech"],
    "hospitality": ["hospitality", "hotel", "travel", "leisure"],
    "sports": ["sports", "fitness", "athletic"],
    "entertainment": ["entertainment", "media", "consumer internet", "digital media"],
}

STAGE_KEYWORDS = {
    "pre-seed": ["pre-seed", "angel", "incubator", "accelerator", "early stage"],
    "seed": ["seed", "startup", "startups", "early stage", "venture capital", "vc"],
    "series a": ["series a", "a round", "institutional seed", "venture capital", "vc"],
    "growth": ["growth", "expansion", "late stage", "series b", "series c", "scale"],
    "private equity": ["private equity", "pe", "buyout", "acquisition"],
}

GENERIC_SECTOR_HINTS = ["technology", "tech", "software", "digital"]

FIELD_MAP = {
    "Investors": "name",
    "Description": "description",
    "Primary Investor Type": "primary_type",
    "Other Investor Types": "other_types",
    "Website": "website",
    "Investor Status": "investor_status",
    "Primary Industry Sector": "primary_industry_sector",
    "Primary Industry Group": "primary_industry_group",
    "All Industries": "all_industries",
    "Verticals": "verticals",
    "Keywords": "keywords",
    "Investments in the last 7 days": "investments_7d",
    "Investments in the last 6 months": "investments_6m",
    "Investments in the last 12 months": "investments_12m",
    "Last Investment Company": "last_investment_company",
    "Last Investment Date": "last_investment_date",
    "Last Investment Size": "last_investment_size",
    "Last Closed Fund Name": "last_closed_fund_name",
    "Last Closed Fund Size": "last_closed_fund_size",
    "Last Closed Fund Close Date": "last_closed_fund_close_date",
    "Preferred Industry": "preferred_industry",
    "Preferred Verticals": "preferred_verticals",
    "Preferred Geography": "preferred_geography",
    "Preferred Investment Types": "preferred_investment_types",
    "Preferred Investment Amount": "preferred_investment_amount",
    "Preferred Investment Amount Min": "preferred_investment_amount_min",
    "Preferred Investment Amount Max": "preferred_investment_amount_max",
    "Preferred Deal Size": "preferred_deal_size",
    "Preferred Deal Size Min": "preferred_deal_size_min",
    "Preferred Deal Size Max": "preferred_deal_size_max",
    "Latest Note": "latest_note",
    "Most Likely Fundraising": "most_likely_fundraising",
    "HQ Location": "hq_location",
    "HQ City": "hq_city",
    "HQ State/Province": "hq_state",
    "HQ Country/Territory/Region": "hq_country",
}

MONTH_NAMES = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
MONTH_NUMS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
