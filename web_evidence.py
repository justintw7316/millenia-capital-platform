import logging
import re
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    BENIGN_HTTP_STATUSES,
    BLOCKED_URL_EXTENSIONS,
    BLOCKED_URL_FRAGMENTS,
    DDGS_QUERY_TEMPLATES,
    DDGS_RETRY_BACKOFF_SEC,
    DDGS_RETRY_TOTAL,
    DISCOVERY_KEYWORDS,
    ENTITY_MATCH_SETTINGS,
    HTTP_RETRY_BACKOFF_SEC,
    HTTP_RETRY_STATUSES,
    HTTP_RETRY_TOTAL,
    MAX_DDGS_SCRAPE_NEWS,
    MAX_DDGS_SCRAPE_OFFICIAL,
    MAX_SEARCH_RESULTS,
    MAX_WEB_PAGES_PER_INVESTOR,
    MIN_WEBSITE_SUMMARY_CHARS,
    NEWS_TRUST_DOMAIN_FRAGMENTS,
    RANKED_CANDIDATE_DEBUG_LIMIT,
    REQUEST_TIMEOUT_SEC,
    SCRAPE_TEXT_MAX_CHARS,
    SEARCH_DELAY_SEC,
    SOCIAL_DOMAIN_FRAGMENTS,
    TRUST_TIERS,
    TRUSTED_PATH_KEYWORDS,
    URL_BUCKET_SCORES,
    URL_RANKING_BONUSES,
    URL_RANKING_PENALTIES,
    USER_AGENT,
)
from utils import dedupe_preserve_order, parse_dates_in_text, to_clean_text, website_domain

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

logger = logging.getLogger("investor_pipeline.web")
TEXT_BUCKETS = ("team", "contact", "portfolio", "careers", "news", "general")
RECENCY_QUERY_KEYS = ("recent_deal_text", "new_fund_text", "hiring_text", "public_signal_text")
RECENT_ACTIVITY_TERMS = (
    "investment",
    "invested",
    "funding",
    "fund",
    "portfolio",
    "announced",
    "press release",
    "news",
    "hiring",
    "careers",
    "launch",
    "closed",
)


def _empty_website_evidence() -> Dict:
    return {
        "combined_text": "",
        "source_urls": [],
        "text_by_category": {bucket: "" for bucket in TEXT_BUCKETS},
        "_meta": {
            "attempted_home_urls": [],
            "attempted_page_urls": [],
            "discovered_pages": [],
            "errors": [],
            "warnings": [],
        },
    }


def _empty_search_signals() -> Dict:
    out = {key: "" for key in DDGS_QUERY_TEMPLATES}
    out["_ddgs_official_combined"] = ""
    out["_ddgs_news_combined"] = ""
    out["_ddgs_official_by_category"] = {bucket: "" for bucket in TEXT_BUCKETS}
    out["_ddgs_official_urls"] = []
    out["_ddgs_news_urls"] = []
    out["_meta"] = {
        "queries": {},
        "query_attempts": {},
        "query_errors": [],
        "query_warnings": [],
        "ranked_candidates": [],
        "scrape_errors": [],
        "scrape_warnings": [],
    }
    return out


def empty_search_signals() -> Dict:
    return _empty_search_signals()


def _build_session() -> requests.Session:
    retry = Retry(
        total=HTTP_RETRY_TOTAL,
        read=HTTP_RETRY_TOTAL,
        connect=HTTP_RETRY_TOTAL,
        status=HTTP_RETRY_TOTAL,
        backoff_factor=HTTP_RETRY_BACKOFF_SEC,
        status_forcelist=HTTP_RETRY_STATUSES,
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _safe_get(session: requests.Session, url: str) -> Tuple[Optional[requests.Response], Optional[str]]:
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT_SEC, allow_redirects=True)
    except requests.RequestException as exc:
        return None, str(exc)

    if response.status_code >= 400:
        return None, f"http {response.status_code}"

    content_type = (response.headers.get("content-type") or "").lower()
    if content_type and "html" not in content_type and "text/plain" not in content_type:
        return None, f"unsupported content-type: {content_type}"

    return response, None


def _extract_http_status(error: Optional[str]) -> Optional[int]:
    if not error:
        return None
    match = re.search(r"\bhttp (\d{3})\b", error.lower())
    if not match:
        return None
    return int(match.group(1))


def _is_benign_fetch_error(error: Optional[str]) -> bool:
    status = _extract_http_status(error)
    if status in BENIGN_HTTP_STATUSES:
        return True
    if error and "unsupported content-type" in error.lower():
        return True
    return False


def _normalize_phrase(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", to_clean_text(text).lower()).strip()


def _compact_alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", to_clean_text(text).lower())


def _investor_tokens(investor_name: str) -> List[str]:
    stopwords = set(ENTITY_MATCH_SETTINGS["name_stopwords"])
    tokens = [token for token in re.split(r"[^a-z0-9]+", investor_name.lower()) if len(token) > 1]
    significant = [token for token in tokens if token not in stopwords]
    return significant or tokens


def _url_mentions_investor(source_url: str, investor_name: str) -> bool:
    compact_name = _compact_alnum(investor_name)
    compact_url = _compact_alnum(source_url)
    if compact_name and compact_name in compact_url:
        return True

    tokens = _investor_tokens(investor_name)
    if not tokens:
        return False
    return sum(1 for token in set(tokens) if token in source_url.lower()) >= min(2, len(set(tokens)))


def is_investor_entity_match(investor_name: str, website: str, text: str, source_url: str = "") -> bool:
    """Require a stronger match for ambiguous names while preserving official-domain recall."""
    text_low = (text or "").lower()
    normalized_name = _normalize_phrase(investor_name)
    compact_name = _compact_alnum(investor_name)
    compact_text = _compact_alnum(text)
    domain = website_domain(website)
    source_domain = website_domain(source_url)

    if normalized_name and normalized_name in text_low:
        return True

    if domain and source_domain and (source_domain == domain or source_domain.endswith("." + domain)):
        return True

    if compact_name:
        short_name_len = ENTITY_MATCH_SETTINGS["short_name_compact_max_len"]
        if len(compact_name) <= short_name_len and (compact_name in compact_text or compact_name in source_domain.replace(".", "")):
            return True

    tokens = sorted(set(_investor_tokens(investor_name)))
    if not tokens:
        return False

    token_hits = sum(1 for token in tokens if token in text_low)
    required_hits = min(max(ENTITY_MATCH_SETTINGS["min_token_hits"], 1), len(tokens))
    token_ratio = token_hits / len(tokens)

    if token_hits >= required_hits and token_ratio >= ENTITY_MATCH_SETTINGS["min_token_ratio"]:
        return True

    return _url_mentions_investor(source_url, investor_name)


def normalize_url(url: str) -> str:
    if not url:
        return ""
    u = url.split("#")[0].strip().rstrip("/").lower()
    if u.startswith("http://"):
        u = "https://" + u[7:]
    return u


def _is_blocked_candidate_url(url: str) -> bool:
    lower_url = normalize_url(url)
    if not lower_url:
        return True
    if any(lower_url.endswith(ext) for ext in BLOCKED_URL_EXTENSIONS):
        return True
    if any(fragment in lower_url for fragment in BLOCKED_URL_FRAGMENTS):
        return True
    return False


def extract_page_evidence(html: str) -> str:
    """Prefer main editorial content and metadata over raw page chrome."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    meta_parts: List[str] = []
    for selector in ('meta[name="description"]', 'meta[property="og:description"]', 'meta[name="twitter:description"]'):
        element = soup.select_one(selector)
        if element and element.get("content"):
            meta_parts.append(to_clean_text(element.get("content", "")))

    main = soup.select_one(
        "main, article, [role='main'], #main, #content, .main-content, .content-main, .post-content, .entry-content"
    )
    if main:
        for tag in main(["nav", "footer", "aside", "form"]):
            tag.decompose()
        body_text = main.get_text(" ", strip=True)
    else:
        for tag in soup(["nav", "footer", "header"]):
            tag.decompose()
        body_text = soup.get_text(" ", strip=True)

    combined = " ".join(meta_parts + [body_text])
    return combined.lower()


def page_bucket(url: str) -> str:
    u = (url or "").lower()
    if any(p in u for p in ("/contact", "/reach", "get-in-touch", "/submit", "/pitch", "/apply")):
        return "contact"
    if any(p in u for p in ("/team", "/people", "/leadership", "/partners", "/staff")):
        return "team"
    if any(p in u for p in ("/portfolio", "/companies", "/investments", "/portfolio-companies")):
        return "portfolio"
    if any(p in u for p in ("/careers", "/jobs", "/join", "/hiring")):
        return "careers"
    if any(p in u for p in ("/news", "/press", "/media", "/blog")):
        return "news"
    return "general"


def discover_site_pages(home_url: str, html: str) -> List[str]:
    """Discover a small, ranked set of same-domain high-value pages."""
    candidates: List[Tuple[str, float]] = [(home_url, 100.0)]
    soup = BeautifulSoup(html, "html.parser")
    home_domain = website_domain(home_url)

    for anchor in soup.select("a[href]"):
        href = (anchor.get("href") or "").strip()
        if not href or href.startswith("#") or href.startswith("mailto:"):
            continue

        full = urljoin(home_url, href)
        parsed = urlparse(full)
        if parsed.scheme not in {"http", "https"}:
            continue
        if website_domain(full) != home_domain or _is_blocked_candidate_url(full):
            continue

        anchor_text = ((anchor.get_text(" ", strip=True) or "") + " " + href).lower()
        if not any(keyword in anchor_text for keyword in DISCOVERY_KEYWORDS):
            continue

        bucket = page_bucket(full)
        score = 10.0 + URL_BUCKET_SCORES.get(bucket, 0)
        score += sum(2.0 for keyword in DISCOVERY_KEYWORDS if keyword in anchor_text)
        if parsed.query:
            score -= 3.0
        if len(parsed.path.split("/")) > 5:
            score -= 2.0
        candidates.append((full, score))

    ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
    return dedupe_preserve_order(url for url, _ in ranked)[:MAX_WEB_PAGES_PER_INVESTOR]


def _candidate_home_urls(website: str) -> List[str]:
    clean_website = to_clean_text(website)
    if not clean_website:
        return []
    if clean_website.startswith("http://") or clean_website.startswith("https://"):
        return [clean_website]
    return [f"https://{clean_website}", f"http://{clean_website}"]


def _root_home_url(url: str) -> str:
    normalized = normalize_url(url)
    if not normalized:
        return ""
    parsed = urlparse(normalized if normalized.startswith("http") else f"https://{normalized}")
    if not parsed.netloc:
        return ""
    return f"https://{parsed.netloc.lower()}"


def _score_official_site_candidate(url: str, title: str, body: str, investor_name: str) -> float:
    normalized = normalize_url(url)
    dom = website_domain(normalized)
    if not dom:
        return -999.0
    if any(fragment in dom for fragment in SOCIAL_DOMAIN_FRAGMENTS):
        return -999.0
    if any(fragment in dom for fragment in NEWS_TRUST_DOMAIN_FRAGMENTS):
        return -999.0

    parsed = urlparse(normalized if normalized.startswith("http") else f"https://{normalized}")
    blob = f"{dom} {title} {body}".lower()
    score = 0.0

    if is_investor_entity_match(investor_name, "", blob, source_url=normalized):
        score += 25.0
    else:
        score -= 20.0

    if investor_name.lower() in blob:
        score += 16.0

    score += sum(4.0 for token in set(_investor_tokens(investor_name)) if token in blob)

    if parsed.path in {"", "/"}:
        score += 8.0
    else:
        bucket = page_bucket(normalized)
        if bucket in {"team", "portfolio", "contact", "general"}:
            score += 4.0
        elif bucket == "news":
            score += 2.0
        else:
            score -= 3.0

    if any(keyword in blob for keyword in ("official", "about", "team", "portfolio", "contact", "partners", "venture")):
        score += 6.0

    path_depth = len([part for part in parsed.path.split("/") if part])
    if path_depth >= 3:
        score -= 4.0
    return score


def discover_official_website(investor_name: str) -> Dict:
    """
    Use a small DDGS pass to identify a likely official website when none is supplied.
    The returned website is intended to be crawled first; snippets remain secondary.
    """
    out = {
        "website": "",
        "source_url": "",
        "meta": {
            "queries": [],
            "errors": [],
            "warnings": [],
            "ranked_candidates": [],
        },
    }

    queries = [
        f'"{investor_name}" official website',
        f'"{investor_name}" venture capital',
        f'"{investor_name}" portfolio team contact',
    ]

    try:
        ddgs = DDGS()
    except Exception as exc:
        out["meta"]["errors"].append(f"failed to initialize DDGS for website discovery: {exc}")
        return out

    merged_by_url: Dict[str, Dict] = {}
    for idx, query in enumerate(queries):
        entries, error = run_search(ddgs, query, max_results=MAX_SEARCH_RESULTS + 2)
        out["meta"]["queries"].append(
            {
                "query": query,
                "result_count": len(entries),
                "error": error or "",
            }
        )
        if error and not entries:
            out["meta"]["warnings"].append(f"website discovery query failed: {query}")
        by_url = _collect_ddgs_urls_from_entries(entries)
        for url, meta in by_url.items():
            existing = merged_by_url.get(url)
            if existing is None or len(meta["title"]) + len(meta["body"]) > len(existing["title"]) + len(existing["body"]):
                merged_by_url[url] = meta
        if idx < len(queries) - 1:
            time.sleep(SEARCH_DELAY_SEC)

    best_by_home: Dict[str, Dict] = {}
    for url, meta in merged_by_url.items():
        homepage = _root_home_url(url)
        if not homepage:
            continue
        score = _score_official_site_candidate(url, meta.get("title", ""), meta.get("body", ""), investor_name)
        current = best_by_home.get(homepage)
        if current is None or score > current["score"]:
            best_by_home[homepage] = {
                "homepage": homepage,
                "source_url": url,
                "score": score,
            }

    ranked = sorted(best_by_home.values(), key=lambda item: item["score"], reverse=True)
    out["meta"]["ranked_candidates"] = [
        {
            "website": item["homepage"],
            "source_url": item["source_url"],
            "score": round(item["score"], 2),
        }
        for item in ranked[:RANKED_CANDIDATE_DEBUG_LIMIT]
    ]

    if ranked and ranked[0]["score"] >= 20.0:
        out["website"] = ranked[0]["homepage"]
        out["source_url"] = ranked[0]["source_url"]
    else:
        out["meta"]["warnings"].append("no confident official website candidate found")
    return out


def fetch_website_evidence(website: str, investor_name: str) -> Dict:
    """Fetch a small set of official-site pages and bucket their extracted evidence."""
    out = _empty_website_evidence()
    out["_meta"]["requested_website"] = website
    if not website:
        out["_meta"]["warnings"].append("no website provided")
        return out

    session = _build_session()
    home_response: Optional[requests.Response] = None
    candidate_home_urls = _candidate_home_urls(website)
    out["_meta"]["attempted_home_urls"] = candidate_home_urls
    failed_home_attempts: List[str] = []

    for home_url in candidate_home_urls:
        response, error = _safe_get(session, home_url)
        if response is not None:
            home_response = response
            break
        failed_home_attempts.append(f"homepage fetch failed for {home_url}: {error}")

    if home_response is None:
        out["_meta"]["errors"].extend(failed_home_attempts)
        return out
    if failed_home_attempts:
        out["_meta"]["warnings"].extend(failed_home_attempts)

    base_url = home_response.url
    try:
        pages = discover_site_pages(base_url, home_response.text)
    except Exception as exc:
        out["_meta"]["errors"].append(f"page discovery failed: {exc}")
        pages = [base_url]

    out["_meta"]["discovered_pages"] = pages
    by_category: Dict[str, List[str]] = {bucket: [] for bucket in TEXT_BUCKETS}
    texts: List[str] = []

    for page in pages[:MAX_WEB_PAGES_PER_INVESTOR]:
        out["_meta"]["attempted_page_urls"].append(page)
        response, error = _safe_get(session, page)
        if response is None:
            message = f"page fetch failed for {page}: {error}"
            if _is_benign_fetch_error(error):
                out["_meta"]["warnings"].append(message)
            else:
                out["_meta"]["errors"].append(message)
            continue

        try:
            text = extract_page_evidence(response.text)
        except Exception as exc:
            out["_meta"]["errors"].append(f"page parsing failed for {page}: {exc}")
            continue

        if not is_investor_entity_match(investor_name, website, text, source_url=response.url):
            continue

        chunk = text[:SCRAPE_TEXT_MAX_CHARS]
        texts.append(chunk)
        out["source_urls"].append(response.url)
        bucket = page_bucket(response.url)
        by_category[bucket if bucket in by_category else "general"].append(chunk)

    out["source_urls"] = dedupe_preserve_order(out["source_urls"])
    out["combined_text"] = " ".join(texts).lower()
    for bucket, parts in by_category.items():
        out["text_by_category"][bucket] = " ".join(parts).lower()
    out["_meta"]["resolved_home_url"] = base_url
    return out


def _website_has_recent_activity_signals(website_evidence: Dict) -> bool:
    by_category = website_evidence.get("text_by_category") or {}
    recent_context = " ".join(
        piece
        for piece in (
            by_category.get("news", ""),
            by_category.get("portfolio", ""),
            by_category.get("careers", ""),
            website_evidence.get("combined_text", ""),
        )
        if piece
    ).lower()
    if not recent_context.strip():
        return False

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    recent_dates = [d for d in parse_dates_in_text(recent_context) if 0 <= (today - d).days <= 365]
    term_hits = sum(1 for term in RECENT_ACTIVITY_TERMS if term in recent_context)
    if recent_dates and term_hits >= 1:
        return True
    if term_hits >= 3 and any(marker in recent_context for marker in ("announced", "invested", "led", "launch", "closed", "hiring")):
        return True
    return False


def evaluate_search_strategy(website_evidence: Dict) -> Dict:
    """
    Decide which DDGS queries to run alongside the official crawl.
    Public news/press signals are now a core input, so a public-signal sweep
    always runs. The only thing that changes with the official-crawl quality
    is whether we also run the broader fit-style queries.
    """
    combined_text = (website_evidence.get("combined_text") or "").strip()
    source_urls = list(website_evidence.get("source_urls") or [])
    has_website_evidence = bool(source_urls) or len(combined_text) >= MIN_WEBSITE_SUMMARY_CHARS

    if not has_website_evidence:
        return {
            "run_search": True,
            "query_keys": list(DDGS_QUERY_TEMPLATES.keys()),
            "reasons": ["website_evidence_weak_or_missing"],
        }

    if not _website_has_recent_activity_signals(website_evidence):
        return {
            "run_search": True,
            "query_keys": list(RECENCY_QUERY_KEYS),
            "reasons": ["augment_official_crawl_with_public_signals"],
        }

    return {
        "run_search": True,
        "query_keys": list(RECENCY_QUERY_KEYS),
        "reasons": ["public_signals_collected_alongside_official_crawl"],
    }


def _url_trust_tier(url: str, official_domain: str) -> str:
    dom = website_domain(url)
    if not dom:
        return "other"
    if official_domain and (dom == official_domain or dom.endswith("." + official_domain)):
        return "official"
    for frag in NEWS_TRUST_DOMAIN_FRAGMENTS:
        if frag in dom:
            return "news"
    return "other"


def _score_ddgs_candidate(url: str, title: str, body: str, investor_name: str, official_domain: str) -> float:
    tier = _url_trust_tier(url, official_domain)
    score = TRUST_TIERS.get(tier, TRUST_TIERS["other"])
    bucket = page_bucket(url)
    score += URL_BUCKET_SCORES.get(bucket, 0)

    blob = f"{title} {body}".lower()
    if investor_name.lower() in blob:
        score += URL_RANKING_BONUSES["investor_name_phrase"]
    if official_domain and official_domain in url.lower():
        score += URL_RANKING_BONUSES["official_domain_url"]
    if any(keyword in url.lower() for keyword in TRUSTED_PATH_KEYWORDS):
        score += URL_RANKING_BONUSES["trusted_path_keyword"]

    url_lower = url.lower()
    if url_lower.endswith(".pdf"):
        score -= URL_RANKING_PENALTIES["pdf"]
    if "linkedin.com/in/" in url_lower:
        score -= URL_RANKING_PENALTIES["profile"]
    if any(fragment in url_lower for fragment in SOCIAL_DOMAIN_FRAGMENTS):
        score -= URL_RANKING_PENALTIES["social"]
    if any(fragment in url_lower for fragment in BLOCKED_URL_FRAGMENTS):
        score -= URL_RANKING_PENALTIES["blocked_fragment"]
    return score


def _collect_ddgs_urls_from_entries(entries: List[Dict]) -> Dict[str, Dict]:
    """Normalize URL -> best title/body seen."""
    by_url: Dict[str, Dict] = {}
    for entry in entries:
        href = to_clean_text(entry.get("href", ""))
        if not href or _is_blocked_candidate_url(href):
            continue
        parsed = urlparse(href if href.startswith("http") else f"https://{href}")
        if parsed.scheme not in {"http", "https"}:
            continue
        clean = normalize_url(href)
        title = to_clean_text(entry.get("title", ""))
        body = to_clean_text(entry.get("body", ""))
        if clean not in by_url:
            by_url[clean] = {"title": title, "body": body}
        else:
            current = by_url[clean]
            if len(title) + len(body) > len(current["title"]) + len(current["body"]):
                by_url[clean] = {"title": title, "body": body}
    return by_url


def _rank_ddgs_urls(
    by_url: Dict[str, Dict],
    investor_name: str,
    official_domain: str,
    skip_urls: Optional[Set[str]],
) -> List[Tuple[str, str, float]]:
    skip_norm = {normalize_url(url) for url in (skip_urls or set())}
    ranked: List[Tuple[str, str, float]] = []
    for url, meta in by_url.items():
        normalized = normalize_url(url)
        if not normalized or normalized in skip_norm or _is_blocked_candidate_url(normalized):
            continue
        tier = _url_trust_tier(normalized, official_domain)
        score = _score_ddgs_candidate(
            normalized,
            meta.get("title", ""),
            meta.get("body", ""),
            investor_name,
            official_domain,
        )
        ranked.append((normalized, tier, score))
    ranked.sort(key=lambda item: item[2], reverse=True)
    return ranked


def _scrape_ddgs_candidates(
    ranked: List[Tuple[str, str, float]],
    investor_name: str,
    website: str,
    official_domain: str,
) -> Dict:
    """Fetch ranked DDGS URLs and keep only official or trusted news evidence."""
    session = _build_session()
    official_chunks: List[str] = []
    news_chunks: List[str] = []
    official_by_cat: Dict[str, List[str]] = defaultdict(list)
    official_urls_used: List[str] = []
    news_urls_used: List[str] = []
    errors: List[str] = []
    warnings: List[str] = []
    official_count = 0
    news_count = 0

    for url, tier, score in ranked:
        if tier == "other" or (tier == "news" and score < 35.0):
            continue
        if tier == "official" and official_count >= MAX_DDGS_SCRAPE_OFFICIAL:
            continue
        if tier == "news" and news_count >= MAX_DDGS_SCRAPE_NEWS:
            continue

        response, error = _safe_get(session, url)
        if response is None:
            message = f"scrape failed for {url}: {error}"
            if _is_benign_fetch_error(error):
                warnings.append(message)
            else:
                errors.append(message)
            continue

        try:
            text = extract_page_evidence(response.text)
        except Exception as exc:
            errors.append(f"parse failed for {url}: {exc}")
            continue

        if not is_investor_entity_match(investor_name, website, text, source_url=response.url):
            continue

        chunk = text[:SCRAPE_TEXT_MAX_CHARS]
        if tier == "official":
            official_chunks.append(chunk)
            official_urls_used.append(response.url)
            bucket = page_bucket(response.url)
            official_by_cat[bucket if bucket in TEXT_BUCKETS else "general"].append(chunk)
            official_count += 1
        elif tier == "news":
            news_chunks.append(chunk)
            news_urls_used.append(response.url)
            news_count += 1

    return {
        "official_combined": " ".join(official_chunks).lower(),
        "news_combined": " ".join(news_chunks).lower(),
        "official_by_category": {
            bucket: " ".join(official_by_cat.get(bucket, [])).lower() for bucket in TEXT_BUCKETS
        },
        "official_urls": dedupe_preserve_order(official_urls_used),
        "news_urls": dedupe_preserve_order(news_urls_used),
        "errors": errors,
        "warnings": warnings,
    }


def run_search(ddgs, query: str, max_results: int = 5) -> Tuple[List[Dict], Optional[str]]:
    """Run a DDGS query with light retry/backoff."""
    for attempt in range(DDGS_RETRY_TOTAL + 1):
        try:
            return list(ddgs.text(query, max_results=max_results)), None
        except Exception as exc:
            if attempt >= DDGS_RETRY_TOTAL:
                return [], str(exc)
            sleep_for = DDGS_RETRY_BACKOFF_SEC * (2**attempt)
            logger.warning("DDGS query retrying in %.2fs after error: %s", sleep_for, exc)
            time.sleep(sleep_for)
    return [], "unknown DDGS failure"


def _build_query_variants(name: str, year: int, domain: str) -> Dict[str, List[Tuple[str, str]]]:
    site_bias = f"site:{domain}" if domain else ""
    variants: Dict[str, List[Tuple[str, str]]] = {}
    for key, template in DDGS_QUERY_TEMPLATES.items():
        primary = template.format(name=name, year=year, site_bias=site_bias).strip()
        fallback = template.format(name=name, year=year, site_bias="").strip()
        query_variants = [("primary", primary)]
        if fallback != primary:
            query_variants.append(("fallback_no_site_bias", fallback))
        variants[key] = query_variants
    return variants


def _run_search_with_fallbacks(ddgs, query_variants: List[Tuple[str, str]], max_results: int) -> Tuple[List[Dict], List[Dict], List[str]]:
    attempts: List[Dict] = []
    warnings: List[str] = []
    for label, query in query_variants:
        entries, error = run_search(ddgs, query, max_results=max_results)
        attempts.append(
            {
                "label": label,
                "query": query,
                "result_count": len(entries),
                "error": error or "",
            }
        )
        if entries:
            if label != "primary":
                warnings.append(f"used {label} DDGS query fallback")
            return entries, attempts, warnings
        if error and label != query_variants[-1][0]:
            warnings.append(f"{label} DDGS query failed; trying fallback")
    return [], attempts, warnings


def search_investor_signals(
    name: str,
    website: str,
    skip_urls: Optional[Set[str]] = None,
    query_keys: Optional[List[str]] = None,
) -> Dict:
    """
    Use DDGS to discover candidate URLs, then scrape only official and trusted-news pages.
    Returned string fields remain DDGS snippets; underscore keys hold scraped evidence.
    """
    out = _empty_search_signals()
    domain = website_domain(website)
    year = datetime.now().year
    requested_keys = list(DDGS_QUERY_TEMPLATES.keys()) if query_keys is None else list(query_keys)
    selected_keys = [key for key in requested_keys if key in DDGS_QUERY_TEMPLATES]
    out["_meta"]["query_mode"] = "full" if set(selected_keys) == set(DDGS_QUERY_TEMPLATES.keys()) else "targeted"
    out["_meta"]["query_keys"] = selected_keys
    if not selected_keys:
        out["_meta"]["query_mode"] = "skipped"
        out["_meta"]["query_warnings"].append("search skipped because official crawl was sufficient")
        return out

    query_variants = _build_query_variants(name, year, domain)
    out["_meta"]["queries"] = {key: query_variants[key][0][1] for key in selected_keys}
    skip_set: Set[str] = set(skip_urls or set())
    merged_by_url: Dict[str, Dict] = {}

    try:
        ddgs = DDGS()
    except Exception as exc:
        out["_meta"]["query_errors"].append(f"failed to initialize DDGS: {exc}")
        return out

    for key in selected_keys:
        variants = query_variants[key]
        entries, attempts, warnings = _run_search_with_fallbacks(ddgs, variants, max_results=MAX_SEARCH_RESULTS)
        out["_meta"]["query_attempts"][key] = attempts
        out["_meta"]["query_warnings"].extend(f"{key}: {warning}" for warning in warnings)
        terminal_error = next((attempt["error"] for attempt in reversed(attempts) if attempt["error"]), "")
        if terminal_error and not entries:
            out["_meta"]["query_errors"].append(f"{key}: {terminal_error}")

        snippets: List[str] = []
        by_url = _collect_ddgs_urls_from_entries(entries)
        for entry in entries:
            title = to_clean_text(entry.get("title", ""))
            body = to_clean_text(entry.get("body", ""))
            href = to_clean_text(entry.get("href", ""))
            joined = f"{title} {body}".lower()
            if is_investor_entity_match(name, website, joined, source_url=href):
                snippets.append(joined)
        for url, meta in by_url.items():
            existing = merged_by_url.get(url)
            if existing is None or len(meta["title"]) + len(meta["body"]) > len(existing["title"]) + len(existing["body"]):
                merged_by_url[url] = meta
        out[key] = " ".join(snippets).lower()
        time.sleep(SEARCH_DELAY_SEC)

    ranked = _rank_ddgs_urls(merged_by_url, name, domain, skip_set)
    out["_meta"]["ranked_candidates"] = [
        {"url": url, "tier": tier, "score": round(score, 2)}
        for url, tier, score in ranked[:RANKED_CANDIDATE_DEBUG_LIMIT]
    ]

    scraped = _scrape_ddgs_candidates(ranked, name, website, domain)
    out["_ddgs_official_combined"] = scraped["official_combined"]
    out["_ddgs_news_combined"] = scraped["news_combined"]
    out["_ddgs_official_by_category"] = scraped["official_by_category"]
    out["_ddgs_official_urls"] = scraped["official_urls"]
    out["_ddgs_news_urls"] = scraped["news_urls"]
    out["_meta"]["scrape_errors"] = scraped["errors"]
    out["_meta"]["scrape_warnings"] = scraped["warnings"]
    return out
