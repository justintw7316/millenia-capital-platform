import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, TypeVar
from urllib.parse import urlparse

import pandas as pd

from config import MONTH_NAMES, MONTH_NUMS

T = TypeVar("T")


def setup_logging(log_path: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure a shared pipeline logger for console and file output."""
    logger = logging.getLogger("investor_pipeline")
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def to_clean_text(val) -> str:
    """Normalize spreadsheet or parsed values into a trimmed string."""
    try:
        if pd.isna(val):
            return ""
    except TypeError:
        pass

    s = str(val).strip()
    if s.lower() in {"nan", "none", "nat"}:
        return ""
    return s


def to_float(val) -> Optional[float]:
    """Parse currency-like values that may use k/m/b shorthand."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = to_clean_text(val).replace(",", "")
    if not s:
        return None
    if s.startswith("$"):
        s = s[1:]
    m = re.search(r"(\d+(\.\d+)?)", s)
    if not m:
        return None
    num = float(m.group(1))
    s_low = s.lower()
    if "b" in s_low or "bn" in s_low or "billion" in s_low:
        num *= 1_000_000_000
    elif "m" in s_low or "mm" in s_low or "million" in s_low:
        num *= 1_000_000
    elif "k" in s_low or "thousand" in s_low:
        num *= 1_000
    return num


def parse_target_check_size_to_usd(target: str) -> Optional[float]:
    return to_float(target)


def parse_date(value) -> Optional[datetime]:
    """Parse spreadsheet dates into normalized datetimes."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value.replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
    except Exception:
        return None


def parse_dates_in_text(text: str) -> List[datetime]:
    """Extract plausible modern dates from noisy public text."""
    if not text:
        return []
    text_lower = text.lower()
    found: List[datetime] = []
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    for m in re.finditer(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", text):
        try:
            d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if 2020 <= d.year <= today.year + 1:
                found.append(d)
        except ValueError:
            pass

    for m in re.finditer(rf"\b{MONTH_NAMES}\s+(\d{{1,2}})[,\s]+(20\d{{2}})\b", text_lower, re.I):
        try:
            mon = next((v for k, v in MONTH_NUMS.items() if m.group(1).lower().startswith(k)), None)
            if mon is not None:
                found.append(datetime(int(m.group(3)), mon, int(m.group(2))))
        except (ValueError, IndexError):
            pass

    for m in re.finditer(rf"\b(\d{{1,2}})\s+{MONTH_NAMES}\s+(20\d{{2}})\b", text_lower, re.I):
        try:
            mon = next((v for k, v in MONTH_NUMS.items() if m.group(2).lower().startswith(k)), None)
            if mon is not None:
                found.append(datetime(int(m.group(3)), mon, int(m.group(1))))
        except (ValueError, IndexError):
            pass

    for m in re.finditer(rf"\b{MONTH_NAMES}\s+(20\d{{2}})\b", text_lower, re.I):
        try:
            mon = next((v for k, v in MONTH_NUMS.items() if m.group(1).lower().startswith(k)), None)
            if mon is not None:
                found.append(datetime(int(m.group(2)), mon, 1))
        except (ValueError, IndexError):
            pass

    for m in re.finditer(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b", text):
        try:
            found.append(datetime(int(m.group(3)), int(m.group(1)), int(m.group(2))))
        except ValueError:
            pass

    return [d for d in found if 2020 <= d.year <= today.year + 1]


def field_blob(inv: Dict, fields: List[str]) -> str:
    return " ".join(to_clean_text(inv.get(f, "")) for f in fields).lower()


def website_domain(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url if url.startswith("http") else f"https://{url}")
    return parsed.netloc.lower().replace("www.", "")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def dedupe_preserve_order(items: Iterable[T]) -> List[T]:
    """Remove duplicates while preserving the first seen order."""
    out: List[T] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def summarize_text(text: str, max_chars: int = 300) -> str:
    """Collapse whitespace and keep a short readable excerpt."""
    cleaned = re.sub(r"\s+", " ", to_clean_text(text))
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def build_evidence_summary(parts: Sequence[str], max_items: int = 6) -> str:
    trimmed = [p for p in parts if p][:max_items]
    return " | ".join(trimmed)
