import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from config import FIELD_MAP, HEADER_ROW, REQUIRED_INVESTOR_FIELDS
from utils import to_clean_text

logger = logging.getLogger("investor_pipeline.data_loader")


def _validate_loaded_investor(inv: Dict) -> bool:
    """Keep only rows that satisfy the minimal pipeline contract."""
    for field in REQUIRED_INVESTOR_FIELDS:
        if not to_clean_text(inv.get(field)):
            return False
    return True


def load_investors(path: Path, max_n: int) -> List[Dict]:
    """Load and normalize investor rows from the workbook."""
    try:
        df = pd.read_excel(path, header=HEADER_ROW)
    except Exception as exc:
        raise RuntimeError(f"Failed to read investor workbook at {path}: {exc}") from exc

    rows: List[Dict] = []
    seen = set()

    for idx, row in df.iterrows():
        name = to_clean_text(row.get("Investors"))
        if not name:
            continue
        if name in seen:
            logger.debug("Skipping duplicate investor row: %s", name)
            continue

        inv: Dict = {}
        for excel_col, key in FIELD_MAP.items():
            inv[key] = to_clean_text(row.get(excel_col))
        inv["name"] = name

        if not _validate_loaded_investor(inv):
            logger.warning("Skipping row %s due to missing required fields for investor %r", idx, name)
            continue

        seen.add(name)
        rows.append(inv)
        if len(rows) >= max_n:
            break

    logger.info("Loaded %s unique investors from %s", len(rows), path)
    return rows
