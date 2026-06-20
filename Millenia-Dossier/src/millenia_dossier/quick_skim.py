from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageStat


def quick_skim_crop(crop_path: str | Path | None, item_type: str = "picture") -> dict[str, Any]:
    if not crop_path:
        return {"label": "needs_review", "reason": "No crop path was available."}

    path = Path(crop_path)
    if not path.exists():
        return {"label": "needs_review", "reason": "Crop file was missing."}

    try:
        image = Image.open(path).convert("RGB")
        width, height = image.size
        area = width * height
        gray = image.convert("L")
        arr = np.asarray(gray).astype("float32")
        std = float(arr.std())
        mean = float(arr.mean())

        dx = np.abs(np.diff(arr, axis=1)).mean() if width > 1 else 0.0
        dy = np.abs(np.diff(arr, axis=0)).mean() if height > 1 else 0.0
        detail_score = float((dx + dy) / 2.0)

        white_ratio = float((arr > 245).sum() / arr.size)

        if item_type == "table":
            if area < 8_000:
                return {"label": "needs_review", "reason": "Small table crop; review before using."}
            if white_ratio > 0.98 and detail_score < 2.0:
                return {"label": "needs_review", "reason": "Table crop looks mostly blank; verify manually."}
            return {"label": "likely_useful", "reason": "Table crops are kept conservative by default."}

        if area < 6_000:
            return {"label": "likely_decorative", "reason": "Very small crop."}
        if white_ratio > 0.985 and detail_score < 2.0:
            return {"label": "likely_empty", "reason": "Mostly blank/white crop."}
        if std < 8.0 and detail_score < 3.0:
            return {"label": "likely_decorative", "reason": "Low visual detail."}
        return {"label": "likely_useful", "reason": "Crop has enough visual detail to review."}
    except Exception as exc:
        return {"label": "needs_review", "reason": f"Quick skim failed: {exc}"}
