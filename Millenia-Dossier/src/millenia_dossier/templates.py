from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "templates"


def list_templates() -> list[str]:
    return sorted([p.stem for p in TEMPLATE_DIR.glob("*.json")])


def load_template(name: str) -> dict[str, Any]:
    path = TEMPLATE_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {name}")
    return json.loads(path.read_text(encoding="utf-8"))
