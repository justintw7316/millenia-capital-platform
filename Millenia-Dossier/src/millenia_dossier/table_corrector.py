from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .ollama_client import OllamaError, chat_json
from .utils import read_json, safe_text, write_json


def _fallback_clean_table(raw_table: dict[str, Any]) -> dict[str, Any]:
    columns = raw_table.get("columns") or []
    rows = raw_table.get("rows") or []
    if not columns and rows:
        columns = list(rows[0].keys())
    return {
        "doc_id": raw_table.get("doc_id"),
        "file_name": raw_table.get("file_name"),
        "item_id": raw_table.get("item_id"),
        "table_id": raw_table.get("table_id"),
        "page_no": raw_table.get("page_no"),
        "caption": "",
        "columns": columns,
        "rows": rows,
        "issues": ["Fallback used because LLM correction did not run or failed."],
    }


def build_table_prompt(raw_table: dict[str, Any]) -> str:
    raw_markdown = safe_text(raw_table.get("raw_markdown"))
    raw_html = safe_text(raw_table.get("raw_html"))[:8000]
    raw_rows = raw_table.get("rows") or []
    raw_columns = raw_table.get("columns") or []
    return f"""
You are correcting one table extracted from a PDF by Docling.

Goal:
Return a clean structured table with columns and rows for Excel/report use.

Rules:
- Do not invent missing values.
- Do not summarize the table.
- Preserve all numbers, names, dates, units, labels, and statuses.
- Fix broken headers, wrapped text, spacing issues, repeated headers, and obvious OCR artifacts.
- If a value is unclear, keep it as-is and add an issue.
- Return strict JSON only. No markdown fences.

Required JSON schema:
{{
  "caption": "short table caption if obvious, otherwise empty string",
  "columns": ["Column A", "Column B"],
  "rows": [
    {{"Column A": "value", "Column B": "value"}}
  ],
  "issues": ["brief issue notes if any"]
}}

Source metadata:
file_name: {raw_table.get('file_name')}
page_no: {raw_table.get('page_no')}
table_id: {raw_table.get('table_id')}

Docling raw columns:
{raw_columns}

Docling raw rows:
{raw_rows}

Docling raw markdown:
{raw_markdown}

Docling raw html/text fallback:
{raw_html}
""".strip()


def correct_tables(
    run_dir: Path,
    llm_model: str,
    ollama_url: str,
    table_ids: list[str] | None = None,
    use_llm: bool = True,
) -> list[dict[str, Any]]:
    raw_tables = read_json(run_dir / "raw_tables.json", default=[])
    selected = raw_tables
    if table_ids:
        selected = [t for t in raw_tables if t.get("table_id") in set(table_ids)]

    cleaned: list[dict[str, Any]] = []
    for raw in selected:
        if not use_llm:
            cleaned_table = _fallback_clean_table(raw)
        else:
            try:
                result = chat_json(build_table_prompt(raw), model=llm_model, ollama_url=ollama_url, temperature=0.05)
                cleaned_table = {
                    "doc_id": raw.get("doc_id"),
                    "file_name": raw.get("file_name"),
                    "item_id": raw.get("item_id"),
                    "table_id": raw.get("table_id"),
                    "page_no": raw.get("page_no"),
                    "caption": safe_text(result.get("caption", "")) if isinstance(result, dict) else "",
                    "columns": result.get("columns", []) if isinstance(result, dict) else [],
                    "rows": result.get("rows", []) if isinstance(result, dict) else [],
                    "issues": result.get("issues", []) if isinstance(result, dict) else ["LLM returned non-object JSON."],
                }
                if not cleaned_table["columns"] or not isinstance(cleaned_table["rows"], list):
                    fallback = _fallback_clean_table(raw)
                    fallback["issues"].append("LLM output had missing columns/rows, fallback used.")
                    cleaned_table = fallback
            except Exception as exc:
                cleaned_table = _fallback_clean_table(raw)
                cleaned_table["issues"].append(f"LLM correction failed: {exc}")
        cleaned.append(cleaned_table)

    write_json(run_dir / "cleaned_tables.json", cleaned)
    return cleaned


def cleaned_tables_to_markdown(cleaned_tables: list[dict[str, Any]]) -> str:
    blocks = []
    for table in cleaned_tables:
        rows = table.get("rows") or []
        columns = table.get("columns") or []
        if rows:
            try:
                df = pd.DataFrame(rows)
                if columns:
                    df = df[[c for c in columns if c in df.columns] + [c for c in df.columns if c not in columns]]
                md = df.to_markdown(index=False)
            except Exception:
                md = str(rows)
        else:
            md = "(No rows extracted.)"
        blocks.append(f"## {table.get('table_id')}\n\n{table.get('caption','')}\n\n{md}\n")
    return "\n\n".join(blocks)
