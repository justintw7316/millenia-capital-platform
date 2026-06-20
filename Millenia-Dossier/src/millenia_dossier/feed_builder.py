from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import read_json, safe_text, write_text


def _table_to_markdown(table: dict[str, Any]) -> str:
    rows = table.get("rows") or []
    if not rows:
        return "(No cleaned table rows available.)"
    try:
        df = pd.DataFrame(rows)
        columns = table.get("columns") or []
        if columns:
            present = [c for c in columns if c in df.columns]
            remainder = [c for c in df.columns if c not in present]
            df = df[present + remainder]
        return df.to_markdown(index=False)
    except Exception:
        return str(rows)


def build_llm_feed(run_dir: Path) -> str:
    layout_items = read_json(run_dir / "layout_items.json", default=[])
    cleaned_tables = read_json(run_dir / "cleaned_tables.json", default=[])
    raw_tables = read_json(run_dir / "raw_tables.json", default=[])
    image_summaries = read_json(run_dir / "image_summaries.json", default=[])

    cleaned_by_item = {t.get("item_id"): t for t in cleaned_tables}
    raw_by_item = {t.get("item_id"): t for t in raw_tables}
    image_by_item = {v.get("item_id"): v for v in image_summaries}

    lines: list[str] = []
    current_doc = None

    for item in sorted(layout_items, key=lambda x: (x.get("doc_id", ""), x.get("reading_order_index", 0))):
        doc_id = item.get("doc_id")
        if doc_id != current_doc:
            current_doc = doc_id
            lines.append(f"\n# Document: {item.get('file_name')}\n")
        item_type = item.get("item_type")
        status = item.get("human_status", "keep")
        if status == "discard":
            continue
        if item_type == "section_header":
            text = safe_text(item.get("text"))
            if text:
                lines.append(f"\n## {text}\n")
        elif item_type == "list_item":
            text = safe_text(item.get("text"))
            if text:
                lines.append(f"- {text}")
        elif item_type == "text":
            text = safe_text(item.get("text"))
            if text:
                lines.append(text)
        elif item_type == "table":
            table = cleaned_by_item.get(item.get("item_id"))
            if table:
                lines.append(f"\n### Cleaned Table: {table.get('table_id')}\n")
                if table.get("caption"):
                    lines.append(f"Caption: {table.get('caption')}\n")
                lines.append(_table_to_markdown(table))
            else:
                raw = raw_by_item.get(item.get("item_id"), {})
                lines.append(f"\n### Raw Docling Table: {raw.get('table_id', item.get('item_id'))}\n")
                lines.append(safe_text(raw.get("raw_markdown")) or "(No raw table text available.)")
        elif item_type == "picture":
            summary = image_by_item.get(item.get("item_id"))
            if summary:
                lines.append(f"\n### Visual Summary: {summary.get('visual_id')}\n")
                lines.append(f"Type: {summary.get('visual_type')}\n")
                lines.append(f"Summary: {summary.get('summary')}\n")
                visible = summary.get("visible_text") or []
                if visible:
                    lines.append("Visible text: " + "; ".join(map(str, visible)))
            else:
                note = safe_text(item.get("human_note"))
                if note:
                    lines.append(f"\n### Visual Reviewer Note\n{note}\n")
    feed = "\n\n".join([x for x in lines if str(x).strip()])
    write_text(run_dir / "file_llm_feed.md", feed)
    return feed
