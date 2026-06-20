from __future__ import annotations

from pathlib import Path
from typing import Any

from .ollama_client import chat_vision_json
from .utils import read_json, safe_text, write_json


def update_visual_notes(run_dir: Path, updates: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    visual_items = read_json(run_dir / "visual_items.json", default=[])
    for item in visual_items:
        visual_id = item.get("visual_id")
        if visual_id in updates:
            item.update(updates[visual_id])
    write_json(run_dir / "visual_items.json", visual_items)
    return visual_items


def build_visual_prompt(item: dict[str, Any]) -> str:
    return f"""
You are analyzing one visual item from a document.

The image may be a chart, figure, diagram, screenshot, logo, decorative graphic, or photo.

Source:
file_name: {item.get('file_name')}
page_no: {item.get('page_no')}
visual_id: {item.get('visual_id')}
quick_skim: {item.get('quick_skim')}

Nearby document text:
{safe_text(item.get('nearby_text'))}

Human reviewer note:
{safe_text(item.get('human_note')) or '(none)'}

Task:
Return strict JSON only with this schema:
{{
  "visual_type": "chart|diagram|figure|photo|logo|decorative|unknown",
  "summary": "clear useful description for later document analysis",
  "visible_text": ["text visible in the image, if any"],
  "importance": "useful|decorative|unclear",
  "issues": ["uncertainties or warnings"]
}}

Rules:
- Use the human reviewer note as context, but do not blindly assume it is correct.
- Do not invent details that are not visible or supported by nearby text.
- If the visual is decorative or empty, say so.
""".strip()


def analyze_visuals(
    run_dir: Path,
    vlm_model: str,
    ollama_url: str,
    visual_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    visual_items = read_json(run_dir / "visual_items.json", default=[])
    selected = visual_items
    if visual_ids:
        selected = [v for v in visual_items if v.get("visual_id") in set(visual_ids)]
    selected = [v for v in selected if v.get("human_status") in {"keep", "needs_review"}]

    summaries = []
    for item in selected:
        crop_path = item.get("crop_path")
        if not crop_path or not Path(crop_path).exists():
            summaries.append({
                "doc_id": item.get("doc_id"),
                "file_name": item.get("file_name"),
                "item_id": item.get("item_id"),
                "visual_id": item.get("visual_id"),
                "page_no": item.get("page_no"),
                "visual_type": "unknown",
                "summary": "No crop image was available for VLM analysis.",
                "visible_text": [],
                "importance": "unclear",
                "human_note": item.get("human_note", ""),
                "issues": ["Missing crop path."],
            })
            continue
        try:
            result = chat_vision_json(
                build_visual_prompt(item),
                image_path=Path(crop_path),
                model=vlm_model,
                ollama_url=ollama_url,
                temperature=0.1,
            )
            if not isinstance(result, dict):
                raise ValueError("VLM returned non-object JSON")
            summaries.append({
                "doc_id": item.get("doc_id"),
                "file_name": item.get("file_name"),
                "item_id": item.get("item_id"),
                "visual_id": item.get("visual_id"),
                "page_no": item.get("page_no"),
                "crop_path": crop_path,
                "human_note": item.get("human_note", ""),
                "visual_type": result.get("visual_type", "unknown"),
                "summary": result.get("summary", ""),
                "visible_text": result.get("visible_text", []),
                "importance": result.get("importance", "unclear"),
                "issues": result.get("issues", []),
            })
        except Exception as exc:
            summaries.append({
                "doc_id": item.get("doc_id"),
                "file_name": item.get("file_name"),
                "item_id": item.get("item_id"),
                "visual_id": item.get("visual_id"),
                "page_no": item.get("page_no"),
                "crop_path": crop_path,
                "human_note": item.get("human_note", ""),
                "visual_type": "unknown",
                "summary": "VLM analysis failed.",
                "visible_text": [],
                "importance": "unclear",
                "issues": [str(exc)],
            })
    write_json(run_dir / "image_summaries.json", summaries)
    return summaries
