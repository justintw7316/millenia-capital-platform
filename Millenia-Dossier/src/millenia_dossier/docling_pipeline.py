from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw

from .quick_skim import quick_skim_crop
from .utils import dataframe_to_records, safe_text, slugify, write_json, write_text


def _import_docling():
    try:
        from docling.document_converter import DocumentConverter
        return DocumentConverter
    except Exception as exc:
        raise RuntimeError(
            "Docling is not installed or failed to import. Install with: pip install docling"
        ) from exc


def _maybe_call(method, *args, **kwargs):
    try:
        return method(*args, **kwargs)
    except TypeError:
        try:
            return method()
        except Exception:
            return None
    except Exception:
        return None


def _get_label(item: Any) -> str:
    label = getattr(item, "label", "")
    if label:
        return str(label)
    return item.__class__.__name__


def get_item_type(item: Any) -> str:
    cls = item.__class__.__name__.lower()
    label = _get_label(item).lower()
    joined = f"{cls} {label}"
    if "table" in joined:
        return "table"
    if any(x in joined for x in ["picture", "image", "figure", "chart"]):
        return "picture"
    if "section" in joined or "header" in joined or "title" in joined:
        return "section_header"
    if "list" in joined:
        return "list_item"
    return "text"


def _get_text(item: Any) -> str:
    for attr in ["text", "orig", "caption"]:
        value = getattr(item, attr, None)
        if value:
            return safe_text(value)
    for method_name in ["export_to_markdown", "to_markdown"]:
        method = getattr(item, method_name, None)
        if callable(method):
            value = _maybe_call(method)
            if value:
                return safe_text(value)
    return ""


def _origin_to_string(origin: Any) -> str:
    if origin is None:
        return "TOPLEFT"
    value = getattr(origin, "value", None)
    if value is not None:
        return str(value).upper()
    name = getattr(origin, "name", None)
    if name is not None:
        return str(name).upper()
    return str(origin).upper()


def _bbox_to_dict(bbox_obj: Any) -> dict[str, Any] | None:
    if bbox_obj is None:
        return None

    candidates = []
    for obj in [bbox_obj, getattr(bbox_obj, "bbox", None)]:
        if obj is not None:
            candidates.append(obj)

    for obj in candidates:
        if all(hasattr(obj, name) for name in ["l", "t", "r", "b"]):
            return {
                "left": float(getattr(obj, "l")),
                "top": float(getattr(obj, "t")),
                "right": float(getattr(obj, "r")),
                "bottom": float(getattr(obj, "b")),
                "coord_origin": _origin_to_string(getattr(obj, "coord_origin", None)),
            }

        vals = {}
        for name in ["left", "top", "right", "bottom", "x0", "y0", "x1", "y1"]:
            if hasattr(obj, name):
                vals[name] = float(getattr(obj, name))
        if vals:
            left = vals.get("left", vals.get("x0"))
            top = vals.get("top", vals.get("y0"))
            right = vals.get("right", vals.get("x1"))
            bottom = vals.get("bottom", vals.get("y1"))
            if None not in [left, top, right, bottom]:
                return {
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "coord_origin": _origin_to_string(getattr(obj, "coord_origin", None)),
                }

    if isinstance(bbox_obj, (list, tuple)) and len(bbox_obj) >= 4:
        return {
            "left": float(bbox_obj[0]),
            "top": float(bbox_obj[1]),
            "right": float(bbox_obj[2]),
            "bottom": float(bbox_obj[3]),
            "coord_origin": "TOPLEFT",
        }
    return None


def _read_page_number(obj: Any) -> int | None:
    for page_attr in ["page_no", "page"]:
        if hasattr(obj, page_attr):
            try:
                return int(getattr(obj, page_attr))
            except Exception:
                pass
    if hasattr(obj, "page_index"):
        try:
            return int(getattr(obj, "page_index")) + 1
        except Exception:
            pass
    return None


def _get_page_and_bbox(item: Any) -> tuple[int | None, dict[str, Any] | None]:
    prov = getattr(item, "prov", None) or getattr(item, "provenance", None)
    if isinstance(prov, list) and prov:
        prov0 = prov[0]
    else:
        prov0 = prov
    page_no = None
    bbox = None
    if prov0 is not None:
        page_no = _read_page_number(prov0)
        bbox = _bbox_to_dict(getattr(prov0, "bbox", None))
    if page_no is None:
        page_no = _read_page_number(item)
    if bbox is None:
        bbox = _bbox_to_dict(getattr(item, "bbox", None))
    return page_no, bbox


def _normalize_bbox_for_pymupdf(bbox: dict[str, Any], page_width: float, page_height: float):
    left = float(bbox["left"])
    top = float(bbox["top"])
    right = float(bbox["right"])
    bottom = float(bbox["bottom"])
    origin = str(bbox.get("coord_origin", "TOPLEFT")).upper()

    if max(abs(left), abs(top), abs(right), abs(bottom)) <= 1.5:
        left *= page_width
        right *= page_width
        top *= page_height
        bottom *= page_height

    if "BOTTOM" in origin and "LEFT" in origin:
        x0 = left
        x1 = right
        y0 = page_height - top
        y1 = page_height - bottom
    else:
        x0 = left
        x1 = right
        y0 = top
        y1 = bottom

    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)

    x0 = max(0, min(page_width, x0))
    x1 = max(0, min(page_width, x1))
    y0 = max(0, min(page_height, y0))
    y1 = max(0, min(page_height, y1))

    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def crop_pdf_region(pdf_path: Path, bbox: dict[str, float], page_no: int | None, out_path: Path) -> str | None:
    if page_no is None or bbox is None:
        return None
    try:
        import fitz
        doc = fitz.open(pdf_path)
        page_index = max(0, min(len(doc)-1, int(page_no) - 1))
        page = doc[page_index]
        rect_vals = _normalize_bbox_for_pymupdf(bbox, page.rect.width, page.rect.height)
        if not rect_vals:
            return None
        rect = fitz.Rect(*rect_vals)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect, alpha=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(out_path)
        return str(out_path)
    except Exception:
        return None


def make_outlined_pdf(pdf_path: Path, layout_items: list[dict[str, Any]], out_path: Path) -> str | None:
    try:
        import fitz
        doc = fitz.open(pdf_path)
        colors = {
            "table": (0, 0.45, 1),
            "picture": (0.1, 0.75, 0.1),
            "section_header": (1, 0.35, 0),
            "list_item": (0.6, 0.2, 0.9),
            "text": (0.8, 0.8, 0.8),
        }
        for item in layout_items:
            bbox = item.get("bbox")
            page_no = item.get("page_no")
            if not bbox or not page_no:
                continue
            page_index = max(0, min(len(doc)-1, int(page_no) - 1))
            page = doc[page_index]
            rect_vals = _normalize_bbox_for_pymupdf(bbox, page.rect.width, page.rect.height)
            if not rect_vals:
                continue
            rect = fitz.Rect(*rect_vals)
            color = colors.get(item.get("item_type"), (0.8, 0.8, 0.8))
            page.draw_rect(rect, color=color, width=2.5)
            label = f"{item.get('item_type')}:{item.get('item_id')}"
            page.insert_text((rect.x0, max(8, rect.y0 - 4)), label, fontsize=10, color=color)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(out_path)
        return str(out_path)
    except Exception:
        return None


def _export_doc_markdown(doc: Any) -> str:
    method = getattr(doc, "export_to_markdown", None)
    if callable(method):
        value = _maybe_call(method)
        if value:
            return str(value)
    return ""


def _export_doc_json(doc: Any) -> Any:
    for method_name in ["export_to_dict", "model_dump", "dict"]:
        method = getattr(doc, method_name, None)
        if callable(method):
            value = _maybe_call(method)
            if value is not None:
                return value
    try:
        return json.loads(doc.model_dump_json())
    except Exception:
        return {"warning": "Could not serialize DoclingDocument with this Docling version."}


def _export_table(item: Any, doc: Any, table_id: str, out_dir: Path) -> dict[str, Any]:
    markdown = ""
    html = ""
    csv_path = None
    columns: list[str] = []
    rows: list[dict[str, str]] = []

    df = None
    method = getattr(item, "export_to_dataframe", None)
    if callable(method):
        df = _maybe_call(method, doc=doc)
        if df is None:
            df = _maybe_call(method)
    if df is not None:
        try:
            columns = [str(c) for c in df.columns]
            rows = dataframe_to_records(df)
            markdown = df.to_markdown(index=False)
            csv_path = str(out_dir / f"{table_id}_raw.csv")
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
        except Exception:
            pass

    for method_name in ["export_to_markdown", "to_markdown"]:
        method = getattr(item, method_name, None)
        if callable(method) and not markdown:
            value = _maybe_call(method, doc=doc)
            if value is None:
                value = _maybe_call(method)
            if value:
                markdown = str(value)

    method = getattr(item, "export_to_html", None)
    if callable(method):
        value = _maybe_call(method, doc=doc)
        if value is None:
            value = _maybe_call(method)
        if value:
            html = str(value)
            html_path = out_dir / f"{table_id}_raw.html"
            write_text(html_path, html)

    md_path = out_dir / f"{table_id}_raw.md"
    write_text(md_path, markdown or safe_text(item))

    return {
        "raw_markdown_path": str(md_path),
        "raw_csv_path": csv_path,
        "raw_html_path": str(out_dir / f"{table_id}_raw.html") if html else None,
        "raw_markdown": markdown,
        "raw_html": html,
        "columns": columns,
        "rows": rows,
    }


def run_docling_on_files(input_files: list[Path], run_dir: Path) -> dict[str, Any]:
    DocumentConverter = _import_docling()
    converter = DocumentConverter()

    parsed_dir = run_dir / "parsed"
    crops_dir = run_dir / "crops"
    raw_tables_dir = run_dir / "raw_tables"
    outlined_dir = run_dir / "outlined_pdfs"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    documents = []
    all_layout_items = []
    all_raw_tables = []
    all_visual_items = []

    for doc_index, pdf_path in enumerate(input_files, start=1):
        doc_id = f"doc_{doc_index:03d}"
        safe_name = slugify(pdf_path.stem, fallback=doc_id)
        result = converter.convert(pdf_path)
        doc = result.document

        raw_md = _export_doc_markdown(doc)
        raw_json = _export_doc_json(doc)
        doc_out_dir = parsed_dir / doc_id
        doc_out_dir.mkdir(parents=True, exist_ok=True)
        raw_md_path = doc_out_dir / "raw_docling.md"
        raw_json_path = doc_out_dir / "raw_docling.json"
        write_text(raw_md_path, raw_md)
        write_json(raw_json_path, raw_json)

        layout_items: list[dict[str, Any]] = []
        raw_tables: list[dict[str, Any]] = []
        visual_items: list[dict[str, Any]] = []

        if hasattr(doc, "iterate_items"):
            iterator = doc.iterate_items()
        else:
            iterator = []
            for obj in getattr(doc, "texts", []) or []:
                iterator.append((obj, 0))
            for obj in getattr(doc, "tables", []) or []:
                iterator.append((obj, 0))
            for obj in getattr(doc, "pictures", []) or []:
                iterator.append((obj, 0))

        for item_index, pair in enumerate(iterator, start=1):
            try:
                item, level = pair
            except Exception:
                item, level = pair, 0
            item_type = get_item_type(item)
            page_no, bbox = _get_page_and_bbox(item)
            item_id = f"{doc_id}_item_{item_index:04d}"
            text = _get_text(item)
            crop_path = None
            if item_type in {"table", "picture"} and bbox:
                crop_path = crop_pdf_region(
                    pdf_path,
                    bbox,
                    page_no,
                    crops_dir / doc_id / f"{item_id}_{item_type}.png",
                )
            quick_skim = quick_skim_crop(crop_path, item_type=item_type) if item_type in {"table", "picture"} else None
            human_status = "keep"
            if quick_skim and quick_skim.get("label") in {"likely_empty", "likely_decorative"} and item_type == "picture":
                human_status = "needs_review"
            if item_type == "table":
                human_status = "needs_correction"

            layout_item = {
                "doc_id": doc_id,
                "file_name": pdf_path.name,
                "source_path": str(pdf_path),
                "item_id": item_id,
                "reading_order_index": item_index,
                "level": level,
                "item_type": item_type,
                "label": _get_label(item),
                "page_no": page_no,
                "bbox": bbox,
                "text": text,
                "crop_path": crop_path,
                "quick_skim": quick_skim,
                "human_status": human_status,
                "human_note": "",
            }
            layout_items.append(layout_item)

            if item_type == "table":
                table_id = f"{doc_id}_table_{len(raw_tables)+1:03d}"
                table_exports = _export_table(item, doc, table_id, raw_tables_dir / doc_id)
                raw_table = {
                    "doc_id": doc_id,
                    "file_name": pdf_path.name,
                    "item_id": item_id,
                    "table_id": table_id,
                    "page_no": page_no,
                    "bbox": bbox,
                    "crop_path": crop_path,
                    **table_exports,
                }
                raw_tables.append(raw_table)
            elif item_type == "picture":
                visual_id = f"{doc_id}_visual_{len(visual_items)+1:03d}"
                visual_items.append({
                    "doc_id": doc_id,
                    "file_name": pdf_path.name,
                    "item_id": item_id,
                    "visual_id": visual_id,
                    "page_no": page_no,
                    "bbox": bbox,
                    "crop_path": crop_path,
                    "quick_skim": quick_skim,
                    "human_status": human_status,
                    "human_note": "",
                    "nearby_text": "",
                })

        by_index = {x["item_id"]: i for i, x in enumerate(layout_items)}
        for visual in visual_items:
            idx = by_index.get(visual["item_id"])
            if idx is not None:
                neighbors = []
                for j in range(max(0, idx-3), min(len(layout_items), idx+4)):
                    if layout_items[j]["item_type"] in {"text", "section_header", "list_item"}:
                        neighbors.append(layout_items[j].get("text", ""))
                visual["nearby_text"] = "\n".join([t for t in neighbors if t])[:3000]

        outlined_path = make_outlined_pdf(pdf_path, layout_items, outlined_dir / f"{safe_name}_outlined.pdf")

        write_json(doc_out_dir / "layout_items.json", layout_items)
        write_json(doc_out_dir / "raw_tables.json", raw_tables)
        write_json(doc_out_dir / "visual_items.json", visual_items)

        documents.append({
            "doc_id": doc_id,
            "file_name": pdf_path.name,
            "source_path": str(pdf_path),
            "raw_docling_md_path": str(raw_md_path),
            "raw_docling_json_path": str(raw_json_path),
            "outlined_pdf_path": outlined_path,
            "layout_items_path": str(doc_out_dir / "layout_items.json"),
            "raw_tables_path": str(doc_out_dir / "raw_tables.json"),
            "visual_items_path": str(doc_out_dir / "visual_items.json"),
        })
        all_layout_items.extend(layout_items)
        all_raw_tables.extend(raw_tables)
        all_visual_items.extend(visual_items)

    manifest = {
        "documents": documents,
        "layout_items_path": str(run_dir / "layout_items.json"),
        "raw_tables_path": str(run_dir / "raw_tables.json"),
        "visual_items_path": str(run_dir / "visual_items.json"),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    write_json(run_dir / "layout_items.json", all_layout_items)
    write_json(run_dir / "raw_tables.json", all_raw_tables)
    write_json(run_dir / "visual_items.json", all_visual_items)
    return manifest
