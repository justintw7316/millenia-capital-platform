from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".csv",
    ".txt", ".md", ".html", ".htm", ".png", ".jpg",
    ".jpeg", ".webp", ".tif", ".tiff",
}

from millenia_dossier.config import get_config
from millenia_dossier.docling_pipeline import run_docling_on_files
from millenia_dossier.excel_exporter import create_excel_export
from millenia_dossier.extraction import run_extraction, run_per_document_extraction, run_merge_dossier, FIELD_NAMES, FIELD_DEFINITIONS, _split_documents, _build_doc_prompt, _build_merge_prompt
from millenia_dossier.feed_builder import build_llm_feed
from millenia_dossier.ollama_client import chat_text_stream, chat_vision_stream, OllamaError
from millenia_dossier.table_corrector import correct_tables, build_table_prompt
from millenia_dossier.utils import now_run_id, read_json, write_json, safe_text, extract_json_from_text
from millenia_dossier.visual_analyzer import analyze_visuals, update_visual_notes, build_visual_prompt

st.set_page_config(page_title="Millenia Dossier — Company Intelligence", layout="wide")


def render_pdf(path: Path, height: int = 700) -> None:
    if not path.exists():
        st.warning("PDF not found.")
        return
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    st.markdown(
        f'<embed src="data:application/pdf;base64,{b64}" width="100%" height="{height}" type="application/pdf">',
        unsafe_allow_html=True,
    )


def show_dataframe(data: Any, empty_message: str = "No data yet.") -> None:
    if not data:
        st.info(empty_message)
        return
    try:
        st.dataframe(pd.DataFrame(data), use_container_width=True)
    except Exception:
        st.json(data)


def init_state():
    if "run_dir" not in st.session_state:
        st.session_state.run_dir = None
    if "uploaded_paths" not in st.session_state:
        st.session_state.uploaded_paths = []


init_state()
config = get_config()

st.title("Millenia Dossier")
st.caption("Docling → LLM table correction → VLM visual review → 56-field company dossier with evidence")

PHASE_LABELS = ["Upload", "Docling", "Review", "Tables", "VLM", "Feed", "Extract", "Excel"]

def phase_status(run_path: Path) -> list[bool]:
    p = Path(run_path)
    return [
        any((p / "input").iterdir()) if (p / "input").exists() else False,
        (p / "run_manifest.json").exists(),
        (p / "layout_items.json").exists(),
        (p / "cleaned_tables.json").exists(),
        (p / "image_summaries.json").exists(),
        (p / "file_llm_feed.md").exists(),
        (p / "company_dossier_merged.json").exists(),
        (p / "millenia_dossier_export.xlsx").exists(),
    ]

def phase_count(run_path: Path) -> tuple[int, int]:
    statuses = phase_status(run_path)
    return sum(statuses), len(statuses)

with st.sidebar:
    runs_dir_raw = st.text_input("Runs Directory", value=str(config.runs_dir))
    runs_dir = Path(runs_dir_raw)

    avail_runs = sorted(runs_dir.glob("millenia_*"), reverse=True) if runs_dir.exists() else []
    cur_run_name = Path(st.session_state.run_dir).name if st.session_state.run_dir else None
    default_idx = 0
    for i, r in enumerate(avail_runs):
        if r.name == cur_run_name:
            default_idx = i
            break

    run_labels = []
    for r in avail_runs:
        d, t = phase_count(r)
        run_labels.append(f"{r.name[:25]} — {d}/{t}")

    st.markdown("### Select Run")
    if run_labels:
        chosen = st.selectbox("Run", run_labels, index=default_idx, label_visibility="collapsed")
        idx = run_labels.index(chosen)
        chosen_path = avail_runs[idx]
        if str(chosen_path) != st.session_state.run_dir:
            st.session_state.run_dir = str(chosen_path)
            input_dir = chosen_path / "input"
            st.session_state.uploaded_paths = [str(f) for f in sorted(input_dir.iterdir())] if input_dir.exists() else []
            st.rerun()

        sel_statuses = phase_status(chosen_path)
        cols = st.columns(len(PHASE_LABELS))
        for ci, (label, ok) in enumerate(zip(PHASE_LABELS, sel_statuses)):
            with cols[ci]:
                st.markdown(f"{'✅' if ok else '⬜'}")
                st.caption(label)
    else:
        st.info("No runs yet. Create one in the Upload tab.")

    st.divider()
    st.header("Settings")
    llm_model = st.text_input("LLM Model", value=config.llm_model)
    vlm_model = st.text_input("VLM Model", value=config.vlm_model)
    ollama_url = st.text_input("Ollama URL", value=config.ollama_url)
    st.divider()
    st.write("Optional switches")
    use_llm_table_correction = st.toggle("Use LLM table correction", value=True)
    use_vlm_visual_review = st.toggle("Use VLM visual review", value=True)

run_dir = Path(st.session_state.run_dir) if st.session_state.run_dir else None

tabs = st.tabs([
    "1) Upload",
    "2) Run Docling",
    "3) Item Review + Quick Skim",
    "4) LLM Table Correction",
    "5) Image / Chart VLM Review",
    "6) Build LLM Feed",
    "7) 56-Field Extraction",
    "8) Excel Preview + Download",
])

with tabs[0]:
    st.subheader("1) Upload Files")
    st.write("Upload files individually or scan an entire folder recursively.")

    uploaded = st.file_uploader("Upload files individually", type=sorted(SUPPORTED_EXTENSIONS), accept_multiple_files=True)

    if "browse_active" not in st.session_state:
        st.session_state.browse_active = False
    if "browse_path" not in st.session_state:
        st.session_state.browse_path = ""
    if "browse_selected" not in st.session_state:
        st.session_state.browse_selected = ""

    bcol1, bcol2, bcol3 = st.columns([2, 1, 1])
    with bcol1:
        folder_path = st.text_input(
            "Or scan a folder recursively",
            value=st.session_state.browse_selected or "",
            placeholder="C:\\path\\to\\documents",
            key="folder_path_input",
        )
    with bcol2:
        st.markdown("###")
        if st.button("Browse…", use_container_width=True):
            st.session_state.browse_active = not st.session_state.browse_active
            if st.session_state.browse_active and not st.session_state.browse_path:
                import string as _s
                _drives = [f"{_d}:\\" for _d in _s.ascii_uppercase if Path(f"{_d}:\\").exists()]
                st.session_state.browse_path = _drives[0] if _drives else "C:\\"
    with bcol3:
        st.markdown("###")
        if st.button("Clear", use_container_width=True, disabled=not st.session_state.browse_selected):
            st.session_state.browse_selected = ""
            st.session_state.browse_path = ""
            st.rerun()

    if st.session_state.browse_active:
        st.markdown("---")
        st.markdown("**Browse filesystem**")
        _cur = Path(st.session_state.browse_path)
        ncol1, ncol2, ncol3 = st.columns([1, 5, 1])
        with ncol1:
            if _cur.parent != _cur:
                if st.button("⬆ Up"):
                    st.session_state.browse_path = str(_cur.parent)
                    st.rerun()
        with ncol2:
            st.caption(f"📁 {_cur}")
        with ncol3:
            if st.button("Select"):
                st.session_state.browse_selected = str(_cur)
                st.session_state["folder_path_input"] = str(_cur)
                st.session_state.browse_active = False
                st.rerun()
        try:
            _subdirs = sorted([d for d in _cur.iterdir() if d.is_dir()])
            if _subdirs:
                _chosen = st.selectbox(
                    "Subdirectories",
                    [""] + [str(d.name) for d in _subdirs],
                    format_func=lambda x: x if x else "— choose directory —",
                )
                if _chosen:
                    st.session_state.browse_path = str(_cur / _chosen)
                    st.rerun()
            else:
                st.info("Empty directory")
        except PermissionError:
            st.warning("Access denied")
        except OSError as _e:
            st.warning(str(_e))
        st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Create new run", type="primary"):
            new_run = runs_dir / now_run_id("millenia")
            input_dir = new_run / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            saved = []

            src_path = folder_path.strip() or st.session_state.browse_selected
            if src_path:
                root = Path(src_path)
                if not root.exists():
                    st.error(f"Folder not found: {root}")
                else:
                    for f in sorted(root.rglob("*")):
                        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                            out = input_dir / f.name
                            out.write_bytes(f.read_bytes())
                            saved.append(str(f.resolve()))
                    if saved:
                        st.success(f"Found {len(saved)} file(s) from folder scan.")
            else:
                for f in uploaded or []:
                    out = input_dir / f.name
                    out.write_bytes(f.getbuffer())
                    saved.append(str(out))
                if saved:
                    st.success(f"Uploaded {len(saved)} file(s).")

            st.session_state.run_dir = str(new_run)
            st.session_state.uploaded_paths = saved
            st.success(f"Created run: {new_run}")
    with col2:
        st.write("Current run:")
        st.code(str(st.session_state.run_dir or "No run yet"))
        sources = st.session_state.uploaded_paths
        if sources:
            st.write(f"Source files: {len(sources)}")
            if len(sources) <= 10:
                for s in sources:
                    st.code(Path(s).name)

    if uploaded or folder_path.strip():
        st.markdown("### Uploaded file preview")
        sources = st.session_state.uploaded_paths or []
        if sources:
            first = Path(sources[0])
            if first.suffix.lower() in {".pdf"}:
                pdf_b64 = base64.b64encode(first.read_bytes()).decode("utf-8")
                st.markdown(
                    f'<embed src="data:application/pdf;base64,{pdf_b64}" width="100%" height="500" type="application/pdf">',
                    unsafe_allow_html=True,
                )
            else:
                st.write(f"Preview not available for {first.suffix} files. {len(sources)} file(s) loaded.")

with tabs[1]:
    st.subheader("2) Run Docling")
    st.write("Parses PDFs with Docling, producing layout items, tables, visuals, and outlined PDFs.")
    run_dir = Path(st.session_state.run_dir) if st.session_state.run_dir else None
    if not run_dir:
        st.warning("Create a run first.")
    else:
        input_paths = [Path(p) for p in st.session_state.uploaded_paths]
        if not input_paths:
            st.warning("No uploaded PDF paths found in this run.")
        else:
            st.write("Input files:")
            for p in input_paths:
                st.code(str(p))
            if st.button("Run Docling parser", type="primary"):
                with st.spinner("Running Docling..."):
                    try:
                        manifest = run_docling_on_files(input_paths, run_dir)
                        st.success("Docling parsing completed.")
                        st.json(manifest)
                    except Exception as exc:
                        st.error(str(exc))

        manifest = read_json(run_dir / "run_manifest.json", default=None)
        if manifest:
            st.markdown("### Outlined PDF Preview")
            docs = manifest.get("documents", [])
            labels = [d.get("file_name", d.get("doc_id")) for d in docs]
            if docs:
                choice = st.selectbox("Choose outlined PDF", labels)
                idx = labels.index(choice)
                outlined = docs[idx].get("outlined_pdf_path")
                if outlined:
                    render_pdf(Path(outlined), height=650)
                else:
                    st.info("Outlined PDF was not created.")

with tabs[2]:
    st.subheader("3) DoclingDocument Item Review + Quick Skim")
    run_dir = Path(st.session_state.run_dir) if st.session_state.run_dir else None
    if not run_dir or not (run_dir / "layout_items.json").exists():
        st.warning("Run Docling first.")
    else:
        layout_items = read_json(run_dir / "layout_items.json", default=[])
        st.write(f"Detected items: {len(layout_items)}")
        doc_pairs = sorted({(i.get("doc_id", ""), i.get("file_name", "")) for i in layout_items if i.get("doc_id")})
        doc_labels = [f"{d[0]}-{d[1]}" for d in doc_pairs]
        c1, c2, c3 = st.columns(3)
        with c1:
            type_filter = st.multiselect("Item types", sorted(set(i.get("item_type") for i in layout_items)), default=[])
        with c2:
            skim_filter = st.multiselect("Quick skim", sorted(set((i.get("quick_skim") or {}).get("label") for i in layout_items if i.get("quick_skim"))), default=[])
        with c3:
            page_filter = st.text_input("Page contains", value="")
        doc_filter = st.multiselect("Document", doc_labels, default=[])
        filtered = layout_items
        if doc_filter:
            selected_doc_ids = {lbl.split("-", 1)[0] for lbl in doc_filter}
            filtered = [i for i in filtered if i.get("doc_id") in selected_doc_ids]
        if type_filter:
            filtered = [i for i in filtered if i.get("item_type") in type_filter]
        if skim_filter:
            filtered = [i for i in filtered if (i.get("quick_skim") or {}).get("label") in skim_filter]
        if page_filter.strip():
            filtered = [i for i in filtered if str(i.get("page_no")) == page_filter.strip()]

        updated = False
        for item in filtered[:100]:
            with st.expander(f"{item.get('reading_order_index')} | {item.get('item_type')} | {item.get('item_id')} | page {item.get('page_no')}"):
                cols = st.columns([1, 2])
                with cols[0]:
                    crop = item.get("crop_path")
                    if crop and Path(crop).exists():
                        st.image(crop, caption="Crop preview", use_container_width=True)
                    skim = item.get("quick_skim")
                    if skim:
                        st.write("Quick skim:", skim.get("label"))
                        st.caption(skim.get("reason", ""))
                    status_options = ["keep", "discard", "needs_review"]
                    if item.get("item_type") == "table":
                        status_options = ["keep", "discard", "needs_correction", "corrected"]
                    new_status = st.selectbox(
                        "Human status",
                        status_options,
                        index=status_options.index(item.get("human_status")) if item.get("human_status") in status_options else 0,
                        key=f"status_{item.get('item_id')}",
                    )
                    note = st.text_area("Human note", value=item.get("human_note", ""), key=f"note_{item.get('item_id')}")
                    if new_status != item.get("human_status") or note != item.get("human_note", ""):
                        item["human_status"] = new_status
                        item["human_note"] = note
                        updated = True
                with cols[1]:
                    st.write("Text/raw preview")
                    st.text_area("", value=item.get("text", "")[:4000], height=200, key=f"text_{item.get('item_id')}", disabled=True)
                    st.json({k: item.get(k) for k in ["doc_id", "file_name", "page_no", "bbox", "label"]})
        if updated:
            if st.button("Save item review updates"):
                write_json(run_dir / "layout_items.json", layout_items)
                visual_items = read_json(run_dir / "visual_items.json", default=[])
                by_item = {i.get("item_id"): i for i in layout_items}
                for v in visual_items:
                    src = by_item.get(v.get("item_id"))
                    if src:
                        v["human_status"] = src.get("human_status", v.get("human_status"))
                        v["human_note"] = src.get("human_note", v.get("human_note", ""))
                write_json(run_dir / "visual_items.json", visual_items)
                st.success("Saved review updates.")

with tabs[3]:
    st.subheader("4) LLM Table Correction")
    run_dir = Path(st.session_state.run_dir) if st.session_state.run_dir else None
    if not run_dir or not (run_dir / "raw_tables.json").exists():
        st.warning("Run Docling first.")
    else:
        raw_tables = read_json(run_dir / "raw_tables.json", default=[])
        st.write(f"Raw tables found: {len(raw_tables)}")
        selected_tables = st.multiselect("Tables to correct", [t.get("table_id") for t in raw_tables], default=[t.get("table_id") for t in raw_tables])
        for table in raw_tables:
            with st.expander(f"{table.get('table_id')} | {table.get('file_name')} | page {table.get('page_no')}"):
                cols = st.columns([1, 2])
                with cols[0]:
                    crop = table.get("crop_path")
                    if crop and Path(crop).exists():
                        st.image(crop, caption="Table crop preview", use_container_width=True)
                with cols[1]:
                    st.markdown("Raw Docling table preview")
                    st.text_area("Raw table markdown", value=table.get("raw_markdown", "")[:6000], height=220, key=f"rawtable_{table.get('table_id')}")
        if st.button("Run table correction", type="primary"):
            if not use_llm_table_correction:
                cleaned = correct_tables(run_dir, llm_model=llm_model, ollama_url=ollama_url, table_ids=selected_tables, use_llm=False)
                st.success(f"Created cleaned_tables.json with {len(cleaned)} table(s).")
            else:
                raw_tables_data = read_json(run_dir / "raw_tables.json", default=[])
                selected_raw = [t for t in raw_tables_data if not selected_tables or t.get("table_id") in set(selected_tables)]
                cleaned = []
                stream_box = st.empty()
                prog = st.empty()
                for idx, raw in enumerate(selected_raw, 1):
                    prog.caption(f"Correcting table {idx}/{len(selected_raw)} — {raw.get('table_id')}")
                    prompt = build_table_prompt(raw)
                    full = []
                    try:
                        for token in chat_text_stream(prompt, model=llm_model, ollama_url=ollama_url, temperature=0.05):
                            full.append(token)
                            stream_box.markdown("".join(full) + " ▌")
                        result = extract_json_from_text("".join(full))
                    except OllamaError:
                        result = {}
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
                        fallback = {
                            "columns": raw.get("columns") or [],
                            "rows": raw.get("rows") or [],
                        }
                        cleaned_table["columns"] = fallback["columns"]
                        cleaned_table["rows"] = fallback["rows"]
                        cleaned_table["issues"].append("Fallback used.")
                    cleaned.append(cleaned_table)
                write_json(run_dir / "cleaned_tables.json", cleaned)
                stream_box.empty()
                st.success(f"Created cleaned_tables.json with {len(cleaned)} table(s).")
        cleaned_tables = read_json(run_dir / "cleaned_tables.json", default=[])
        if cleaned_tables:
            st.markdown("### Cleaned table preview")
            for table in cleaned_tables:
                st.write(f"**{table.get('table_id')}** — {table.get('caption','')}")
                show_dataframe(table.get("rows", []), empty_message="No rows in this cleaned table.")
                if table.get("issues"):
                    st.caption("Issues: " + "; ".join(map(str, table.get("issues"))))

with tabs[4]:
    st.subheader("5) Image / Chart VLM Review")
    st.write("Add optional human context notes before sending visual crops to the VLM.")
    run_dir = Path(st.session_state.run_dir) if st.session_state.run_dir else None
    if not run_dir or not (run_dir / "visual_items.json").exists():
        st.warning("Run Docling first.")
    else:
        visual_items = read_json(run_dir / "visual_items.json", default=[])
        st.write(f"Visual items found: {len(visual_items)}")
        updates = {}
        selected_visuals = []
        for item in visual_items:
            with st.expander(f"{item.get('visual_id')} | {item.get('file_name')} | page {item.get('page_no')}"):
                cols = st.columns([1, 2])
                with cols[0]:
                    crop = item.get("crop_path")
                    if crop and Path(crop).exists():
                        st.image(crop, caption="Visual crop", use_container_width=True)
                    st.write("Quick skim:", (item.get("quick_skim") or {}).get("label", "none"))
                    status = st.selectbox(
                        "Status",
                        ["keep", "discard", "needs_review"],
                        index=["keep", "discard", "needs_review"].index(item.get("human_status")) if item.get("human_status") in ["keep", "discard", "needs_review"] else 0,
                        key=f"visual_status_{item.get('visual_id')}",
                    )
                    run_this = st.checkbox("Run VLM on this visual", value=status in {"keep", "needs_review"}, key=f"run_vlm_{item.get('visual_id')}")
                    if run_this:
                        selected_visuals.append(item.get("visual_id"))
                with cols[1]:
                    note = st.text_area("Reviewer note for VLM context", value=item.get("human_note", ""), key=f"visual_note_{item.get('visual_id')}")
                    st.text_area("Nearby text", value=item.get("nearby_text", "")[:3000], height=160, disabled=True, key=f"nearby_{item.get('visual_id')}")
                    updates[item.get("visual_id")] = {"human_status": status, "human_note": note}
        if st.button("Save visual notes/statuses"):
            update_visual_notes(run_dir, updates)
            st.success("Saved visual notes/statuses.")
        if st.button("Run VLM on selected visuals", type="primary", disabled=not use_vlm_visual_review):
            update_visual_notes(run_dir, updates)
            visual_items_data = read_json(run_dir / "visual_items.json", default=[])
            selected = [v for v in visual_items_data if selected_visuals and v.get("visual_id") in set(selected_visuals)]
            selected = [v for v in selected if v.get("human_status") in {"keep", "needs_review"}]
            summaries = []
            stream_box = st.empty()
            prog = st.empty()
            for idx, item in enumerate(selected, 1):
                crop_path = item.get("crop_path")
                if not crop_path or not Path(crop_path).exists():
                    summaries.append({
                        "doc_id": item.get("doc_id"), "file_name": item.get("file_name"),
                        "item_id": item.get("item_id"), "visual_id": item.get("visual_id"),
                        "page_no": item.get("page_no"), "visual_type": "unknown",
                        "summary": "No crop image was available for VLM analysis.",
                        "visible_text": [], "importance": "unclear",
                        "human_note": item.get("human_note", ""), "issues": ["Missing crop path."],
                    })
                    continue
                prog.caption(f"Analyzing visual {idx}/{len(selected)} — {item.get('visual_id')}")
                prompt = build_visual_prompt(item)
                full = []
                try:
                    for token in chat_vision_stream(prompt, image_path=Path(crop_path), model=vlm_model, ollama_url=ollama_url):
                        full.append(token)
                        stream_box.markdown("".join(full) + " ▌")
                    result = extract_json_from_text("".join(full))
                    if not isinstance(result, dict):
                        raise ValueError("VLM returned non-object JSON")
                except Exception as exc:
                    result = {"visual_type": "unknown", "summary": "VLM analysis failed.", "visible_text": [], "importance": "unclear", "issues": [str(exc)]}
                summaries.append({
                    "doc_id": item.get("doc_id"), "file_name": item.get("file_name"),
                    "item_id": item.get("item_id"), "visual_id": item.get("visual_id"),
                    "page_no": item.get("page_no"), "crop_path": crop_path,
                    "human_note": item.get("human_note", ""),
                    "visual_type": result.get("visual_type", "unknown"),
                    "summary": result.get("summary", ""),
                    "visible_text": result.get("visible_text", []),
                    "importance": result.get("importance", "unclear"),
                    "issues": result.get("issues", []),
                })
            write_json(run_dir / "image_summaries.json", summaries)
            stream_box.empty()
            st.success(f"Created image_summaries.json with {len(summaries)} summaries.")
        summaries = read_json(run_dir / "image_summaries.json", default=[])
        if summaries:
            st.markdown("### Image summary preview")
            show_dataframe(summaries)

with tabs[5]:
    st.subheader("6) Build LLM Feed")
    run_dir = Path(st.session_state.run_dir) if st.session_state.run_dir else None
    if not run_dir or not (run_dir / "layout_items.json").exists():
        st.warning("Run Docling first.")
    else:
        if st.button("Build file_llm_feed.md", type="primary"):
            feed = build_llm_feed(run_dir)
            st.success("Built file_llm_feed.md")
            st.text_area("Feed preview", value=feed[:15000], height=500)
        feed_path = run_dir / "file_llm_feed.md"
        if feed_path.exists():
            feed_text = feed_path.read_text(encoding="utf-8")
            st.download_button("Download file_llm_feed.md", data=feed_text, file_name="file_llm_feed.md")
            st.text_area("Current feed", value=feed_text[:15000], height=500)

with tabs[6]:
    st.subheader("7) 56-Field Company Dossier Extraction")
    st.write("Two-phase extraction: per-document JSONs → merged dossier with evidence tracking.")
    run_dir = Path(st.session_state.run_dir) if st.session_state.run_dir else None
    if not run_dir or not (run_dir / "file_llm_feed.md").exists():
        st.warning("Build the LLM feed first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Step 1: Extract per-document JSONs", type="primary"):
                feed_path = run_dir / "file_llm_feed.md"
                if not feed_path.exists():
                    st.error(f"LLM feed not found: {feed_path}")
                else:
                    feed_text = feed_path.read_text(encoding="utf-8")
                    docs = _split_documents(feed_text)
                    json_dir = run_dir / "json_for_each_file"
                    json_dir.mkdir(parents=True, exist_ok=True)
                    manifest = read_json(run_dir / "run_manifest.json", default={})
                    doc_manifest = manifest.get("documents", [])
                    per_doc_results = []
                    stream_box = st.empty()
                    prog = st.empty()
                    for i, doc in enumerate(docs):
                        doc_id = f"doc_{i+1:03d}"
                        file_name = doc["doc_name"]
                        source_path = ""
                        if i < len(doc_manifest):
                            file_name = doc_manifest[i].get("file_name", doc["doc_name"])
                            source_path = doc_manifest[i].get("source_path", "")
                        prog.caption(f"Extracting document {i+1}/{len(docs)} — {file_name}")
                        prompt = _build_doc_prompt(doc)
                        full = []
                        try:
                            for token in chat_text_stream(prompt, model=llm_model, ollama_url=ollama_url, temperature=0.1, timeout=300):
                                full.append(token)
                                stream_box.markdown("".join(full) + " ▌")
                            result = extract_json_from_text("".join(full))
                            if not isinstance(result, dict):
                                raise ValueError("LLM returned non-object JSON")
                        except Exception as exc:
                            result = {
                                "document_category": "Unknown",
                                "short_summary": f"Extraction failed: {exc}",
                                "long_summary": "",
                                "fields": {},
                                "key_people": [],
                            }
                        doc_output = {
                            "doc_id": doc_id,
                            "file_name": file_name,
                            "source_path": source_path,
                            "document_category": result.get("document_category", "Unknown"),
                            "short_summary": result.get("short_summary", ""),
                            "long_summary": result.get("long_summary", ""),
                            "doc_fields": {},
                            "key_people": result.get("key_people", []),
                            "source_markdown_file": str(feed_path),
                        }
                        raw_fields = result.get("fields", {})
                        for fname in FIELD_NAMES:
                            fdata = raw_fields.get(fname, {})
                            if not isinstance(fdata, dict):
                                fdata = {"value": fdata, "answer": "", "evidence_quote": ""}
                            value = fdata.get("value")
                            answer = fdata.get("answer", "")
                            evidence_quote = fdata.get("evidence_quote", "")
                            doc_output["doc_fields"][fname] = {
                                "value": value,
                                "answer": answer,
                                "evidence": [
                                    {"doc_id": doc_id, "file_name": file_name, "quote": evidence_quote or "", "page_start": None, "page_end": None}
                                ] if evidence_quote else [],
                            }
                        doc_json_path = json_dir / f"{doc_id}.json"
                        write_json(doc_json_path, doc_output)
                        per_doc_results.append(doc_output)
                    stream_box.empty()
                    st.success(f"Extracted {len(per_doc_results)} document(s). JSONs saved to json_for_each_file/")
        with col2:
            step1_done = (run_dir / "json_for_each_file").exists() and any((run_dir / "json_for_each_file").iterdir())
            if st.button("Step 2: Merge dossier", type="primary", disabled=not step1_done):
                json_dir = run_dir / "json_for_each_file"
                if not json_dir.exists():
                    st.error(f"No per-document JSONs found at {json_dir}")
                else:
                    per_doc_fields = []
                    for path in sorted(json_dir.glob("*.json")):
                        data = read_json(path, default={})
                        if isinstance(data, dict) and data.get("doc_fields"):
                            per_doc_fields.append(data)
                    if not per_doc_fields:
                        st.error("No per-document extractions found to merge.")
                    else:
                        merge_prompt = _build_merge_prompt(per_doc_fields, llm_model)
                        stream_box = st.empty()
                        full = []
                        try:
                            for token in chat_text_stream(merge_prompt, model=llm_model, ollama_url=ollama_url, temperature=0.1, timeout=600):
                                full.append(token)
                                stream_box.markdown("".join(full) + " ▌")
                            merged = extract_json_from_text("".join(full))
                            if not isinstance(merged, dict):
                                raise ValueError("Merge returned non-object JSON")
                        except Exception as exc:
                            merged = {"short_summary": f"Merge failed: {exc}", "long_summary": "", "final_fields": {}}
                        final_fields = merged.get("final_fields", {})
                        for fname in FIELD_NAMES:
                            if fname not in final_fields:
                                final_fields[fname] = {"value": None, "answer": "Not found in any document.", "evidence": []}
                        documents = []
                        for doc in per_doc_fields:
                            documents.append({
                                "doc_id": doc.get("doc_id"),
                                "file_name": doc.get("file_name"),
                                "document_category": doc.get("document_category"),
                                "short_summary": doc.get("short_summary"),
                                "long_summary": doc.get("long_summary"),
                                "source_markdown_file": doc.get("source_markdown_file"),
                                "source_json_file": str(json_dir / f"{doc.get('doc_id')}.json"),
                            })
                        dossier = {
                            "short_summary": merged.get("short_summary", ""),
                            "long_summary": merged.get("long_summary", ""),
                            "final_fields": final_fields,
                            "documents": documents,
                            "model_used": llm_model,
                            "provider_used": "ollama",
                            "source_document_count": len(per_doc_fields),
                            "final_fields_count": len([f for f in final_fields.values() if f.get("value") is not None]),
                        }
                        write_json(run_dir / "company_dossier_merged.json", dossier)
                        stream_box.empty()
                        st.success(f"Created company_dossier_merged.json with {dossier['final_fields_count']} populated fields.")

        dossier = read_json(run_dir / "company_dossier_merged.json", default=None)
        if dossier:
            st.markdown("### Company Dossier Preview")
            st.write(f"**Short summary:** {dossier.get('short_summary', '')}")
            st.write(f"**Long summary:** {dossier.get('long_summary', '')}")

            final_fields = dossier.get("final_fields", {})
            field_rows = []
            for fname, fdata in final_fields.items():
                val = fdata.get("value")
                evidence = fdata.get("evidence", []) or []
                field_rows.append({
                    "field": fname,
                    "value": str(val) if val is not None else "",
                    "evidence_count": len(evidence),
                })
            show_dataframe(field_rows, "No fields extracted yet.")

            with st.expander("Browse per-document JSON artifacts"):
                json_dir = run_dir / "json_for_each_file"
                if json_dir.exists():
                    for path in sorted(json_dir.glob("*.json")):
                        data = read_json(path, default={})
                        if isinstance(data, dict):
                            st.write(f"**{path.name}** — {data.get('short_summary', '')[:200]}")
                            with st.expander("Full JSON"):
                                st.json(data)

            dossier_path = run_dir / "company_dossier_merged.json"
            if dossier_path.exists():
                st.download_button(
                    "Download company_dossier_merged.json",
                    data=dossier_path.read_bytes(),
                    file_name="company_dossier_merged.json",
                    mime="application/json",
                )

with tabs[7]:
    st.subheader("8) Excel Preview + Download")
    run_dir = Path(st.session_state.run_dir) if st.session_state.run_dir else None
    if not run_dir:
        st.warning("Create a run first.")
    else:
        if st.button("Generate Excel workbook", type="primary"):
            out = create_excel_export(run_dir)
            st.success(f"Created {out.name}")
        excel_path = run_dir / "millenia_dossier_export.xlsx"
        if excel_path.exists():
            st.download_button(
                "Download Excel workbook",
                data=excel_path.read_bytes(),
                file_name="millenia_dossier_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        st.markdown("### Preview key output tables")
        output_choice = st.selectbox("Preview", ["dossier_fields", "cleaned_tables", "image_summaries", "layout_items"])
        if output_choice == "dossier_fields":
            dossier = read_json(run_dir / "company_dossier_merged.json", default=None) or {}
            final_fields = dossier.get("final_fields", {})
            rows = []
            for fname, fdata in final_fields.items():
                rows.append({
                    "field": fname,
                    "value": str(fdata.get("value", "")) if fdata.get("value") is not None else "",
                    "evidence_count": len(fdata.get("evidence", []) or []),
                })
            show_dataframe(rows)
        elif output_choice == "cleaned_tables":
            cleaned_tables = read_json(run_dir / "cleaned_tables.json", default=[])
            rows = []
            for table in cleaned_tables:
                for i, row in enumerate(table.get("rows", []) or [], start=1):
                    rows.append({"table_id": table.get("table_id"), "row_number": i, **row})
            show_dataframe(rows)
        elif output_choice == "image_summaries":
            show_dataframe(read_json(run_dir / "image_summaries.json", default=[]))
        elif output_choice == "layout_items":
            show_dataframe(read_json(run_dir / "layout_items.json", default=[]))
