# Millenia Dossier

**PDF → Docling → LLM Extraction → Structured Company Dossier with Evidence**

Automated pipeline that ingests company PDF files, parses them with Docling OCR, then uses local LLMs (via Ollama) to extract 62 structured fields into a company dossier with per-field evidence tracking. Built as a Streamlit app with 8 tabbed phases.

---

## Features

- **Browse UI** — Inline directory browser for folder selection (no native OS picker)
- **8-tab pipeline** — Upload → Docling → Item Review → Table Correction → VLM Review → Build Feed → 62-Field Extraction → Excel Export
- **62-field company schema** — Financials, fundraise, product, traction, market, risk, IP, and more
- **Per-field evidence** — Every field value links back to source `{doc_id, file_name, quote, page_start, page_end}`
- **Streaming LLM** — Token-by-token output with typing cursor for all LLM and VLM phases
- **Two-phase extraction** — Per-document JSONs saved for debugging → merged into final dossier
- **Select Run** — Sidebar run picker to resume or re-run phases from previous sessions
- **Excel export** — Multi-sheet workbook with dossier fields, documents, layout items, and settings
- **Fully offline** — All LLM inference runs locally via Ollama; no API keys required

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) running locally (default: `http://localhost:11434`)
- At least one model pulled (recommended: `nemotron-3-super:cloud` for text, `llama3.2-vision:latest` for vision)

### Install

```bash
pip install streamlit pandas openpyxl python-dotenv Pillow requests pymupdf numpy docling
```

### Configure

```bash
cp .env.example .env
# Edit .env to set your models (defaults work out of the box)
```

### Run

```bash
streamlit run app/streamlit_app.py
```

Or double-click `run_app.bat` on Windows.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `nemotron-3-super:cloud` | Text LLM for extraction, table correction |
| `VLM_MODEL` | `llama3.2-vision:latest` | Vision LLM for image analysis |
| `RUNS_DIR` | `runs` | Directory for run output artifacts |

---

## Pipeline Architecture

```
┌──────────┐    ┌──────────┐    ┌─────────────┐    ┌──────────────────┐
│  Upload  │ →  │ Docling  │ →  │   Item      │ →  │     Table        │
│ (PDFs)   │    │ (OCR)    │    │   Review    │    │   Correction     │
└──────────┘    └──────────┘    └─────────────┘    └────────┬─────────┘
                                                            │
┌──────────┐    ┌──────────────┐    ┌──────────┐           │
│  Excel   │ ←  │ 62-Field     │ ←  │  Build   │ ←─────────┘
│  Export  │    │ Extraction   │    │  Feed    │
└──────────┘    └──────────────┘    └──────────┘
                    │
              ┌─────┴─────┐
              │ per-doc   │
              │ JSONs     │
              └───────────┘
```

---

## Pipeline Phases

### 1. Upload
Select a folder of input files (PDF, DOCX, PPTX, XLSX, images, text, markdown). Files are copied into the run directory's `input/` folder. Supports native file picker text input or the inline Browse UI.

### 2. Docling
Parses each file with Docling OCR: extracts text, tables, pictures, section headers, and list items. Generates:
- **Crop PNGs** for each table and picture (2× resolution)
- **Outlined PDF** with detected items highlighted
- **Monolithic JSONs**: `run_manifest.json`, `layout_items.json`, `raw_tables.json`, `visual_items.json`
- **Per-doc artifacts** in `parsed/doc_NNN/`

### 3. Item Review *(UI only)*
Human review of all layout items. Filter by document, type (table/picture/text), or page. Set status and notes for each item to guide downstream phases.

### 4. Table Correction
LLM-corrects selected tables from raw Docling output. Each table is cleaned via `chat_text_stream()` for streaming output. Saves to `cleaned_tables.json`.

### 5. VLM Review
Vision LLM analyzes selected pictures/charts. Each crop image is sent to `chat_vision_stream()` with nearby text context. Saves descriptions to `image_summaries.json`.

### 6. Build Feed
Assembles a single markdown feed (`file_llm_feed.md`) from layout items, cleaned tables, and image summaries. Pure data assembly — no LLM calls, runs in milliseconds.

### 7. 62-Field Extraction
Two sub-phases:

- **Step 1 (Per-document):** Splits the feed by document → for each doc, LLM extracts all 62 fields → saves per-doc JSONs to `json_for_each_file/<doc_id>.json`. Runs via `chat_json_stream()`.

- **Step 2 (Merge):** Reads all per-doc JSONs → LLM merges them into a single `company_dossier_merged.json` with consolidated evidence.

### 8. Excel Export
Generates `millenia_dossier_export.xlsx` with multiple sheets: `Dossier Fields`, `Documents`, `Layout Items`, `Raw Tables`, `Image Summaries`, and `Settings`.

---

## Run Management

- Each run creates a directory: `runs/millenia_YYYYMMDD_HHMMSS_abc123/`
- The **sidebar selectbox** lists all existing runs as `name — N/8`
- Phase status is shown as a grid: ✅ (complete) / ⬜ (pending)
- Switching runs restores `uploaded_paths` from the run's `input/` directory
- Re-running a phase overwrites only that phase's output files
- Phases are detected as complete by the presence of key output files (not a state database)

---

## Output Format

```json
{
  "short_summary": "Brief company description",
  "long_summary": "Detailed analysis across all documents",
  "final_fields": {
    "company_name": {
      "value": "Acme Corp",
      "answer": true,
      "evidence": [
        {
          "doc_id": "doc_001",
          "file_name": "pitch_deck.pdf",
          "quote": "Acme Corp is a leading...",
          "page_start": 1,
          "page_end": 1
        }
      ]
    }
  },
  "documents": [
    {
      "doc_id": "doc_001",
      "file_name": "pitch_deck.pdf",
      "text": "..."
    }
  ],
  "model_used": "nemotron-3-super:cloud",
  "run_id": "millenia_20260606_120000_abc123"
}
```

---

## Output Files Reference

| File | Phase | Description |
|------|-------|-------------|
| `input/` | Upload | Copied source files |
| `run_manifest.json` | Docling | PDF metadata (pages, dimensions) |
| `layout_items.json` | Docling | All parsed items with type, page, bbox, text |
| `raw_tables.json` | Docling | Raw table data from Docling |
| `visual_items.json` | Docling | Picture/chart item metadata |
| `parsed/doc_NNN/` | Docling | Per-document crop PNGs and JSONs |
| `cleaned_tables.json` | Table Correction | LLM-corrected table data |
| `image_summaries.json` | VLM Review | VLM-generated image descriptions |
| `file_llm_feed.md` | Build Feed | Consolidated markdown for LLM |
| `json_for_each_file/*.json` | Extraction | Per-document field extractions |
| `company_dossier_merged.json` | Extraction | Final merged dossier |
| `millenia_dossier_export.xlsx` | Excel | Multi-sheet Excel workbook |

---

## Project Structure

```
Millenia-Dossier/
├── app/
│   └── streamlit_app.py              # Main UI — 8 tabs, Browse UI, Select Run
├── src/
│   └── millenia_dossier/
│       ├── __init__.py
│       ├── config.py                  # .env loading
│       ├── extraction.py              # 62 field definitions + per-doc + merge
│       ├── docling_pipeline.py        # PDF parsing, cropping, layout extraction
│       ├── ollama_client.py           # LLM/VLM calls + streaming generators
│       ├── table_corrector.py         # Table cleanup via LLM
│       ├── visual_analyzer.py         # VLM image analysis
│       ├── feed_builder.py            # Markdown feed assembly
│       ├── excel_exporter.py          # Excel workbook generation
│       ├── quick_skim.py              # Crop image analysis
│       └── utils.py                   # JSON/IO helpers
├── docs/
│   ├── FOLDER_BROWSER_REVERT.md       # Revert Browse UI to file picker
│   └── pipeline_efficiency_analysis.md # Bottleneck analysis
├── runs/                              # Run output directories
│   └── millenia_YYYYMMDD_HHMMSS_xxx/
│       ├── input/                     # Copied source files
│       ├── parsed/                    # Docling per-doc artifacts
│       ├── json_for_each_file/        # Per-doc extraction JSONs
│       ├── run_manifest.json
│       ├── layout_items.json
│       ├── raw_tables.json
│       ├── visual_items.json
│       ├── cleaned_tables.json
│       ├── image_summaries.json
│       ├── file_llm_feed.md
│       └── company_dossier_merged.json
├── _revert_backups/
│   └── streamlit_app.pre-browse.bak   # Pre-Browse-UI backup
├── .env                               # Configuration (user-edited)
├── .env.example                       # Configuration template
├── requirements.txt                   # Python dependencies
└── run_app.bat                        # Windows launcher
```

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| "Connection refused" | Ollama not running | Start Ollama: `ollama serve` |
| Model not found | Model not pulled | `ollama pull nemotron-3-super:cloud` |
| Docling hangs on first run | Downloading model files | Wait 2–3 minutes for the initial download |
| Streaming output not showing | Using non-streaming function | Ensure `chat_text_stream()` is called (not `chat_text()`) |
| "No module named 'millenia_dossier'" | Wrong working directory | Run from `Millenia-Dossier/` (app adds `src/` to `sys.path` at startup) |
| Phase stuck at 0% | LLM timeout | Check Ollama terminal for errors; increase timeout in `ollama_client.py` |
| Excel export empty | Extraction not complete | Wait for Step 2 (Merge) to finish; check `company_dossier_merged.json` |
| Browse UI shows no drives | Permissions | Run Streamlit with appropriate access; check `Path.drives()` output |
| Run appears with 0/8 phases | Empty run directory | Select or create a run that has processed phases |
| Docling crops fail on some PDFs | PyMuPDF rendering error | Check that the PDF is not password-protected or corrupted |
| VLM returns empty descriptions | Vision model not available | Ensure `VLM_MODEL` supports vision (e.g., `llama3.2-vision:latest`) |
| Evidence quotes truncated | 30k char document truncation | Split very large documents; check `_split_documents` in `extraction.py` |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI Framework | [Streamlit](https://streamlit.io) |
| PDF Parsing | [Docling](https://github.com/DS4SD/docling) |
| OCR / Layout | Docling (built-in) |
| PDF Rendering | [PyMuPDF](https://pymupdf.readthedocs.io) |
| LLM Inference | [Ollama](https://ollama.ai) (local) |
| Image Analysis | Ollama Vision Models |
| Data Processing | `pandas`, `numpy` |
| Excel Export | `openpyxl` |
| Configuration | `python-dotenv` |

---

## License

Proprietary — Millenia. All rights reserved.
