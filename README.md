# Millenia Capital — Investor-Founder Matching Platform

The investor-research core of Millenia's capital-formation system: take a startup's
documents, build a structured company profile, and match the company to the right
investors using a hybrid (vector + structured) ranking engine — with live market
signals layered in.

This public repository integrates the work of several contributors into one
pipeline. It is a focused slice of Millenia's broader 14-step capital-formation
process (it covers document intake and investor discovery).

---

## The pipeline

```
  company documents
        │
        ▼
┌───────────────────┐     adapters/dossier_adapter      ┌──────────────────────┐
│  Millenia-Dossier │ ── company_profile_extracted ───▶ │   matching engine    │
│  (Docling + LLM)  │            (canonical IR)         │  (pgvector hybrid)   │
└───────────────────┘            then ─▶ Deal           │  ranked investors +  │
                                                        │  explanations        │
┌─────────────────────┐   adapters (signal feed)        │                      │
│ investor-signal-tool│ ── InvestorSignal records ────▶ │  recency / intent    │
│  (live web/news)    │      data/signals/*.json        │  enrichment          │
└─────────────────────┘                                 └──────────────────────┘
```

A founder's documents are extracted into a **canonical company profile**, which —
once a human approves it — is flattened onto a `Deal`. The **matching engine**
ranks investors for that deal, and the **signal tool** feeds live activity signals
that keep each investor's recency/intent fresh.

---

## Components

| Path | What it is | Maturity |
|------|-----------|----------|
| `matching/`, `core/`, `integrations/`, `migrations/`, `scripts/` | **pgvector hybrid matching engine** — hard eligibility filters + multi-namespace vector retrieval + keyword + structured industry/stage/check/recency features, with explainable reasons and a human approval queue. | MVP → production-leaning |
| `Millenia-Dossier/` | **Document-extraction pipeline** — Streamlit app that OCRs company PDFs with Docling and extracts structured fields with per-field evidence. | MVP |
| `investor-signal-tool/` | **Live investor signal detector** — crawls investor sites + news to score real-time investment intent/recency. | MVP |
| `adapters/` | **Integration layer** — converts the dossier output into the canonical profile and onto a `Deal`, normalises financial strings to numbers, and (in progress) feeds the signal tool's output to the matcher as `InvestorSignal`s. | in progress |
| `tests/` | pytest suite for the matching engine and adapters. | growing |

---

## Design principles (high-stakes by default)

Capital decisions are high-stakes, so the engine is built to **fail loud rather
than mislead**:

- **No silent degradation.** If the embedding model can't load, the matcher
  refuses to score on meaningless hash vectors instead of emitting confident-
  looking noise (override explicitly with `allow_hash_fallback=True`).
- **Honest provenance.** Enriched/estimated investor attributes are flagged so a
  guess is never read as a verified fact.
- **Human-in-the-loop.** Document-extracted profiles are **not** auto-consumed by
  matching; they require employee approval first (`approval.status == "approved"`).
- **Evidence over confidence scores.** Extracted fields carry sources + evidence +
  a `review_status`, not opaque model confidence.

---

## Development

```bash
pip install -r requirements-dev.txt
python3 -m pytest            # run the test suite
```

The matching package imports without an Anthropic API key (LLM-backed steps check
for it at point of use). Heavy deps (sentence-transformers, pgvector) are optional
and loaded lazily; the suite runs on a clean Python install.

---

## Data & confidentiality

This is a **public** repository. Confidential company materials (data-room PDFs,
cap tables, signed agreements) are **never** committed — see `.gitignore`. Both
the dossier and signal tools run against locally-supplied files that stay on disk.

---

## Contributors

- **Justin** ([@justintw7316](https://github.com/justintw7316)) — matching engine, integration
- **Angel Mencia** ([@AMenxia](https://github.com/AMenxia)) — Millenia-Dossier extraction pipeline
- **Benjamin Lu** ([@JieminBenn](https://github.com/JieminBenn)) — investor-signal-tool
