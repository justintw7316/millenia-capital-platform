from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .ollama_client import chat_json, chat_text
from .utils import read_json, safe_text, write_json

FIELD_DEFINITIONS: dict[str, str] = {
    "company_name": "Common company name or brand name.",
    "legal_company_name": "Full legal company name, such as Example, Inc. or Example LLC.",
    "website": "Company website URL or domain.",
    "industry_or_sector": "Industry, sector, vertical, or market category.",
    "company_stage_or_company_idea": "Company stage, concept, or high-level company idea.",
    "headquarters_location": "Headquarters or main operating location.",
    "incorporation_location": "State/country/jurisdiction of incorporation or formation.",
    "year_founded": "Year the company was founded.",
    "legal_entity_structure": "Legal entity type, such as C-Corp, LLC, corporation, limited partnership, etc.",
    "one_line_description": "One clear sentence explaining what the company does.",
    "company_problem": "The problem, pain point, or unmet need the company addresses.",
    "company_solution": "The solution, product, service, platform, or technology the company provides.",
    "business_model": "How the company makes or plans to make money.",
    "target_customers": "Customer types, buyers, users, accounts, patient groups, or market segments.",
    "competitive_advantage": "Moat, differentiation, unique advantage, or reason the company can win.",
    "product_status": "Product stage/status, such as concept, prototype, beta, launched, preclinical, IND-enabling, FDA-cleared, etc.",
    "revenue": "Revenue amount or revenue description, monthly or annual if stated.",
    "mrr": "Monthly recurring revenue.",
    "arr": "Annual recurring revenue.",
    "growth_rate": "Growth rate, including MoM, QoQ, YoY, revenue growth, user growth, or pipeline growth.",
    "gross_margin": "Gross margin or contribution margin.",
    "burn_rate": "Monthly burn or cash burn rate.",
    "runway_months": "Runway in months or years.",
    "cash_on_hand": "Cash balance or cash available.",
    "ltv": "Customer lifetime value.",
    "cac": "Customer acquisition cost.",
    "ltv_cac_ratio": "LTV/CAC ratio.",
    "payback_period": "CAC payback period or payback period.",
    "amount_raised_to_date": "Total amount raised so far.",
    "cap_table_summary": "Cap table, ownership, share structure, option pool, or major ownership summary.",
    "previous_investors": "Existing or previous investors.",
    "valuation_last_round": "Valuation from a prior financing round.",
    "target_raise_amount": "Current fundraising target or amount being raised.",
    "minimum_investment": "Minimum investment amount, if listed.",
    "valuation_current_ask": "Current valuation ask if stated.",
    "pre_money_valuation": "Current pre-money valuation.",
    "post_money_valuation": "Current post-money valuation.",
    "current_round": "Current round type, such as Seed, Series A, bridge, SAFE, etc.",
    "instrument": "Investment instrument, such as SAFE, equity, priced round, convertible note, debt, etc.",
    "use_of_funds": "How the company plans to use the funds.",
    "funding_milestones": "Milestones expected to be reached with current or future funding.",
    "users_total": "Total users, customers, patients, accounts, downloads, installations, or similar.",
    "users_active": "Active users or active customers.",
    "dau": "Daily active users.",
    "mau": "Monthly active users.",
    "sales_pipeline": "Sales pipeline value, stage, opportunities, accounts, or prospects.",
    "conversion_rates": "Conversion rates in funnel, sales, product, trial, or user conversion.",
    "average_order_value": "Average order value, average contract value, or average selling price.",
    "retention_rates": "Retention rate, renewal rate, repeat usage, or customer retention.",
    "churn_rate": "Churn rate or customer loss rate.",
    "engagement_metrics": "Engagement, usage, activity, frequency, session, or other product usage metrics.",
    "partnerships": "Partnerships, strategic partners, collaborators, channels, distributors, or alliances.",
    "contracts_signed": "Signed contracts, agreements, customer contracts, enterprise agreements, or committed deals.",
    "lois": "Letters of intent, memorandums of understanding, non-binding indications, or LOIs.",
    "key_milestones": "Important company, product, regulatory, technical, commercial, or financing milestones.",
    "key_hires": "Important hires, planned hires, new executives, or important team additions.",
    "incorporation_documents": "Corporate charter, certificate of incorporation, bylaws, good standing, or corporate documents.",
    "ip_ownership": "IP ownership, licensing rights, assignments, exclusive licenses, or ownership status.",
    "trademarks_patents": "Patents, patent applications, trademarks, issued patents, pending patents, or patent families.",
    "regulatory_exposure": "Regulatory risk, FDA, SEC, HIPAA, financial regulation, clinical regulation, medical device regulation, etc.",
    "risk_disclosures": "Explicit risks, concerns, dependencies, litigation, compliance risks, or warnings.",
    "market_size": "Market size, TAM, SAM, SOM, market opportunity, market growth, or addressable market.",
}

FIELD_NAMES = list(FIELD_DEFINITIONS.keys())

LIST_FIELDS = {
    "industry_or_sector", "target_customers", "previous_investors",
    "use_of_funds", "funding_milestones", "users_total", "users_active",
    "sales_pipeline", "engagement_metrics", "partnerships", "contracts_signed",
    "lois", "key_milestones", "key_hires", "incorporation_documents",
    "ip_ownership", "trademarks_patents", "regulatory_exposure",
    "risk_disclosures", "market_size", "cap_table_summary",
}

DOCUMENT_CATEGORIES = [
    "Pitch deck / investor deck", "Investment tear sheet / offering summary",
    "Executive summary", "Investor update", "Term sheet",
    "Financial model / projections", "Cap table / ownership summary",
    "Revenue metrics", "Product metrics", "Cohort / retention analysis",
    "Churn analysis", "CAC / LTV / unit economics analysis",
    "SAFE agreement / legal document", "Convertible note agreement / legal document",
    "Subscription agreement / purchase agreement", "Customer list",
    "Sales pipeline", "Customer contract", "Enterprise customer agreement",
    "Key commercial contract", "Founder resume / biography", "Employee roster",
    "Hiring plan", "Organizational chart", "Product demo / screenshots",
    "Product roadmap", "Technical architecture", "Security policy",
    "Privacy policy", "Market / TAM analysis", "Competitive landscape",
    "Strategic partnership document", "Licensing agreement", "IP assignment",
    "Patent / IP summary", "Corporate charter / certificate of incorporation",
    "Bylaws", "Board minutes / board consent", "Certificate of good standing",
    "Option plan / equity incentive plan", "Bank statement", "Tax filing",
    "Insurance policy", "Litigation disclosure", "Vendor agreement",
    "Millenia engagement agreement", "Unknown File Type",
]


def _field_lines() -> str:
    return "\n".join(f"- {n}: {d}" for n, d in FIELD_DEFINITIONS.items())


def _split_documents(feed_text: str) -> list[dict[str, str]]:
    marker = "# Document:"
    parts = feed_text.split(marker)
    docs = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        first_line = part.split("\n")[0].strip()
        doc_name = first_line or f"doc_{len(docs)+1}"
        body = part[len(first_line):].strip() if first_line else part
        docs.append({"doc_name": doc_name, "text": body or part})
    return docs


def _build_doc_prompt(doc: dict[str, str]) -> str:
    return f"""
You are extracting structured company information from one document.

Document name: {doc["doc_name"]}

Allowed fields:
{_field_lines()}

Allowed document categories:
{chr(10).join(f'- {c}' for c in DOCUMENT_CATEGORIES)}

Return JSON with this exact shape:
{{
  "document_category": "one category from the list",
  "short_summary": "brief factual summary of this document",
  "long_summary": "detailed factual summary",
  "fields": {{
    "field_name": {{
      "value": "string, list, number, or null",
      "answer": "short explanation",
      "evidence_quote": "exact quote from the document supporting this value"
    }}
  }},
  "key_people": [
    {{
      "full_name": "full name",
      "role": "role/title or null",
      "evidence_quote": "exact quote supporting this person"
    }}
  ]
}}

Rules:
- Use only the document text below.
- Do not guess or invent facts.
- Use null for unknown fields.
- For evidence_quote, copy a short exact phrase from the document.
- Return only JSON. No markdown fences.

Document text:
{doc["text"][:30000]}
""".strip()


def _build_merge_prompt(per_doc_fields: list[dict[str, Any]], model: str) -> str:
    doc_summaries = []
    for i, doc in enumerate(per_doc_fields):
        doc_name = doc.get("doc_name", f"Document {i+1}")
        fields_summary = []
        for fname, fdata in (doc.get("doc_fields", {}) or {}).items():
            val = fdata.get("value")
            if val is not None:
                fields_summary.append(f"  {fname}: {val}")
        doc_summaries.append(f"Document: {doc_name}\nFields:\n" + "\n".join(fields_summary))

    return f"""
You are merging extracted company fields from multiple documents into one unified company dossier.

Each document below was analyzed independently. Some fields may have conflicting or complementary values.

For each field, produce:
- value: the best consolidated value across all documents (null if not found anywhere)
- answer: a brief explanation of how you determined this value
- evidence: an array of {{{{ "doc_id", "file_name", "quote", "page_start", "page_end" }}}} objects

Rules:
- For scalar fields, pick the most specific and reliable value. Note conflicts in the answer.
- For list fields, combine values from all documents, deduplicate.
- Use null for fields with no support in any document.
- Return only JSON.

Target schema:
{{
  "short_summary": "one-paragraph company summary",
  "long_summary": "detailed company summary",
  "final_fields": {{
    "field_name": {{
      "value": null or string or list,
      "answer": "explanation",
      "evidence": [{{"doc_id": "...", "file_name": "...", "quote": "...", "page_start": null, "page_end": null}}]
    }}
  }}
}}

Documents to merge:
{json.dumps(doc_summaries, indent=2)}

All 56 allowed field names:
{json.dumps(FIELD_NAMES, indent=2)}
""".strip()


def run_per_document_extraction(
    run_dir: Path,
    llm_model: str,
    ollama_url: str,
) -> list[dict[str, Any]]:
    feed_path = run_dir / "file_llm_feed.md"
    if not feed_path.exists():
        raise FileNotFoundError(f"LLM feed not found: {feed_path}")

    feed_text = feed_path.read_text(encoding="utf-8")
    docs = _split_documents(feed_text)

    json_dir = run_dir / "json_for_each_file"
    json_dir.mkdir(parents=True, exist_ok=True)

    per_doc_results = []
    manifest = read_json(run_dir / "run_manifest.json", default={})
    doc_manifest = manifest.get("documents", [])

    for i, doc in enumerate(docs):
        doc_id = f"doc_{i+1:03d}"
        file_name = doc["doc_name"]
        source_path = ""
        if i < len(doc_manifest):
            file_name = doc_manifest[i].get("file_name", doc["doc_name"])
            source_path = doc_manifest[i].get("source_path", "")

        prompt = _build_doc_prompt(doc)
        try:
            result = chat_json(prompt, model=llm_model, ollama_url=ollama_url, temperature=0.1, timeout=300)
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
                    {
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "quote": evidence_quote or "",
                        "page_start": None,
                        "page_end": None,
                    }
                ] if evidence_quote else [],
            }

        doc_json_path = json_dir / f"{doc_id}.json"
        write_json(doc_json_path, doc_output)
        per_doc_results.append(doc_output)

    return per_doc_results


def run_merge_dossier(
    run_dir: Path,
    llm_model: str,
    ollama_url: str,
) -> dict[str, Any]:
    json_dir = run_dir / "json_for_each_file"
    if not json_dir.exists():
        raise FileNotFoundError(f"No per-document JSONs found at {json_dir}")

    per_doc_fields = []
    for path in sorted(json_dir.glob("*.json")):
        data = read_json(path, default={})
        if isinstance(data, dict) and data.get("doc_fields"):
            per_doc_fields.append(data)

    if not per_doc_fields:
        raise ValueError("No per-document extractions found to merge.")

    merge_prompt = _build_merge_prompt(per_doc_fields, llm_model)

    try:
        merged = chat_json(merge_prompt, model=llm_model, ollama_url=ollama_url, temperature=0.1, timeout=600)
        if not isinstance(merged, dict):
            raise ValueError("Merge returned non-object JSON")
    except Exception as exc:
        merged = {
            "short_summary": f"Merge failed: {exc}",
            "long_summary": "",
            "final_fields": {},
        }

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
    return dossier


def run_extraction(
    run_dir: Path,
    llm_model: str,
    ollama_url: str,
    analysis_mode: str = "Consolidated Dossier",
) -> dict[str, Any]:
    per_doc_results = run_per_document_extraction(run_dir, llm_model=llm_model, ollama_url=ollama_url)
    dossier = run_merge_dossier(run_dir, llm_model=llm_model, ollama_url=ollama_url)
    return dossier
