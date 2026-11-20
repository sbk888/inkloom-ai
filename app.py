import io
import json
import os
import re
import uuid
import sqlite3
from datetime import datetime

import fitz
import streamlit as st
from google import genai
from google.genai import types
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    KeepTogether, PageBreak
)
from xml.sax.saxutils import escape

from evidence_engine import analyze_evidence
from verification import enforce_review_guardrails


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Inkloom AI",
    page_icon="I",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #f7f8fc; }
.block-container { padding-top: 2rem; }
h1 { color: #25234a; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# API CONFIGURATION
# ==========================================

MODEL_NAME = "gemini-3.8-flash"


def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "")


def get_gemini_client():
    api_key = get_api_key()
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


# ==========================================
# DOCUMENT PROCESSING
# ==========================================

def extract_text(uploaded_file):
    if uploaded_file is None:
        return ""

    file_bytes = uploaded_file.getvalue()

    if uploaded_file.name.lower().endswith(".pdf"):
        with fitz.open(stream=file_bytes, filetype="pdf") as document:
            return "\n".join(page.get_text() for page in document)

    if uploaded_file.name.lower().endswith(".txt"):
        return file_bytes.decode("utf-8", errors="replace")

    return ""


def build_source_records(uploaded_file, manual_text):
    """Keep page/line provenance for each supplied guideline line."""
    records = []
    if uploaded_file is not None:
        raw = uploaded_file.getvalue()
        filename = uploaded_file.name
        if filename.lower().endswith(".pdf"):
            with fitz.open(stream=raw, filetype="pdf") as document:
                for page_number, page in enumerate(document, start=1):
                    page_text = page.get_text()
                    if page_text.strip():
                        records.append({
                            "text": page_text, "source": filename,
                            "location": f"Page {page_number}"
                        })
                    for line_number, line in enumerate(page_text.splitlines(), start=1):
                        if line.strip():
                            records.append({
                                "text": line.strip(), "source": filename,
                                "location": f"Page {page_number}, line {line_number}"
                            })
        elif filename.lower().endswith(".txt"):
            text = raw.decode("utf-8", errors="replace")
            if text.strip():
                records.append({
                    "text": text, "source": filename, "location": "Text document"
                })
            for line_number, line in enumerate(text.splitlines(), start=1):
                if line.strip():
                    records.append({
                        "text": line.strip(), "source": filename,
                        "location": f"Line {line_number}"
                    })

    for line_number, line in enumerate((manual_text or "").splitlines(), start=1):
        if line.strip():
            records.append({
                "text": line.strip(), "source": "Manual guideline input",
                "location": f"Line {line_number}"
            })
    return records


def normalize_source_line(value):
    return re.sub(r"\s+", " ", re.sub(r"^[-*•]\s*", "", str(value))).strip().casefold()


def find_source_references(statement, source_records):
    target = normalize_source_line(statement)
    if not target:
        return []
    matches = []
    for record in source_records or []:
        record_text = normalize_source_line(record.get("text", ""))
        if record_text == target or (len(target) >= 12 and target in record_text):
            ref = {"source": record.get("source", "Guidelines"), "location": record.get("location", "") }
            if ref not in matches:
                matches.append(ref)
    return matches


def attach_source_traceability(results, source_records):
    for result in results:
        result["evidence_sources"] = find_source_references(result.get("evidence", ""), source_records)
        result["restriction_sources"] = find_source_references(result.get("restriction", ""), source_records)
        for match in result.get("approved_matches", []):
            match["sources"] = find_source_references(match.get("text", ""), source_records)
        for match in result.get("restriction_matches", []):
            match["sources"] = find_source_references(match.get("text", ""), source_records)
    return results


# ==========================================
# GUIDELINE EXTRACTION
# ==========================================

def extract_sections(guidelines):
    approved = []
    restrictions = []
    other = []
    section = "other"

    approved_headings = {
        "approved facts", "verified facts", "product facts",
        "approved claims", "approved information"
    }
    restriction_headings = {
        "restrictions", "prohibited claims", "claims not allowed",
        "restricted claims", "prohibited statements"
    }
    other_headings = {
        "brand tone", "brand voice", "brand identity", "brand name",
        "product", "company", "target audience"
    }

    for original_line in guidelines.splitlines():
        line = original_line.strip()
        if not line:
            continue

        cleaned = re.sub(r"^[-*•]\s*", "", line).strip()
        heading = cleaned.lower().strip(": ")

        if heading in approved_headings:
            section = "approved"
            continue
        if heading in restriction_headings:
            section = "restricted"
            continue
        if heading in other_headings:
            section = "other"
            continue
        if cleaned.endswith(":") and len(cleaned) < 50:
            continue

        if section == "approved":
            approved.append(cleaned)
        elif section == "restricted":
            restrictions.append(cleaned)
        else:
            other.append(cleaned)

    return approved, restrictions, other


def get_brand_name(other_details):
    for item in other_details:
        match = re.match(r"(?i)^brand\s*name\s*:\s*(.+)$", item)
        if match:
            return match.group(1).strip()
    return "Not specified"


# ==========================================
# BASIC RULE ENGINE
# ==========================================

RISK_PATTERNS = {
    "Absolute claim": [
        r"\b100%\b", r"\bguaranteed\b", r"\bguarantees\b",
        r"\balways\b", r"\bnever\b", r"\bpermanent(?:ly)?\b",
        r"\bcompletely\b", r"\brisk-free\b"
    ],
    "Health-related claim": [
        r"\bcures?\b", r"\bheals?\b", r"\btreats?\b",
        r"\bremoves?\b", r"\bprevents?\b", r"\beliminates?\b",
        r"\bclinically proven\b"
    ],
    "Performance claim": [
        r"\binstant\b", r"\bfastest\b", r"\bbest\b",
        r"\bnumber one\b", r"\bmost effective\b"
    ],
    "Quantified claim": [
        r"\b\d+(?:\.\d+)?\s*(%|percent|days?|hours?|minutes?|times?|weeks?|months?)\b"
    ]
}


def detect_risks(sentence):
    found = []
    for category, patterns in RISK_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                found.append(category)
                break
    return found


# ==========================================
# FALLBACK EVIDENCE MATCHING
# ==========================================

def tokenize(text):
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def similarity(text1, text2):
    words1 = tokenize(text1)
    words2 = tokenize(text2)
    if not words1 or not words2:
        return 0
    overlap = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return overlap / union if union else 0


def split_sentences(content):
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", content)
        if sentence.strip()
    ]


def basic_review(content, guidelines):
    approved, restrictions, _ = extract_sections(guidelines)
    results = []

    for sentence in split_sentences(content):
        risks = detect_risks(sentence)
        evidence = ""
        best_score = 0

        for fact in approved:
            score = similarity(sentence, fact)
            if score > best_score:
                best_score = score
                evidence = fact

        restriction = ""
        for item in restrictions:
            words = [
                word for word in re.findall(r"[a-zA-Z]+", item.lower())
                if len(word) > 4
            ]
            if any(word in sentence.lower() for word in words):
                restriction = item
                break

        if restriction:
            status = "Restricted claim"
        elif risks:
            status = "Needs verification"
        elif best_score >= 0.25:
            status = "Evidence found"
        else:
            status = "Unsupported claim"

        results.append({
            "sentence": sentence,
            "status": status,
            "risks": risks,
            "evidence": evidence,
            "restriction": restriction,
            "reason": (
                "Basic rule-based review. AI analysis was not used."
            ),
            "suggestion": (
                "Verify this statement against approved brand sources."
            ),
            "approved_matches": [],
            "restriction_matches": []
        })

    return results


# ==========================================
# AI REVIEW + SEMANTIC RETRIEVAL
# ==========================================

def ai_review(content, guidelines, source_records=None):
    client = get_gemini_client()
    if client is None:
        raise ValueError("Gemini API key not configured.")

    approved, restrictions, other = extract_sections(guidelines)
    sentences = split_sentences(content)

    if not sentences:
        raise ValueError("No meaningful sentences found.")

    retrieval_data = []
    for sentence in sentences:
        evidence_result = analyze_evidence(
            client=client,
            claim=sentence,
            approved_facts=approved,
            restrictions=restrictions
        )
        retrieval_data.append({
            "sentence": sentence,
            "approved_candidates": evidence_result["approved_evidence"],
            "restriction_candidates": evidence_result["restriction_matches"]
        })

    prompt = f"""
You are Inkloom AI, an evidence-grounded brand content compliance analyst.

Analyze every marketing sentence using ONLY the supplied brand sources.

APPROVED FACTS:
{json.dumps(approved, ensure_ascii=False)}

BRAND RESTRICTIONS:
{json.dumps(restrictions, ensure_ascii=False)}

OTHER BRAND INFORMATION:
{json.dumps(other, ensure_ascii=False)}

SEMANTIC RETRIEVAL RESULTS:
{json.dumps(retrieval_data, ensure_ascii=False)}

MARKETING SENTENCES:
{json.dumps(sentences, ensure_ascii=False)}

INSTRUCTIONS:
1. Analyze every meaningful marketing sentence.
2. Use retrieval candidates to identify potentially relevant facts and restrictions.
3. Similarity scores are NOT proof of factual support or a confirmed violation.
4. Decide whether a restriction actually conflicts with the sentence meaning.
5. A related approved fact does not automatically support the entire claim.
6. Only use exact approved facts as supporting evidence.
7. Only use exact supplied restrictions when identifying a restriction conflict.
8. Never invent product facts, benefits, numbers or claims.
9. Identify unsupported medical, performance, quantified, absolute and exaggerated claims.
10. If evidence is insufficient, say so clearly.
11. Provide a safer alternative using only approved facts.
12. Do not classify a claim as Evidence found merely because its similarity score is high.

Use these statuses:
- Restricted claim
- Needs verification
- Evidence found
- Unsupported claim

Return valid JSON only:
{{
  "results": [
    {{
      "sentence": "Original sentence",
      "status": "One of the four statuses",
      "risk_categories": [],
      "evidence": "",
      "restriction": "",
      "reason": "",
      "suggestion": ""
    }}
  ]
}}

Every evidence quotation must exactly match an approved fact.
Every restriction must exactly match a supplied restriction.
If no exact supporting evidence exists, use an empty evidence string.
If no actual restriction conflict exists, use an empty restriction string.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    data = json.loads(response.text)
    returned_results = data.get("results", [])
    valid_statuses = {
        "Restricted claim", "Needs verification",
        "Evidence found", "Unsupported claim"
    }

    validated = []
    for index, item in enumerate(returned_results):
        sentence = str(item.get("sentence", "")).strip()
        if not sentence:
            continue

        status = str(item.get("status", "")).strip()
        if status not in valid_statuses:
            status = "Needs verification"

        evidence = str(item.get("evidence", "")).strip()
        if evidence not in approved:
            evidence = ""

        restriction = str(item.get("restriction", "")).strip()
        if restriction not in restrictions:
            restriction = ""

        # Match by position to avoid ambiguity with repeated sentences.
        matching_retrieval = (
            retrieval_data[index]
            if index < len(retrieval_data)
            else {"approved_candidates": [], "restriction_candidates": []}
        )

        risks = item.get("risk_categories", [])
        if not isinstance(risks, list):
            risks = []

        checked = enforce_review_guardrails({
            "sentence": sentence,
            "status": status,
            "risks": [str(risk) for risk in risks],
            "evidence": evidence,
            "restriction": restriction,
            "reason": str(item.get("reason", "")),
            "suggestion": str(item.get("suggestion", "")),
            "approved_matches": matching_retrieval["approved_candidates"],
            "restriction_matches": matching_retrieval["restriction_candidates"]
        }, approved, restrictions)
        validated.append(checked)

    if not validated:
        raise ValueError("AI returned no valid sentence analysis.")

    return attach_source_traceability(validated, source_records or [])


# ==========================================
# AI CONTENT REWRITING
# ==========================================

def rewrite_content(original, guidelines, finding):
    client = get_gemini_client()
    if client is None:
        raise ValueError("Gemini API key not configured.")

    approved, restrictions, _ = extract_sections(guidelines)

    prompt = f"""
You are the content rewriting assistant for Inkloom AI.

Rewrite this marketing sentence to make it responsible and consistent
with the supplied approved brand information.

ORIGINAL SENTENCE:
{original}

REVIEW FINDING:
{finding}

APPROVED FACTS:
{json.dumps(approved, ensure_ascii=False)}

BRAND RESTRICTIONS:
{json.dumps(restrictions, ensure_ascii=False)}

RULES:
- Use only supplied approved facts.
- Never invent numbers, benefits or guarantees.
- Never violate brand restrictions.
- If no safe factual rewrite is possible, say so.
- Keep the sentence natural and marketing-friendly.
- Return only the rewritten sentence or a brief explanation that
  more evidence is needed.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig()
    )

    if not response.text:
        raise ValueError("AI could not generate a rewrite.")

    return response.text.strip()


# ==========================================
# AUDIT REPORT HELPERS
# ==========================================

def create_audit_id():
    return "INK-" + uuid.uuid4().hex[:10].upper()


REVIEW_DB_PATH = os.getenv("INKLOOM_REVIEW_DB", "inkloom_review_history.db")


def init_review_db():
    with sqlite3.connect(REVIEW_DB_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS review_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT NOT NULL,
                claim_index INTEGER NOT NULL,
                claim_text TEXT NOT NULL,
                decision TEXT NOT NULL,
                edited_text TEXT,
                reviewer_note TEXT,
                updated_at TEXT NOT NULL,
                UNIQUE(report_id, claim_index)
            )
        """)


def save_review_decision(report_id, claim_index, claim_text, decision, edited_text, reviewer_note):
    updated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    with sqlite3.connect(REVIEW_DB_PATH) as connection:
        connection.execute("""
            INSERT INTO review_decisions
            (report_id, claim_index, claim_text, decision, edited_text, reviewer_note, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(report_id, claim_index) DO UPDATE SET
                claim_text=excluded.claim_text, decision=excluded.decision,
                edited_text=excluded.edited_text, reviewer_note=excluded.reviewer_note,
                updated_at=excluded.updated_at
        """, (report_id, claim_index, claim_text, decision, edited_text, reviewer_note, updated_at))


def load_review_decisions(report_id):
    init_review_db()
    with sqlite3.connect(REVIEW_DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT claim_index, claim_text, decision, edited_text, reviewer_note, updated_at "
            "FROM review_decisions WHERE report_id=? ORDER BY claim_index",
            (report_id,)
        ).fetchall()
    return [dict(row) for row in rows]


def make_pdf_report(
    results, report_id, generated_at, brand_name,
    mode_used, source_filename, guideline_text, review_decisions=None
):
    """Build a PDF audit report in memory and return its bytes."""

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Inkloom AI Audit Report",
        author="Inkloom AI"
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=25,
        textColor=colors.HexColor("#25234a"),
        alignment=TA_CENTER,
        spaceAfter=5 * mm
    ))
    styles.add(ParagraphStyle(
        name="SubTitle",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#55556b"),
        alignment=TA_CENTER,
        spaceAfter=7 * mm
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#25234a"),
        spaceBefore=5 * mm,
        spaceAfter=2 * mm
    ))
    styles.add(ParagraphStyle(
        name="SmallBody",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=12,
        spaceAfter=2 * mm
    ))
    styles.add(ParagraphStyle(
        name="Tiny",
        parent=styles["BodyText"],
        fontSize=7.5,
        leading=10
    ))

    def para(value, style="SmallBody"):
        return Paragraph(escape(str(value)).replace("\n", "<br/>"), styles[style])

    total = len(results)
    counts = {
        status: sum(1 for result in results if result["status"] == status)
        for status in (
            "Evidence found", "Restricted claim",
            "Needs verification", "Unsupported claim"
        )
    }

    story = [
        Paragraph("INKLOOM AI", styles["ReportTitle"]),
        Paragraph(
            "EVIDENCE-GROUNDED BRAND CONTENT AUDIT REPORT",
            styles["SubTitle"]
        ),
        Paragraph("1. Audit information", styles["SectionHeading"])
    ]

    metadata = [
        ["Report ID", report_id, "Generated", generated_at],
        ["Brand", brand_name, "Analysis mode", mode_used],
        ["Guideline source", source_filename or "Manual input", "Claims reviewed", str(total)]
    ]
    metadata_table = Table(metadata, colWidths=[27*mm, 57*mm, 29*mm, 55*mm])
    metadata_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f2f3f8")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#25234a")),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#25234a")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d8d9e2")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6)
    ]))
    story.extend([metadata_table, Paragraph("2. Review summary", styles["SectionHeading"])])

    summary_data = [
        ["Total claims", "Evidence found", "Restricted", "Needs verification", "Unsupported"],
        [
            str(total), str(counts["Evidence found"]),
            str(counts["Restricted claim"]),
            str(counts["Needs verification"]),
            str(counts["Unsupported claim"])
        ]
    ]
    summary_table = Table(summary_data, colWidths=[36*mm, 36*mm, 30*mm, 39*mm, 27*mm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#25234a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f2f3f8")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d8d9e2")),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7)
    ]))
    story.extend([summary_table, Paragraph("3. Claim-by-claim findings", styles["SectionHeading"])])

    for number, result in enumerate(results, start=1):
        block = [
            Paragraph(
                f"<b>Claim {number}: {escape(result.get('status', 'Not classified'))}</b>",
                styles["SmallBody"]
            ),
            para("Original claim: " + result.get("sentence", "")),
            para("Reason: " + result.get("reason", "")),
            para("Risk categories: " + (", ".join(result.get("risks", [])) or "None identified")),
            para("Approved evidence: " + (result.get("evidence") or "No exact approved fact identified.")),
            para("Restriction: " + (result.get("restriction") or "No confirmed restriction recorded.")),
            para("Suggested action: " + result.get("suggestion", ""))
        ]
        for source_ref in result.get("evidence_sources", []):
            block.append(para(f"Approved evidence source: {source_ref.get('source')} — {source_ref.get('location')}", "Tiny"))
        for source_ref in result.get("restriction_sources", []):
            block.append(para(f"Restriction source: {source_ref.get('source')} — {source_ref.get('location')}", "Tiny"))

        approved_matches = result.get("approved_matches", [])
        restriction_matches = result.get("restriction_matches", [])

        if approved_matches:
            block.append(para("Potentially relevant approved facts:", "Tiny"))
            for match in approved_matches:
                block.append(para(
                    f"- {match.get('text', '')} ({match.get('similarity', 0)}% similarity)",
                    "Tiny"
                ))

        if restriction_matches:
            block.append(para("Potentially relevant restrictions:", "Tiny"))
            for match in restriction_matches:
                block.append(para(
                    f"- {match.get('text', '')} ({match.get('similarity', 0)}% similarity)",
                    "Tiny"
                ))

        block.append(Spacer(1, 3 * mm))
        story.append(KeepTogether(block))

    if review_decisions:
        story.append(Paragraph("4. Human review decisions", styles["SectionHeading"]))
        for decision in review_decisions:
            story.append(para(
                f"Claim {decision.get('claim_index', 0) + 1}: {decision.get('decision', 'Pending')} | "
                f"Reviewer note: {decision.get('reviewer_note', '')} | "
                f"Edited text: {decision.get('edited_text', '') or 'Not supplied'}"
            ))

    story.extend([
        Paragraph("5. Source and interpretation notes", styles["SectionHeading"]),
        para(
            "The findings in this report are AI-generated review assistance. "
            "Semantic similarity scores indicate retrieval relevance only; "
            "they do not prove factual support or a policy violation."
        ),
        para(
            "Reviewers should verify each finding against the original brand "
            "guidelines, product documentation and applicable requirements "
            "before approving or publishing marketing content."
        ),
        para(
            "Guideline text supplied for this review: "
            + (guideline_text[:1500] if guideline_text else "Not supplied.")
        ),
        Spacer(1, 4 * mm),
        para(
            "END OF REPORT — Inkloom AI | Human review remains required.",
            "Tiny"
        )
    ])

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(18 * mm, 10 * mm, f"Report {report_id}")
        canvas.drawRightString(
            A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}"
        )
        canvas.restoreState()

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )
    return buffer.getvalue()


def make_text_report(results, report_id, generated_at, brand_name, mode_used, review_decisions=None):
    total = len(results)
    statuses = [
        "Evidence found", "Restricted claim",
        "Needs verification", "Unsupported claim"
    ]
    lines = [
        "INKLOOM AI",
        "EVIDENCE-GROUNDED BRAND CONTENT AUDIT REPORT",
        "=" * 65,
        f"Report ID: {report_id}",
        f"Generated: {generated_at}",
        f"Brand: {brand_name}",
        f"Analysis engine: {mode_used}",
        f"Total claims reviewed: {total}",
        "",
        "SUMMARY"
    ]

    for status in statuses:
        count = sum(1 for item in results if item["status"] == status)
        lines.append(f"{status}: {count}")

    lines.extend(["", "=" * 65, "DETAILED FINDINGS", ""])

    for index, result in enumerate(results, start=1):
        lines.extend([
            f"CLAIM {index}",
            f"Original: {result.get('sentence', '')}",
            f"Status: {result.get('status', '')}",
            f"Risks: {', '.join(result.get('risks', []))}",
            f"Reason: {result.get('reason', '')}",
            f"Approved evidence: {result.get('evidence', '')}",
            f"Restriction: {result.get('restriction', '')}",
            f"Suggested action: {result.get('suggestion', '')}",
            "Potentially relevant approved facts:"
        ])

        for match in result.get("approved_matches", []):
            lines.append(
                f"- {match.get('text', '')} ({match.get('similarity', 0)}% similarity)"
            )

        lines.append("Potentially relevant restrictions:")
        for match in result.get("restriction_matches", []):
            lines.append(
                f"- {match.get('text', '')} ({match.get('similarity', 0)}% similarity)"
            )

        lines.extend(["-" * 65, ""])

    if review_decisions:
        lines.extend(["", "HUMAN REVIEW DECISIONS", ""])
        for decision in review_decisions:
            lines.append(
                f"Claim {decision.get('claim_index', 0) + 1}: {decision.get('decision', 'Pending')} | "
                f"Note: {decision.get('reviewer_note', '')} | "
                f"Edited: {decision.get('edited_text', '')}"
            )

    lines.extend([
        "NOTE: Similarity scores indicate retrieval relevance, not proof.",
        "AI-generated findings require human verification."
    ])
    return "\n".join(lines)


# ==========================================
# MAIN USER INTERFACE
# ==========================================

try:
    init_review_db()
except Exception as database_error:
    st.warning(f"Review history storage is unavailable: {database_error}")

st.title("INKLOOM AI")
st.caption("Evidence-Grounded Brand Content Review Platform")
st.write(
    "Analyze marketing content, identify risky claims and improve "
    "brand compliance using approved sources."
)
st.divider()


# ==========================================
# SOURCE UPLOAD
# ==========================================

st.subheader("1. Brand Guidelines")

uploaded_file = st.file_uploader(
    "Upload approved brand guidelines",
    type=["pdf", "txt"]
)

manual_guidelines = st.text_area(
    "Or paste brand guidelines",
    placeholder="""Brand Name: GlowCare

Approved Facts:
- Contains 2% niacinamide.
- Designed to improve skin appearance.
- Suitable for daily skincare use.

Restrictions:
- Do not claim permanent acne removal.
- Do not guarantee results within a specific number of days.
- Do not make unsupported medical claims.

Brand Tone:
Friendly, informative and trustworthy.""",
    height=180
)

document_text = extract_text(uploaded_file)
guidelines = "\n".join([document_text, manual_guidelines]).strip()

if guidelines:
    approved, restrictions, other = extract_sections(guidelines)
    st.success("Brand sources loaded.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Approved facts", len(approved))
    col2.metric("Restrictions", len(restrictions))
    col3.metric("Other details", len(other))
    with st.expander("View loaded sources"):
        st.write(guidelines)

st.divider()


# ==========================================
# MARKETING CONTENT
# ==========================================

st.subheader("2. Marketing Content")
marketing_content = st.text_area(
    "Enter the content you want to review",
    placeholder="""Our cream permanently removes acne in 3 days.

Our cream contains 2% niacinamide.

Experience healthier-looking skin with GlowCare.""",
    height=180
)


# ==========================================
# ANALYSIS MODE
# ==========================================

st.divider()
st.subheader("3. Analysis Settings")

api_available = bool(get_api_key())
if api_available:
    st.success("Gemini API key detected.")
else:
    st.warning(
        "Gemini API key not found. Basic rule-based analysis will be available."
    )

analysis_mode = st.radio(
    "Choose analysis mode",
    ["AI Semantic Analysis", "Basic Rule-Based Analysis"],
    index=0 if api_available else 1
)


# ==========================================
# ANALYZE BUTTON
# ==========================================

source_records = build_source_records(uploaded_file, manual_guidelines)


if st.button(
    "Analyze Marketing Content",
    type="primary",
    use_container_width=True
):
    if not guidelines:
        st.error("Please upload or enter brand guidelines.")
    elif not marketing_content.strip():
        st.error("Please enter marketing content.")
    else:
        try:
            with st.spinner("Inkloom is analyzing your content..."):
                if analysis_mode == "AI Semantic Analysis":
                    results = ai_review(marketing_content, guidelines, source_records)
                    mode_used = "Gemini AI + Semantic Retrieval"
                else:
                    results = basic_review(marketing_content, guidelines)
                    results = attach_source_traceability(results, source_records)
                    mode_used = "Rule-based engine"

            report_id = create_audit_id()
            generated_at = datetime.now().astimezone().strftime(
                "%Y-%m-%d %H:%M:%S %Z"
            )

            st.session_state["results"] = results
            st.session_state["mode_used"] = mode_used
            st.session_state["review_guidelines"] = guidelines
            st.session_state["report_id"] = report_id
            st.session_state["generated_at"] = generated_at
            st.session_state["brand_name"] = get_brand_name(
                extract_sections(guidelines)[2]
            )
            st.session_state["source_filename"] = (
                uploaded_file.name if uploaded_file else "Manual input"
            )
            st.session_state["source_records"] = source_records

            st.success("Content analysis completed!")
        except Exception as error:
            st.error(f"Analysis failed: {error}")


# ==========================================
# RESULTS DASHBOARD
# ==========================================

if "results" in st.session_state:
    results = st.session_state["results"]

    st.divider()
    st.header("Content Review Dashboard")
    st.caption(
        f"Report ID: {st.session_state['report_id']}  |  "
        f"Generated: {st.session_state['generated_at']}"
    )
    st.caption(f"Analysis engine: {st.session_state['mode_used']}")

    total = len(results)
    evidence_count = sum(r["status"] == "Evidence found" for r in results)
    restricted_count = sum(r["status"] == "Restricted claim" for r in results)
    verification_count = sum(r["status"] == "Needs verification" for r in results)
    unsupported_count = sum(r["status"] == "Unsupported claim" for r in results)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", total)
    c2.metric("Evidence found", evidence_count)
    c3.metric("Restricted", restricted_count)
    c4.metric("Needs verification", verification_count)
    c5.metric("Unsupported", unsupported_count)

    st.divider()

    for index, result in enumerate(results):
        with st.expander(
            f"Claim {index + 1} — {result['status']}",
            expanded=False
        ):
            st.write("**Original content**")
            st.write(result["sentence"])

            st.write("**Status**")
            if result["status"] == "Restricted claim":
                st.error(result["status"])
            elif result["status"] in ("Needs verification", "Unsupported claim"):
                st.warning(result["status"])
            else:
                st.success(result["status"])

            st.write("**Detected risk categories**")
            if result["risks"]:
                for risk in result["risks"]:
                    st.write(f"- {risk}")
            else:
                st.write("No risk categories returned.")

            st.write("**Reason for this finding**")
            st.write(result.get("reason", ""))

            st.write("**Supporting approved evidence**")
            if result["evidence"]:
                st.info(result["evidence"])
            else:
                st.warning("No exact approved supporting fact identified.")

            if result["restriction"]:
                st.write("**Matching brand restriction**")
                st.error(result["restriction"])

            st.divider()
            st.write("**Semantic Evidence Retrieval**")

            approved_matches = result.get("approved_matches", [])
            if approved_matches:
                st.write("Potentially relevant approved facts:")
                for match in approved_matches:
                    st.write(
                        f"- {match['text']} ({match['similarity']}% similarity)"
                    )
                    for source_ref in match.get("sources", []):
                        st.caption(f"Source: {source_ref['source']} — {source_ref['location']}")
            else:
                st.write("No relevant approved facts retrieved.")

            restriction_matches = result.get("restriction_matches", [])
            if restriction_matches:
                st.write("Potentially relevant restrictions:")
                for match in restriction_matches:
                    st.write(
                        f"- {match['text']} ({match['similarity']}% similarity)"
                    )
                    for source_ref in match.get("sources", []):
                        st.caption(f"Source: {source_ref['source']} — {source_ref['location']}")
            else:
                st.write("No relevant restrictions retrieved.")

            st.caption(
                "Similarity scores indicate retrieval relevance, "
                "not proof of compliance or violation."
            )

            st.write("**Suggested improvement**")
            st.write(result["suggestion"])

            st.write("**Evidence traceability**")
            evidence_sources = result.get("evidence_sources", [])
            restriction_sources = result.get("restriction_sources", [])
            if evidence_sources:
                for source_ref in evidence_sources:
                    st.caption(f"Approved fact source: {source_ref['source']} — {source_ref['location']}")
            elif result.get("evidence"):
                st.caption("Evidence text was selected, but its exact source location could not be mapped.")
            if restriction_sources:
                for source_ref in restriction_sources:
                    st.caption(f"Restriction source: {source_ref['source']} — {source_ref['location']}")
            elif result.get("restriction"):
                st.caption("Restriction text was selected, but its exact source location could not be mapped.")
            if not evidence_sources and not restriction_sources:
                st.caption("No exact source location was mapped for this finding.")

            checks = result.get("verification_checks", {})
            if checks:
                with st.expander("Verification guardrail checks"):
                    st.write(f"Exact approved evidence: {'Pass' if checks.get('exact_approved_evidence') else 'Not established'}")
                    st.write(f"Exact restriction match: {'Pass' if checks.get('exact_restriction_match') else 'Not established'}")
                    st.write(f"Risk flags present: {'Yes' if checks.get('risk_flags_present') else 'No'}")
                    st.caption(checks.get("guardrail_note", ""))

            st.write("**Human review decision**")
            current_decisions = load_review_decisions(st.session_state["report_id"])
            existing_decision = next((d for d in current_decisions if d["claim_index"] == index), {})
            with st.form(key=f"review_form_{st.session_state['report_id']}_{index}"):
                decision_value = st.selectbox(
                    "Reviewer decision",
                    ["Pending", "Approved", "Rejected", "Edited"],
                    index=["Pending", "Approved", "Rejected", "Edited"].index(existing_decision.get("decision", "Pending"))
                )
                edited_value = st.text_area(
                    "Edited claim (required when decision is Edited)",
                    value=existing_decision.get("edited_text", "") or "",
                    key=f"edited_claim_{st.session_state['report_id']}_{index}"
                )
                note_value = st.text_area(
                    "Reviewer note",
                    value=existing_decision.get("reviewer_note", "") or "",
                    key=f"review_note_{st.session_state['report_id']}_{index}"
                )
                save_decision = st.form_submit_button("Save human review decision")
            if save_decision:
                if decision_value == "Edited" and not edited_value.strip():
                    st.error("Please enter the edited claim before saving an Edited decision.")
                else:
                    try:
                        save_review_decision(
                            st.session_state["report_id"], index, result["sentence"],
                            decision_value, edited_value.strip(), note_value.strip()
                        )
                        st.success("Human review decision saved.")
                        st.rerun()
                    except Exception as error:
                        st.error(f"Could not save review decision: {error}")

            if st.button(
                "Generate AI Rewrite",
                key=f"rewrite_{index}"
            ):
                if not get_api_key():
                    st.error("Configure your Gemini API key first.")
                else:
                    try:
                        with st.spinner("Generating safer wording..."):
                            rewritten = rewrite_content(
                                result["sentence"],
                                st.session_state["review_guidelines"],
                                result["reason"]
                            )
                        st.session_state[f"rewrite_result_{index}"] = rewritten
                    except Exception as error:
                        st.error(f"Rewrite failed: {error}")

            rewrite_key = f"rewrite_result_{index}"
            if rewrite_key in st.session_state:
                st.success("AI rewrite suggestion")
                st.write(st.session_state[rewrite_key])

    # ======================================
    # PHASE 7: AUDIT REPORT EXPORT
    # ======================================

    st.divider()
    st.subheader("4. Download Audit Report")

    st.write(
        "Your audit report includes report metadata, a review summary, "
        "claim-by-claim findings, evidence retrieval candidates and "
        "human-review notes."
    )

    report_id = st.session_state["report_id"]
    generated_at = st.session_state["generated_at"]
    brand_name = st.session_state["brand_name"]
    mode_used = st.session_state["mode_used"]
    source_filename = st.session_state.get("source_filename", "Manual input")
    guideline_text = st.session_state["review_guidelines"]

    st.code(
        f"Report ID: {report_id}\n"
        f"Brand: {brand_name}\n"
        f"Generated: {generated_at}",
        language="text"
    )

    try:
        review_decisions = load_review_decisions(report_id)
        st.caption(f"Human review decisions saved: {len(review_decisions)} / {len(results)}")
        pdf_bytes = make_pdf_report(
            results=results,
            report_id=report_id,
            generated_at=generated_at,
            brand_name=brand_name,
            mode_used=mode_used,
            source_filename=source_filename,
            guideline_text=guideline_text,
            review_decisions=review_decisions
        )

        col_pdf, col_txt = st.columns(2)
        col_pdf.download_button(
            "Download PDF Audit Report",
            data=pdf_bytes,
            file_name=f"inkloom_audit_{report_id}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        text_report = make_text_report(
            results, report_id, generated_at, brand_name, mode_used,
            review_decisions=review_decisions
        )
        col_txt.download_button(
            "Download TXT Report",
            data=text_report,
            file_name=f"inkloom_audit_{report_id}.txt",
            mime="text/plain",
            use_container_width=True
        )
        if review_decisions:
            st.download_button(
                "Export Human Review History (JSON)",
                data=json.dumps(review_decisions, indent=2, ensure_ascii=False),
                file_name=f"inkloom_review_history_{report_id}.json",
                mime="application/json"
            )
        st.caption("Review decisions are stored in the app's local SQLite database. On Streamlit Community Cloud, local storage may reset when the app is restarted or redeployed; download the review history to retain a copy.")
    except Exception as error:
        st.error(f"Could not generate the PDF report: {error}")


# ==========================================
# FOOTER
# ==========================================

st.divider()
st.caption("INKLOOM AI | Evidence-Grounded Brand Content Review")
st.caption(
    "AI-generated findings and suggestions require human verification. "
    "Similarity or AI assessments do not independently prove the factual "
    "accuracy of a marketing claim. Reviewer decisions remain the final "
    "human-controlled step."
)
