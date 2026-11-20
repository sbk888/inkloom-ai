"""Deterministic guardrails for Inkloom AI claim-review outputs."""

VALID_STATUSES = {
    "Restricted claim",
    "Needs verification",
    "Evidence found",
    "Unsupported claim",
}


def enforce_review_guardrails(result, approved_facts, restrictions):
    """Validate model output against exact source strings and risk signals.

    This does not decide legal compliance. It prevents unsupported model labels
    from being treated as verified evidence or confirmed restriction matches.
    """
    item = dict(result)
    approved = {str(x).strip() for x in approved_facts if str(x).strip()}
    restricted = {str(x).strip() for x in restrictions if str(x).strip()}

    evidence = str(item.get("evidence", "")).strip()
    restriction = str(item.get("restriction", "")).strip()
    status = str(item.get("status", "Needs verification")).strip()
    risks = item.get("risks", item.get("risk_categories", []))
    if not isinstance(risks, list):
        risks = []
    risks = [str(x) for x in risks if str(x).strip()]

    evidence_valid = evidence in approved
    restriction_valid = restriction in restricted
    if not evidence_valid:
        evidence = ""
    if not restriction_valid:
        restriction = ""

    if restriction_valid:
        status = "Restricted claim"
    elif status == "Restricted claim":
        # A restriction must be explicitly grounded in a supplied source.
        status = "Needs verification"
    elif status == "Evidence found" and not evidence_valid:
        status = "Unsupported claim"
    elif status == "Evidence found" and risks:
        # A matching fact does not neutralize an absolute/medical/quantified risk.
        status = "Needs verification"
    elif status not in VALID_STATUSES:
        status = "Needs verification"

    item.update({
        "status": status,
        "risks": risks,
        "evidence": evidence,
        "restriction": restriction,
        "verification_checks": {
            "exact_approved_evidence": evidence_valid,
            "exact_restriction_match": restriction_valid,
            "risk_flags_present": bool(risks),
            "guardrail_note": "Automated checks support review; they do not establish legal or regulatory compliance."
        }
    })
    return item
