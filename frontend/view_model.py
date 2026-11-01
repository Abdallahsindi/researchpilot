"""Pure display helpers for the Streamlit UI (no Streamlit or HTTP imports, easy to test).

Labels map the backend's actual status values. Anything unrecognised is shown as-is in
grey and is never presented as verified.
"""

SOURCE_TYPES = {
    "document": {"label": "Uploaded PDF", "color": "blue"},
    "web": {"label": "Web page", "color": "green"},
    "demo": {"label": "Synthetic demo", "color": "orange"},
}

CLAIM_STATUSES = {
    "verified": {"label": "Verified", "color": "green"},
    "unverified": {"label": "Unverified", "color": "orange"},
    "unsupported": {"label": "Unsupported", "color": "red"},
    "contradicted": {"label": "Contradicted", "color": "red"},
    "invalid_reference": {"label": "Invalid reference", "color": "red"},
}

ANALYSIS_STATUSES = {
    "completed": (
        "success",
        "Verified findings available",
        "At least one claim passed verification. Only claims marked Verified are evidence-backed.",
    ),
    "no_verified_findings": (
        "warning",
        "No verified findings",
        "The Analyst proposed claims, but none passed verification. Nothing here should be "
        "treated as an evidence-backed answer.",
    ),
    "insufficient_evidence": (
        "warning",
        "Insufficient evidence",
        "The collected sources do not answer this question, so no findings are presented.",
    ),
    "failed": (
        "error",
        "Analysis failed",
        "The model was unavailable, failed, timed out, or produced unusable output. The "
        "collected sources are still shown.",
    ),
}

RESEARCH_STATUSES = {
    "completed": "Evidence collected",
    "insufficient_evidence": "Too little evidence collected",
    "failed": "No evidence could be collected",
}


def source_type_info(source_type: str) -> dict:
    return SOURCE_TYPES.get(source_type, {"label": source_type or "unknown", "color": "gray"})


def claim_status_info(status: str) -> dict:
    return CLAIM_STATUSES.get(status, {"label": status or "unknown", "color": "gray"})


def analysis_status_info(status: str) -> tuple[str, str, str]:
    """(message kind, short label, explanation)."""
    return ANALYSIS_STATUSES.get(status, ("info", status, f"Analysis status: {status}"))


def research_status_label(status: str) -> str:
    return RESEARCH_STATUSES.get(status, status)


def source_reference(source: dict) -> str:
    """Where a source can be checked: PDF file and page, or the URL."""
    if source.get("source_type") == "document":
        return f"{source.get('filename')}, page {source.get('page_number')}"
    return source.get("url") or "(no URL)"


def evidence_mix(run: dict) -> dict[str, int]:
    """Number of sources of each type, in display order."""
    counts = {"document": 0, "web": 0, "demo": 0}
    for source in run.get("sources", []):
        counts[source["source_type"]] = counts.get(source["source_type"], 0) + 1
    return counts


def demo_evidence_notice(run: dict) -> str | None:
    """Explain Demo mode without implying a PDF-only run used synthetic evidence."""
    if run.get("mode") != "demo":
        return None
    mix = evidence_mix(run)
    if mix["demo"]:
        return None
    if mix["document"]:
        return (
            "Search is configured for Demo mode. This run used no synthetic demo sources; "
            "it relies on uploaded PDF evidence."
        )
    return "Search is configured for Demo mode. This run used no synthetic demo sources."


def run_evidence_info(run: dict) -> dict[str, str]:
    """Label a finished run by the evidence it actually used, not by search configuration."""
    mix = evidence_mix(run)
    if mix["demo"]:
        return {
            "label": "Synthetic demo evidence",
            "color": "orange",
            "description": (
                f"This run includes {mix['demo']} synthetic Demo-mode page(s). Those entries "
                "are sample data, not live web research, and are not treated as real evidence."
            ),
        }
    if mix["web"]:
        label = "Documents + live web evidence" if mix["document"] else "Live web research"
        description = f"This run used {mix['web']} live web page(s)"
        if mix["document"]:
            description += f" and {mix['document']} uploaded-PDF passage(s)."
        else:
            description += "."
        return {"label": label, "color": "green", "description": description}
    if mix["document"]:
        return {
            "label": "Local PDF Research",
            "color": "blue",
            "description": (
                f"This run used {mix['document']} uploaded-PDF passage(s) and no synthetic "
                "demo sources."
            ),
        }
    return {
        "label": "No evidence",
        "color": "gray",
        "description": "This run did not collect any sources.",
    }


def final_analyst_output(run: dict) -> dict | None:
    """The Analyst output that was verified (the last attempt that parsed)."""
    analysis = run.get("analysis") or {}
    for attempt in reversed(analysis.get("attempts", [])):
        if attempt.get("output"):
            return attempt["output"]
    return None


def claim_counts(run: dict) -> dict[str, int]:
    verification = (run.get("analysis") or {}).get("verification") or {}
    counts: dict[str, int] = {}
    for claim in verification.get("claims", []):
        counts[claim["status"]] = counts.get(claim["status"], 0) + 1
    return counts


def cited_source_ids(run: dict) -> set[str]:
    verification = (run.get("analysis") or {}).get("verification") or {}
    return {sid for c in verification.get("claims", []) for sid in c["cited_source_ids"]}


def report_filename(run: dict) -> str:
    return f"researchpilot_report_{str(run.get('run_id', 'run'))[:8]}.md"


def search_scope(document_count: int, health: dict | None) -> str:
    """One sentence describing what a research run will search."""
    docs = f"{document_count} uploaded document{'s' if document_count != 1 else ''}"
    if health is None:
        return f"{docs} (backend not connected)"
    if health.get("mode") == "demo":
        return f"{docs} + web search in Demo mode (synthetic sample pages, not the real web)"
    return f"{docs} + live web search ({health.get('search_provider')})"
