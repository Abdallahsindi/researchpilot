"""Report Writer agent: renders the final Markdown report from verified results.

Deterministic on purpose: formatting validated data needs no model call, and a renderer
cannot add claims, numbers, links, or confidence scores that the earlier stages did not
produce. Only `verified` claims appear under "Key findings"; everything else is labelled.
"""

from app.schemas.analysis import (
    AnalysisResult,
    AnalysisStatus,
    ClaimVerification,
    VerificationStatus,
)
from app.schemas.research import ResearchRequest, Source, SourceType

REJECTED = (
    VerificationStatus.UNSUPPORTED,
    VerificationStatus.INVALID_REFERENCE,
    VerificationStatus.CONTRADICTED,
)


def _missing_criteria(request: ResearchRequest, verified: list[ClaimVerification]) -> list[str]:
    """Requested criteria that no verified finding addresses.

    A criterion is covered only by a verified claim labelled with that criterion. Unverified
    observations and claims without a criterion label cannot support a decision, so they do
    not count here.
    """
    return [
        criterion
        for criterion in request.criteria
        if not any(claim.criterion == criterion for claim in verified)
    ]


def _join_criteria(criteria: list[str]) -> str:
    if len(criteria) == 1:
        return criteria[0]
    if len(criteria) == 2:
        return f"{criteria[0]} and {criteria[1]}"
    return ", ".join(criteria[:-1]) + f", and {criteria[-1]}"


def _missing_evidence_recommendation(missing: list[str]) -> str:
    """Conservative recommendation when one or more decision criteria are unsupported.

    This is deterministic text, not a model judgement: it only states what the verification
    results do and do not establish. The Analyst's own recommendation is withheld because it
    may not mention criteria that have no verified evidence.
    """
    unsupported = _join_criteria(missing)
    return (
        "**Insufficient verified evidence for a final decision.** The verified findings do "
        f"not cover {unsupported}. Further evidence is needed for {unsupported} before the "
        "decision can be made. The absence of verified evidence must not be read as "
        "satisfactory performance on these criteria."
    )


def source_reference(source: Source) -> str:
    if source.source_type is SourceType.DOCUMENT:
        return f"{source.id}: {source.filename}, p. {source.page_number}"
    if source.source_type is SourceType.DEMO:
        return f"{source.id}: synthetic demo entry"
    return f"{source.id}: {source.domain}"


def _refs(claim: ClaimVerification, sources: dict[str, Source]) -> str:
    refs = "; ".join(source_reference(sources[i]) for i in claim.cited_source_ids)
    return f" [{refs}]" if refs else ""


def _type_label(source: Source) -> str:
    return {
        SourceType.DOCUMENT: "Uploaded PDF",
        SourceType.WEB: "Web",
        SourceType.DEMO: "SYNTHETIC (demo)",
    }[source.source_type]


def _source_cell(source: Source) -> str:
    if source.source_type is SourceType.DOCUMENT:
        return f"{source.filename}, page {source.page_number} (`{source.chunk_id}`)"
    return f"[{source.title}]({source.url})"


def render_report(
    request: ResearchRequest,
    sources: list[Source],
    analysis: AnalysisResult,
    *,
    run_warnings: list[str],
    demo_mode: bool,
) -> str:
    by_id = {s.id: s for s in sources}
    verification = analysis.verification
    claims = verification.claims if verification else []
    verified = [c for c in claims if c.status is VerificationStatus.VERIFIED]
    unverified = [c for c in claims if c.status is VerificationStatus.UNVERIFIED]
    rejected = [c for c in claims if c.status in REJECTED]
    output = analysis.final_output
    lines: list[str] = [f"# Research report: {request.question}", ""]

    status_line = f"**Analysis status:** `{analysis.status.value}`"
    if claims:
        status_line += (
            f" · {len(verified)} verified, {len(unverified)} unverified, "
            f"{len(rejected)} rejected claim(s)"
        )
    lines.append(status_line)
    if analysis.model_id:
        lines.append(f"**Analyst model:** `{analysis.model_id}` (local, greedy decoding)")
    demo_sources = [s for s in sources if s.source_type is SourceType.DEMO]
    if demo_sources:
        lines.append(
            "> ⚠️ This run includes synthetic sources from Demo mode. Entries marked "
            "SYNTHETIC are invented sample data, not real evidence."
        )
    elif demo_mode:
        if any(s.source_type is SourceType.DOCUMENT for s in sources):
            lines.append(
                "> ℹ️ Search is configured for Demo mode. This run used no synthetic demo "
                "sources; it relies on uploaded PDF evidence."
            )
        else:
            lines.append(
                "> ℹ️ Search is configured for Demo mode. This run used no synthetic demo sources."
            )

    lines += ["", "## Research question and objectives", "", f"**Question:** {request.question}"]
    for label, values in (
        ("Goals", request.goals),
        ("Criteria", request.criteria),
        ("Constraints", request.constraints),
    ):
        if values:
            lines.append(f"**{label}:** " + "; ".join(values))

    lines += ["", "## Executive summary", ""]
    if analysis.status is AnalysisStatus.FAILED:
        lines.append("The analysis could not be completed, so no findings are presented.")
        lines += [f"- {e}" for e in analysis.errors]
    elif analysis.status is AnalysisStatus.INSUFFICIENT_EVIDENCE:
        lines.append(
            "The collected evidence is not sufficient to answer this question, so no "
            "findings are presented."
        )
        if output and output.limitations:
            lines += [f"- Analyst note: {item}" for item in output.limitations]
    elif not verified:
        lines.append(
            "The Analyst proposed findings, but none passed verification. Nothing in this "
            "report should be treated as an evidence-backed answer."
        )
    else:
        answer_check = verification.answer_check if verification else None
        if output and answer_check and answer_check.code_checks_passed:
            lines += [output.answer, ""]
            lines.append(
                "*Summary written by the Analyst model. Its wording and numbers match the "
                "sources of the verified findings, but it was not separately reviewed.*"
            )
        else:
            lines.append(f"Based on {len(verified)} verified finding(s):")
            lines += [f"- {c.claim}" for c in verified[:3]]
            lines += [
                "",
                "*The Analyst's own summary is omitted because its wording did not match the "
                "verified sources closely enough.*",
            ]

    lines += ["", "## Key findings (verified)", ""]
    if verified:
        for n, claim in enumerate(verified, start=1):
            marker = " *(synthetic demo evidence only)*" if claim.synthetic_only else ""
            lines.append(f"{n}. {claim.claim}{_refs(claim, by_id)}{marker}")
    else:
        lines.append("No finding was verified.")

    if unverified:
        lines += [
            "",
            "## Unverified observations",
            "",
            "*Not confirmed by verification. Do not treat these as facts.*",
            "",
        ]
        for claim in unverified:
            lines.append(f"- {claim.claim}{_refs(claim, by_id)} — {claim.explanations[-1]}")

    if request.criteria:
        lines += ["", "## Comparisons and trade-offs", ""]
        for criterion in request.criteria:
            relevant = [c for c in verified + unverified if c.criterion == criterion]
            lines.append(f"**{criterion}**")
            if not relevant:
                lines.append("- No verified evidence for this criterion.")
            for claim in relevant:
                label = "" if claim.status is VerificationStatus.VERIFIED else " *(unverified)*"
                lines.append(f"- {claim.claim}{_refs(claim, by_id)}{label}")
            lines.append("")

    lines += ["", "## Recommendation", ""]
    rec_check = verification.recommendation_check if verification else None
    missing = _missing_criteria(request, verified)
    if not verified:
        lines.append(
            "No recommendation is made: there are not enough verified findings to support one."
        )
    elif missing:
        lines.append(_missing_evidence_recommendation(missing))
        lines.append("")
        lines.append(
            "*This recommendation is generated from the verification results, not written by "
            "the Analyst model. The model's own recommendation is withheld because it did not "
            "cover every requested criterion with verified evidence.*"
        )
    elif output and output.recommendation and rec_check and rec_check.status not in REJECTED:
        lines.append(output.recommendation)
        lines.append("")
        basis = "; ".join(request.criteria + request.constraints) or "the research question"
        lines.append(
            f"*A judgement by the Analyst model, not a verified fact. It is based on the "
            f"verified findings above and the stated criteria and constraints ({basis}).*"
        )
    else:
        lines.append(
            "No recommendation is made: the Analyst did not provide a recommendation that "
            "passed the checks, even though every requested criterion has verified evidence."
        )
    limitations = output.limitations if output else []
    if limitations:
        lines += ["", "**Limitations noted by the Analyst** (model-written):"]
        lines += [f"- {item}" for item in limitations]

    lines += ["", "## Contradictory or insufficient evidence", ""]
    issues = [
        f"- **{c.status.value}:** {c.claim}{_refs(c, by_id)} — {' '.join(c.explanations)}"
        for c in rejected
    ]
    issues += [
        f"- No verified evidence for criterion **{criterion}**."
        for criterion in request.criteria
        if not any(c.criterion == criterion for c in verified)
    ]
    issues += [f"- {w}" for w in run_warnings]
    lines += issues or ["None identified."]

    lines += [
        "",
        "## Sources",
        "",
        "| ID | Type | Reference | Cited in findings |",
        "|---|---|---|---|",
    ]
    cited = {i for c in claims for i in c.cited_source_ids}
    for source in sources:
        cited_mark = "yes" if source.id in cited else ""
        lines.append(
            f"| {source.id} | {_type_label(source)} | {_source_cell(source)} | {cited_mark} |"
        )

    lines += ["", "## Verification status", ""]
    if claims:
        lines += ["| # | Status | Claim | Explanation |", "|---|---|---|---|"]
        for n, claim in enumerate(claims, start=1):
            text = claim.claim.replace("|", "\\|")
            why = " ".join(claim.explanations).replace("|", "\\|")
            lines.append(f"| {n} | `{claim.status.value}` | {text} | {why} |")
        lines.append("")
    if verification is None:
        lines.append("No claims reached the verification stage.")
        return "\n".join(lines).rstrip() + "\n"
    method = "with" if verification.llm_review_enabled else "without"
    lines.append(
        f"Verification ran {method} LLM review. Code checks confirm that cited IDs exist in "
        "this run, document citations match stored chunk metadata, and numbers and wording "
        "appear in the cited text. `verified` also requires the LLM review to judge the claim "
        "supported. These checks reduce, but do not eliminate, the risk of errors: read the "
        "cited sources before relying on a finding."
    )
    return "\n".join(lines).rstrip() + "\n"
