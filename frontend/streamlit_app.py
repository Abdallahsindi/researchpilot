"""ResearchPilot web interface.

    uv run streamlit run frontend/streamlit_app.py

A thin client for the FastAPI backend (RESEARCHPILOT_API_URL, default http://localhost:8000).
Every result shown here comes from the API: PDFs are registered with POST /documents,
research runs with POST /research (analyze=true), and the saved run is re-read from
GET /research/{run_id}. The UI never fabricates results or progress.
"""

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontend.api_client import (  # noqa: E402
    ApiError,
    ResearchPilotClient,
    api_url_from_env,
    build_research_payload,
)
from frontend.view_model import (  # noqa: E402
    analysis_status_info,
    cited_source_ids,
    claim_counts,
    claim_status_info,
    demo_evidence_notice,
    evidence_mix,
    final_analyst_output,
    report_filename,
    research_status_label,
    run_evidence_info,
    search_scope,
    source_reference,
    source_type_info,
)

EXAMPLE_CASES = ROOT / "sample_data" / "pdfs" / "researchpilot_test_report_cases.json"
EXAMPLE_PDF = ROOT / "sample_data" / "pdfs" / "researchpilot_test_report.pdf"

st.set_page_config(page_title="ResearchPilot", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      .rp-subtitle {color: #4b5563; margin-top: -0.6rem; margin-bottom: 0.4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

state = st.session_state
state.setdefault("uploads", {})  # browser file key -> {"status", "info"|"error", "created"}
state.setdefault("documents", {})  # document_id -> DocumentInfo registered in this session
state.setdefault("run", None)  # last run, as returned by GET /research/{run_id}
state.setdefault("run_error", None)

client = ResearchPilotClient(api_url_from_env())


def file_key(file) -> str:
    """Identifies a browser-selected file across reruns."""
    return getattr(file, "file_id", None) or f"{file.name}:{file.size}"


def upload_pending(files) -> list[str]:
    """Upload every selected file that is not registered yet. Returns failure messages."""
    failures = []
    for file in files:
        key = file_key(file)
        if state.uploads.get(key, {}).get("status") == "uploaded":
            continue
        with st.spinner(f"Uploading {file.name} to the backend…"):
            try:
                info, created = client.upload_document(file.name, file.getvalue())
            except ApiError as exc:
                state.uploads[key] = {"status": "failed", "error": exc.message}
                failures.append(f"{file.name}: {exc.message}")
                continue
        state.uploads[key] = {"status": "uploaded", "info": info, "created": created}
        state.documents[info["document_id"]] = info
        state[f"use_{info['document_id']}"] = True  # newly registered PDFs are searched
    return failures


def apply_example_document_selection(health: dict | None) -> str | None:
    """Select the bundled sample PDF after the example-question button is clicked.

    Runs before the document checkboxes are created because Streamlit does not allow a
    widget's session value to be changed after that widget has rendered. This keeps the
    example question from being sent against an unrelated PDF that happened to be checked.
    """
    if not state.get("select_example_document"):
        return None
    state["select_example_document"] = False
    if health is None:
        return "The backend is not reachable, so the sample PDF cannot be selected."
    try:
        documents = client.list_documents()
        selected = next((d for d in documents if d["filename"] == EXAMPLE_PDF.name), None)
        if selected is None and EXAMPLE_PDF.exists():
            selected, created = client.upload_document(EXAMPLE_PDF.name, EXAMPLE_PDF.read_bytes())
            if created:
                state.uploads[f"sample:{EXAMPLE_PDF.name}"] = {
                    "status": "uploaded",
                    "info": selected,
                    "created": True,
                }
        if selected is None:
            return f"Could not find or upload the sample PDF `{EXAMPLE_PDF.name}`."
    except (ApiError, OSError) as exc:
        return f"Could not register the sample PDF: {exc}"

    selected_id = selected["document_id"]
    state.documents[selected_id] = selected
    all_ids = {d["document_id"] for d in documents} | set(state.documents) | {selected_id}
    for document_id in all_ids:
        state[f"use_{document_id}"] = document_id == selected_id
    return None


# ---------------------------------------------------------------- header and status
st.title("ResearchPilot")
st.markdown(
    '<p class="rp-subtitle">Evidence-grounded research for uploaded PDFs and web sources. '
    "A local language model analyses the retrieved passages, code checks every citation, and "
    "the report keeps verified findings separate from unverified observations.</p>",
    unsafe_allow_html=True,
)
try:
    health = client.health()
except ApiError as exc:
    health = None
    st.error(f"Backend not reachable. {exc.message}")

if health:
    status_cols = st.columns([1, 2, 3])
    with status_cols[0]:
        st.badge("Backend connected", color="green")
    with status_cols[1]:
        if health.get("mode") == "demo":
            st.badge("Search configured: Demo corpus (offline)", color="gray")
        else:
            st.badge(f"Search configured: {health.get('search_provider')} (live)", color="green")
    status_cols[2].caption(f"API {client.base_url}")

example_selection_error = apply_example_document_selection(health)
if example_selection_error:
    st.error(example_selection_error)

with st.sidebar:
    st.subheader("Open a saved run")
    run_id_input = st.text_input("Run ID", placeholder="paste a run_id")
    if st.button("Load run", disabled=not run_id_input.strip(), use_container_width=True):
        try:
            state.run, state.run_error = client.get_run(run_id_input.strip()), None
        except ApiError as exc:
            state.run_error = exc.message

    st.divider()
    st.subheader("Search configuration")
    with st.expander("What will this run search?", expanded=False):
        if health is None:
            st.caption("Search settings are unavailable while the backend is offline.")
        elif health.get("mode") == "demo":
            st.caption(
                "Demo mode: web results come from a small offline set of synthetic sample pages "
                "(demo.example), not from the real web, only when a run uses web search. Evidence "
                "from uploaded PDFs is real document text, so a PDF-only run contains no "
                "synthetic sources and is labelled Local PDF Research. Live web search needs "
                "DEMO_MODE=false and a TAVILY_API_KEY on the backend."
            )
        else:
            st.caption(
                "Live web search is configured. The sources tab still labels every source by "
                "the evidence the run actually used."
            )
        st.caption(
            "The configured search provider is separate from the evidence shown for each run."
        )

    st.divider()
    st.subheader("Sample data")
    st.caption(
        "PDF: `sample_data/pdfs/researchpilot_test_report.pdf` (synthetic educational test "
        "data). Questions with expected pages: "
        "`sample_data/pdfs/researchpilot_test_questions.md`."
    )

# ---------------------------------------------------------------- 1. documents | 2. question
left, right = st.columns([5, 7], gap="large")

with left:
    st.subheader("1. Documents")
    files = st.file_uploader(
        "Select PDF files", type=["pdf"], accept_multiple_files=True, key="pdf_files"
    )
    files = files or []
    pending = [f for f in files if state.uploads.get(file_key(f), {}).get("status") != "uploaded"]
    for f in files:
        record = state.uploads.get(file_key(f))
        with st.container(border=True):
            st.markdown(f"**{f.name}** · {f.size / 1024:.0f} KB")
            if record is None:
                st.caption("Selected in the browser — not uploaded yet")
            elif record["status"] == "uploaded":
                info = record["info"]
                note = "" if record["created"] else " (already in the library)"
                st.markdown(
                    f":green[Uploaded{note}] · `{info['document_id']}` · "
                    f"{info['page_count']} pages · {info['chunk_count']} chunks"
                )
                for warning in info["warnings"]:
                    st.caption(f"Note: {warning}")
            else:
                st.markdown(f":red[Upload failed:] {record['error']}")
    if st.button(
        f"Upload {len(pending)} selected file{'s' if len(pending) != 1 else ''}",
        disabled=not pending or health is None,
        use_container_width=True,
    ):
        for failure in upload_pending(pending):
            st.error(failure)
        st.rerun()

    library = dict(state.documents)
    if health is not None:
        try:
            library.update({d["document_id"]: d for d in client.list_documents()})
        except ApiError as exc:
            st.warning(f"Document library unavailable: {exc.message}")

    st.markdown(f"**Registered documents ({len(library)})**")
    if library:
        for doc_id, info in library.items():
            st.checkbox(
                f"{info['filename']} — {info['page_count']} pages",
                key=f"use_{doc_id}",
                help=f"document_id {doc_id}",
            )
    else:
        st.caption("No documents are registered with the backend yet.")

    selected_ids = [doc_id for doc_id in library if state.get(f"use_{doc_id}")]
    st.caption(
        f"{len(selected_ids)} of {len(library)} registered document(s) will be searched. "
        "Only documents listed here have been accepted by the backend; a file selected in "
        "the browser is not uploaded until the upload finishes."
    )
    if library and not selected_ids:
        st.warning(
            "No registered PDF is selected, so the run will not search any uploaded document."
        )

with right:
    st.subheader("2. Research question")
    if EXAMPLE_CASES.exists() and st.button("Load example question for the sample PDF"):
        example = json.loads(EXAMPLE_CASES.read_text())["ui_demo_request"]
        state["question"] = example["question"]
        state["criteria"] = "\n".join(example["criteria"])
        state["constraints"] = "\n".join(example["constraints"])
        # Select/register the sample PDF on the next run, before the checkboxes render.
        state["select_example_document"] = True
        st.rerun()
    with st.form("research_form", border=False):
        question = st.text_area(
            "Research question (required)",
            key="question",
            placeholder="e.g. Should the college install the same battery system at its "
            "second campus?",
            height=100,
        )
        with st.expander("Analysis settings (optional)"):
            criteria = st.text_area(
                "Comparison criteria, one per line",
                key="criteria",
                placeholder="efficiency\nsafety\ncost",
            )
            constraints = st.text_area(
                "Constraints, one per line",
                key="constraints",
                placeholder="limited capital budget",
            )
            goals = st.text_area("Goals, one per line", key="goals")
        pending_note = (
            f" {len(pending)} selected file(s) not uploaded yet will be uploaded first."
            if pending
            else ""
        )
        st.caption(f"This run will search {search_scope(len(selected_ids), health)}.{pending_note}")
        submitted = st.form_submit_button("Analyze", type="primary", use_container_width=True)

if submitted:
    payload, problems = build_research_payload(
        question, goals=goals, constraints=constraints, criteria=criteria, analyze=True
    )
    if problems:
        for problem in problems:
            st.error(problem)
    elif health is None:
        st.error("The backend is not reachable, so the research request was not sent.")
    else:
        failures = upload_pending(pending)
        if failures:
            for failure in failures:
                st.error(f"Upload failed, so the research was not started. {failure}")
        else:
            # Re-read the selection after any uploads above, then send those registered IDs.
            known = library | state.documents
            payload["document_ids"] = [d for d in known if state.get(f"use_{d}")]
            with st.spinner(
                "Running research and analysis. This waits for the API to return the finished "
                "run; the first analysis also loads the local model and can take minutes."
            ):
                try:
                    created = client.research(payload)
                    state.run, state.run_error = client.get_run(created["run_id"]), None
                except ApiError as exc:
                    state.run, state.run_error = None, exc.message

if state.run_error:
    st.error(f"Research request failed: {state.run_error}")


# ---------------------------------------------------------------- results
def render_retrieval_diagnostics(run: dict) -> None:
    index = run.get("document_index")
    searches = run.get("document_retrieval") or []
    if not index and not searches:
        return
    with st.expander("Retrieval diagnostics (index, selected PDFs, similarity scores)"):
        if index:
            st.markdown(
                f"**Indexed chunks:** {index['indexed_chunks']} · "
                f"**Selected documents:** {len(index['selected_documents'])}"
            )
            for doc in index["selected_documents"]:
                st.caption(
                    f"`{doc['document_id']}` · {doc['filename']} · "
                    f"{doc['page_count']} pages · {doc['chunk_count']} chunks indexed"
                )
        for search in searches:
            st.markdown(f"**{search['query_id']}** · {search['query']}")
            details = [
                f"threshold {search['min_score']:.2f}",
                f"{search['accepted']} accepted",
                f"{search['rejected']} rejected",
            ]
            if search.get("fallback_used"):
                details.append("calibrated fallback used")
            st.caption(" · ".join(details))
            if search.get("reason"):
                st.caption(search["reason"])
            for candidate in search.get("candidates", []):
                outcome = "kept" if candidate["accepted"] else "rejected"
                st.markdown(
                    f"- {candidate['filename']}, page {candidate['page_number']} · "
                    f"similarity {candidate['score']:.3f} · {outcome}"
                )


def render_sources(run: dict) -> None:
    if not run["sources"]:
        st.info("No sources were collected.")
        return
    cited = cited_source_ids(run)
    groups = [
        ("document", "Uploaded PDF passages"),
        ("web", "Live web pages"),
        ("demo", "Synthetic demo pages (not real evidence)"),
    ]
    for source_type, heading in groups:
        sources = [s for s in run["sources"] if s["source_type"] == source_type]
        if not sources:
            continue
        st.markdown(f"**{heading}**")
        for s in sources:
            with st.container(border=True):
                top = st.columns([6, 2])
                top[0].markdown(f"**{s['id']}** · {s['title']}")
                with top[1]:
                    info = source_type_info(s["source_type"])
                    st.badge(info["label"], color=info["color"])
                if s["source_type"] == "document":
                    st.caption(f"{source_reference(s)} · chunk `{s['chunk_id']}`")
                elif s.get("url"):
                    st.caption(s["url"])
                details = [f"similarity {s['score']:.2f}"] if s.get("score") is not None else []
                if s["id"] in cited:
                    details.append("cited by the Analyst")
                if details:
                    st.caption(" · ".join(details))
                with st.expander("Text passed to the agents"):
                    st.write(s["snippet"])


def render_analysis(run: dict) -> None:
    analysis = run.get("analysis")
    if analysis is None:
        st.info("This run was made without analysis (analyze=false).")
        return
    plan, metrics = run["plan"], run["metrics"]
    cols = st.columns(4)
    cols[0].metric("Planner queries", len(plan["queries"]))
    cols[1].metric("Sources collected", metrics["sources_returned"])
    cols[2].metric("Analyst attempts", len(analysis["attempts"]))
    cols[3].metric("LLM calls", metrics["llm_calls"])
    with st.expander("Planned search queries"):
        for q in plan["queries"]:
            st.markdown(f"- **{q['id']}** {q['text']} — {q['purpose']}")

    output = final_analyst_output(run)
    if output:
        st.markdown("**Analyst output** (written by the model; see Verification for what holds up)")
        st.caption(
            "The original model output is shown for inspection. The final report applies the "
            "verification rules and withholds this recommendation when a requested criterion "
            "has no verified evidence."
        )
        st.markdown(f"*Answer (unverified):* {output['answer']}")
        if output.get("recommendation"):
            st.markdown(f"*Recommendation (a judgement, not a fact):* {output['recommendation']}")
        if output.get("limitations"):
            st.markdown("*Limitations stated by the Analyst:*")
            for item in output["limitations"]:
                st.markdown(f"- {item}")
    for attempt in analysis["attempts"]:
        title = f"Analyst attempt {attempt['attempt']}"
        if attempt["parse_error"]:
            title += " — rejected (malformed output)"
        elif attempt["feedback"]:
            title += " — revision after Verifier feedback"
        with st.expander(title):
            if attempt["feedback"]:
                st.markdown("Feedback given to the Analyst:")
                st.code(attempt["feedback"], language=None)
            if attempt["parse_error"]:
                st.error(attempt["parse_error"])
            if attempt["invented_source_ids"]:
                invented = ", ".join(attempt["invented_source_ids"])
                st.warning(f"Invented source IDs removed: {invented}")
            st.code(attempt["raw_output"] or "(no output)", language="json")
            st.caption(
                f"{attempt['duration_ms']:.0f} ms · {attempt['prompt_tokens']} prompt tokens · "
                f"{attempt['completion_tokens']} generated tokens"
            )


def render_verification(run: dict) -> None:
    verification = (run.get("analysis") or {}).get("verification")
    if not verification:
        st.info("No claims reached the Verifier in this run.")
        return
    counts = claim_counts(run)
    cols = st.columns(max(len(counts), 1))
    for col, (status, n) in zip(cols, sorted(counts.items()), strict=False):
        col.metric(claim_status_info(status)["label"], n)
    review = "with" if verification["llm_review_enabled"] else "without"
    st.caption(
        "A valid citation is not a verified fact. Verified means the cited source exists in "
        "this run, the PDF page and text match the stored chunk, numbers and wording appear in "
        "the source, and the LLM review judged the claim supported. This run was verified "
        f"{review} LLM review."
    )
    sources = {s["id"]: s for s in run["sources"]}
    for n, claim in enumerate(verification["claims"], start=1):
        info = claim_status_info(claim["status"])
        with st.container(border=True):
            head = st.columns([6, 2])
            head[0].markdown(f"**{n}.** {claim['claim']}")
            with head[1]:
                st.badge(info["label"], color=info["color"])
            refs = []
            for sid in claim["cited_source_ids"]:
                if sid in sources:
                    kind = source_type_info(sources[sid]["source_type"])["label"]
                    refs.append(f"{sid} ({kind}: {source_reference(sources[sid])})")
            st.markdown("Cites: " + ("; ".join(refs) if refs else "no valid source"))
            if claim["invalid_source_ids"]:
                st.markdown(f"Invalid IDs: {', '.join(claim['invalid_source_ids'])}")
            review_status = claim.get("review_status")
            if review_status:
                reason = claim.get("review_reason")
                review_line = f"LLM review: {review_status.replace('_', ' ')}"
                if reason:
                    review_line += f" — {reason}"
                st.caption(review_line)
            st.caption("Why: " + " ".join(claim["explanations"]))


def render_report(run: dict) -> None:
    if not run.get("report"):
        st.info("No report: this run was made without analysis.")
        return
    st.download_button(
        "Download report (.md)",
        data=run["report"],
        file_name=report_filename(run),
        mime="text/markdown",
    )
    with st.container(border=True):
        st.markdown(run["report"])


run = state.run
if run:
    st.divider()
    st.subheader("Results")
    st.markdown(f"> {run['request']['question']}")  # the question stored with the run
    analysis = run.get("analysis")
    if analysis:
        kind, label, explanation = analysis_status_info(analysis["status"])
    else:
        kind, label, explanation = "info", "Not run", "This run was made without analysis."
    mix = evidence_mix(run)
    evidence = run_evidence_info(run)
    cols = st.columns(4)
    cols[0].metric("Analysis", label)
    cols[1].metric("Verified claims", claim_counts(run).get("verified", 0))
    cols[2].metric("Sources", len(run["sources"]))
    cols[3].metric("Duration", f"{run['metrics']['duration_ms'] / 1000:.1f} s")

    st.badge(evidence["label"], color=evidence["color"])
    if mix["demo"]:
        st.warning(evidence["description"])
    else:
        st.caption(evidence["description"])
        demo_notice = demo_evidence_notice(run)
        if demo_notice:
            st.caption(demo_notice)

    getattr(st, kind)(explanation)
    for error in (analysis or {}).get("errors", []):
        st.error(error)
    st.caption(
        f"Evidence: {mix['document']} uploaded-PDF passage(s) · {mix['web']} web page(s) · "
        f"{mix['demo']} synthetic demo page(s) · {research_status_label(run['status'])} · "
        f"{len(run['request']['document_ids'])} document(s) searched · run `{run['run_id']}`"
    )
    for warning in run["warnings"]:
        st.warning(warning)
    for error in run["errors"]:
        st.error(f"{error['query_id']} search failed: {error['message']}")
    render_retrieval_diagnostics(run)

    tabs = st.tabs(["Sources", "Agent analysis", "Verification", "Report"])
    with tabs[0]:
        render_sources(run)
    with tabs[1]:
        render_analysis(run)
    with tabs[2]:
        render_verification(run)
    with tabs[3]:
        render_report(run)
