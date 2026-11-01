"""Tests for the Streamlit frontend.

The UI is exercised with Streamlit's AppTest harness against the real FastAPI app running
in-process (TestClient), with the scripted LLM and offline embeddings from conftest. This
checks the HTTP contract and what the UI displays; it is not a browser test and says nothing
about real model quality.
"""

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

import frontend.api_client as api_client
from app.main import create_app
from frontend.api_client import ApiError, ResearchPilotClient, build_research_payload, split_lines
from frontend.view_model import (
    claim_status_info,
    demo_evidence_notice,
    evidence_mix,
    report_filename,
    run_evidence_info,
    search_scope,
    source_reference,
)
from tests.conftest import ScriptedLLM, source_ids_containing

# Starlette's TestClient warns about per-request timeouts; the real httpx client honours them.
pytestmark = pytest.mark.filterwarnings("ignore:You should not use the 'timeout' argument")

APP_PATH = str(Path(__file__).resolve().parents[1] / "frontend" / "streamlit_app.py")
QUESTION = "Why was lithium iron phosphate chosen for the battery storage?"


def cite_battery(prompt: str) -> str:
    ids = source_ids_containing(prompt, "lithium iron phosphate")
    claim = "The battery system uses lithium iron phosphate cells."
    return json.dumps(
        {
            "answer": claim,
            "findings": [
                {"claim": claim, "source_ids": ids[:1], "criterion": "safety"},
                {"claim": "The containers are painted green.", "source_ids": ids[:1]},
            ],
            "limitations": ["No cost data."],
        }
    )


@pytest.fixture
def backend(make_settings):
    llm = ScriptedLLM(cite_battery)
    app = create_app(make_settings(analysis_max_revisions=0), llm_factory=lambda: llm)
    return TestClient(app), llm


@pytest.fixture
def ui(backend, monkeypatch):
    test_client, _ = backend
    monkeypatch.setattr(api_client, "make_http_client", lambda base_url: test_client)
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    return at


def _all_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    for kind in ("success", "warning", "error", "info"):
        parts += [e.value for e in getattr(at, kind)]
    return "\n".join(str(p) for p in parts)


# ---------------------------------------------------------------- API client


def test_client_matches_the_real_api_contract(backend, sample_pdf):
    client = ResearchPilotClient("http://testserver", client=backend[0])

    assert client.health()["mode"] == "demo"
    info, created = client.upload_document("report.pdf", sample_pdf)
    again, created_again = client.upload_document("report.pdf", sample_pdf)
    payload, problems = build_research_payload(
        QUESTION, criteria="safety", document_ids=[info["document_id"]]
    )
    run = client.research(payload)

    assert created and not created_again and again["document_id"] == info["document_id"]
    assert problems == [] and payload["analyze"] is True
    assert build_research_payload("   ")[1] == [
        "Enter a research question before pressing Analyze."
    ]
    assert run["analysis"]["status"] == "completed"
    assert client.get_run(run["run_id"]) == run
    assert [d["document_id"] for d in client.list_documents()] == [info["document_id"]]


def test_client_reports_api_and_connection_errors(backend):
    client = ResearchPilotClient("http://testserver", client=backend[0])
    with pytest.raises(ApiError) as validation:
        client.research({"question": "short"})
    assert validation.value.status_code == 422 and "question" in validation.value.message
    with pytest.raises(ApiError, match="HTTP 415"):
        client.upload_document("notes.txt", b"plain text")

    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    offline = ResearchPilotClient(
        "http://localhost:8000", client=httpx.Client(transport=httpx.MockTransport(refuse))
    )
    with pytest.raises(
        ApiError, match="Cannot reach the ResearchPilot API at http://localhost:8000"
    ):
        offline.health()


def test_payload_building_and_validation():
    assert split_lines("cost\n- safety\n\nCost\n  latency, p95 ") == [
        "cost",
        "safety",
        "latency, p95",
    ]
    payload, problems = build_research_payload("Too short", criteria="\n".join(map(str, range(11))))
    assert payload["criteria"][:2] == ["0", "1"]
    assert any("10-1000 characters" in p for p in problems)
    assert any("At most 10 criteria" in p for p in problems)


def test_demo_evidence_notice_distinguishes_configuration_from_actual_sources():
    pdf_only = {"mode": "demo", "sources": [{"source_type": "document"}]}
    synthetic = {"mode": "demo", "sources": [{"source_type": "demo"}]}

    notice = demo_evidence_notice(pdf_only)
    assert "Search is configured for Demo mode" in notice
    assert "relies on uploaded PDF evidence" in notice
    # The report warns about actual synthetic sources; the UI does not add a second notice.
    assert demo_evidence_notice(synthetic) is None


def test_run_evidence_labels_reflect_the_sources_actually_used():
    pdf_only = {"sources": [{"source_type": "document"}, {"source_type": "document"}]}
    synthetic = {"sources": [{"source_type": "document"}, {"source_type": "demo"}]}
    live = {
        "sources": [
            {"source_type": "document"},
            {"source_type": "web"},
            {"source_type": "web"},
        ]
    }

    pdf_info = run_evidence_info(pdf_only)
    assert pdf_info["label"] == "Local PDF Research"
    assert "2 uploaded-PDF passage(s)" in pdf_info["description"]
    assert "no synthetic demo sources" in pdf_info["description"]

    synthetic_info = run_evidence_info(synthetic)
    assert synthetic_info["label"] == "Synthetic demo evidence"
    assert "sample data, not live web research" in synthetic_info["description"]

    live_info = run_evidence_info(live)
    assert live_info["label"] == "Documents + live web evidence"
    assert "2 live web page(s)" in live_info["description"]
    assert "1 uploaded-PDF passage(s)" in live_info["description"]


def test_view_model_never_promotes_unknown_statuses():
    assert claim_status_info("verified")["label"] == "Verified"
    assert search_scope(1, {"mode": "demo"}).endswith("(synthetic sample pages, not the real web)")
    assert search_scope(2, {"mode": "live", "search_provider": "tavily"}) == (
        "2 uploaded documents + live web search (tavily)"
    )
    assert evidence_mix({"sources": [{"source_type": "demo"}, {"source_type": "document"}]}) == {
        "document": 1,
        "web": 0,
        "demo": 1,
    }
    unknown = claim_status_info("something_new")
    assert unknown["label"] == "something_new" and unknown["color"] == "gray"
    assert (
        source_reference({"source_type": "document", "filename": "a.pdf", "page_number": 3})
        == "a.pdf, page 3"
    )
    assert source_reference({"source_type": "demo", "url": "https://demo.example/x"}) == (
        "https://demo.example/x"
    )
    assert report_filename({"run_id": "12345678-abcd"}) == "researchpilot_report_12345678.md"


# ---------------------------------------------------------------- Streamlit app


def analyze(at: AppTest) -> None:
    at.button(key="FormSubmitter:research_form-Analyze").click()
    at.run()


def test_app_shows_backend_down_and_does_not_send(monkeypatch):
    monkeypatch.setenv("RESEARCHPILOT_API_URL", "http://127.0.0.1:9")
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    assert not at.exception
    assert any("Backend not reachable" in e.value for e in at.error)
    at.text_area(key="question").input(QUESTION)
    analyze(at)
    assert any("not reachable" in e.value for e in at.error)
    assert at.session_state["run"] is None


def test_selected_file_is_not_reported_as_uploaded(ui, backend, sample_pdf):
    at = ui
    at.file_uploader(key="pdf_files").upload("report.pdf", sample_pdf, "application/pdf")
    at.run()

    assert "Selected in the browser — not uploaded yet" in _all_text(at)
    assert at.session_state["documents"] == {}
    assert backend[0].get("/documents").json() == []  # the backend has not received it
    assert "0 uploaded documents" in _all_text(at)
    assert "1 selected file(s) not uploaded yet will be uploaded first" in _all_text(at)


def test_upload_button_registers_document_with_backend(ui, backend, sample_pdf):
    at = ui
    at.file_uploader(key="pdf_files").upload("report.pdf", sample_pdf, "application/pdf")
    at.run()
    next(b for b in at.button if b.label == "Upload 1 selected file").click()
    at.run()

    registered = backend[0].get("/documents").json()
    document_id = registered[0]["document_id"]
    assert [d["filename"] for d in registered] == ["report.pdf"]
    assert at.session_state["documents"].keys() == {document_id}  # the real returned ID
    assert at.checkbox(key=f"use_{document_id}").value is True
    text = _all_text(at)
    assert f":green[Uploaded] · `{document_id}` · 4 pages" in text
    assert "1 uploaded document + web search in Demo mode" in text


def test_analyze_uploads_pending_file_and_sends_exact_question_and_ids(ui, backend, sample_pdf):
    at, (client, llm) = ui, backend
    at.file_uploader(key="pdf_files").upload("report.pdf", sample_pdf, "application/pdf")
    at.run()
    typed = "  Why was lithium iron phosphate chosen for the battery storage?  "
    at.text_area(key="question").input(typed)
    at.text_area(key="criteria").input("safety")
    analyze(at)

    assert not at.exception
    document_id = client.get("/documents").json()[0]["document_id"]
    run = at.session_state["run"]
    assert run["request"]["question"] == typed.strip()  # exactly what was typed
    assert run["request"]["document_ids"] == [document_id]
    assert run["request"]["analyze"] is True and run["request"]["criteria"] == ["safety"]
    assert len(llm.analyst_calls) == 1  # one request through FastAPI
    assert client.get(f"/research/{run['run_id']}").json() == run  # shown run is the stored one

    text = _all_text(at)
    assert f"> {typed.strip()}" in text  # the question returned by the stored run
    assert "1 document(s) searched" in text
    assert "report.pdf, page 2" in text
    assert any("Retrieval diagnostics" in expander.label for expander in at.expander)
    assert "similarity 0." in text and "kept" in text
    statuses = [c["status"] for c in run["analysis"]["verification"]["claims"]]
    assert statuses == ["verified", "unsupported"]
    assert text.count(":green-badge[Verified]") == 1  # the rejected claim is not relabelled
    assert ":red-badge[Unsupported]" in text
    assert "Demo mode: web results come from a small offline set of synthetic" in text
    assert [t.label for t in at.tabs] == ["Sources", "Agent analysis", "Verification", "Report"]
    assert any(m.value.strip() == run["report"].strip() for m in at.markdown)


def test_empty_question_is_rejected_before_any_upload(ui, backend, sample_pdf):
    at = ui
    at.file_uploader(key="pdf_files").upload("report.pdf", sample_pdf, "application/pdf")
    at.run()
    analyze(at)

    assert any("Enter a research question" in e.value for e in at.error)
    assert at.session_state["run"] is None
    assert backend[0].get("/documents").json() == []


def test_example_button_fills_the_form_and_selects_the_sample_pdf(ui, backend):
    next(b for b in ui.button if b.label.startswith("Load example question")).click()
    ui.run()

    assert ui.text_area(key="question").value.startswith("Based on the pilot results")
    assert ui.text_area(key="criteria").value == "efficiency\nsafety\ncost"
    documents = backend[0].get("/documents").json()
    assert [doc["filename"] for doc in documents] == ["researchpilot_test_report.pdf"]
    assert ui.checkbox(key=f"use_{documents[0]['document_id']}").value is True


def test_example_selection_clears_an_unrelated_document(ui, backend, sample_pdf):
    at, (client, _) = ui, backend
    at.file_uploader(key="pdf_files").upload("report.pdf", sample_pdf, "application/pdf")
    at.run()
    next(b for b in at.button if b.label == "Upload 1 selected file").click()
    at.run()
    uploaded_id = client.get("/documents").json()[0]["document_id"]
    assert at.checkbox(key=f"use_{uploaded_id}").value is True

    next(b for b in at.button if b.label.startswith("Load example question")).click()
    at.run()

    sample = next(
        doc
        for doc in client.get("/documents").json()
        if doc["filename"] == "researchpilot_test_report.pdf"
    )
    assert at.checkbox(key=f"use_{uploaded_id}").value is False
    assert at.checkbox(key=f"use_{sample['document_id']}").value is True


def test_failed_model_and_insufficient_evidence_are_explained(make_settings, monkeypatch):
    from app.services.llm import LLMLoadError

    def unavailable():
        raise LLMLoadError("Could not load model 'Qwen/Qwen2.5-1.5B-Instruct': offline")

    client = TestClient(create_app(make_settings(), llm_factory=unavailable))
    monkeypatch.setattr(api_client, "make_http_client", lambda base_url: client)
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    at.text_area(key="question").input(
        "Should a small team use pgvector or a dedicated vector database?"
    )
    analyze(at)
    text = _all_text(at)
    assert at.session_state["run"]["analysis"]["status"] == "failed"
    assert "The model was unavailable" in text and "offline" in text
    assert "Synthetic demo pages (not real evidence)" in text  # demo sources labelled

    at.text_area(key="question").input("What is the best recipe for sourdough bread starter?")
    analyze(at)
    assert at.session_state["run"]["analysis"]["status"] == "insufficient_evidence"
    assert "The collected sources do not answer this question" in _all_text(at)


def test_load_saved_run_by_id(ui, backend):
    run = ResearchPilotClient("http://testserver", client=backend[0]).research(
        {"question": "Should a small team use pgvector?"}
    )
    ui.sidebar.text_input[0].input(run["run_id"])
    ui.run()  # a browser reruns after typing, which enables the button
    next(b for b in ui.sidebar.button if b.label == "Load run").click()
    ui.run()

    assert ui.session_state["run"] == run
    assert "This run was made without analysis" in _all_text(ui)
