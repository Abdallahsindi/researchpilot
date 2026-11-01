"""End-to-end tests of the analysis branch of the LangGraph workflow.

The LLM is scripted (tests/conftest.py::ScriptedLLM), so these tests check routing,
validation, bounded revision, error handling, and persistence, not real model quality.
"""

import json
import re
import time

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.analysis import AnalysisStatus, ReviewStatus, VerificationStatus
from app.schemas.research import ResearchRequest, ResearchResult, RunStatus
from app.services.llm import LLMLoadError
from tests.conftest import ScriptedLLM, source_ids_containing

QUESTION = "Why was lithium iron phosphate chosen for the battery storage?"
GOOD_CLAIM = (
    "The battery system uses lithium iron phosphate cells with a usable capacity of 2.4 "
    "megawatt hours."
)


def analyst_reply(findings, answer="Lithium iron phosphate was chosen for safety.", **extra):
    return json.dumps({"answer": answer, "findings": findings, "limitations": [], **extra})


def citing_battery_page(prompt: str) -> str:
    ids = source_ids_containing(prompt, "lithium iron phosphate")
    return analyst_reply([{"claim": GOOD_CLAIM, "source_ids": ids[:1], "criterion": "safety"}])


@pytest.fixture
def make_analysis_service(make_service, sample_pdf):
    def factory(llm, **overrides):
        service = make_service(**overrides)
        service._llm_factory = lambda: llm
        info, _ = service.documents().ingest(sample_pdf, "report.pdf")
        return service, info.document_id

    return factory


def request(document_id: str, **fields) -> ResearchRequest:
    return ResearchRequest(
        question=QUESTION, document_ids=[document_id], criteria=["safety"], analyze=True, **fields
    )


async def test_analysis_disabled_preserves_behaviour_and_never_creates_llm(make_service):
    service = make_service()

    def must_not_be_called():
        raise AssertionError("the LLM must not be created when analyze is false")

    service._llm_factory = must_not_be_called

    result = await service.run(ResearchRequest(question="Should a small team use pgvector?"))

    assert result.status is RunStatus.COMPLETED
    assert result.analysis is None and result.report is None
    assert result.metrics.llm_calls == 0


async def test_full_graph_produces_verified_document_finding(make_analysis_service):
    llm = ScriptedLLM(citing_battery_page)
    service, document_id = make_analysis_service(llm)

    result = await service.run(request(document_id))

    analysis = result.analysis
    assert analysis.status is AnalysisStatus.COMPLETED
    assert analysis.model_id == "scripted-llm" and llm.loaded
    claim = analysis.verification.claims[0]
    assert claim.status is VerificationStatus.VERIFIED
    cited = next(s for s in result.sources if s.id == claim.cited_source_ids[0])
    assert (cited.source_type.value, cited.filename, cited.page_number) == (
        "document",
        "report.pdf",
        2,
    )
    assert f"[{cited.id}: report.pdf, p. 2]" in result.report
    assert result.metrics.llm_calls == 1 + llm.review_calls
    assert len(analysis.attempts) == 1  # nothing failed, so no revision
    # The Analyst saw the real collected sources, wrapped as untrusted blocks.
    assert "lithium iron phosphate" in llm.analyst_calls[0]
    assert "Criteria to compare: safety" in llm.analyst_calls[0]


async def test_extended_result_is_persisted_and_reloaded(make_analysis_service):
    service, document_id = make_analysis_service(ScriptedLLM(citing_battery_page))

    result = await service.run(request(document_id))
    stored = service.get(result.run_id)

    assert stored == result
    assert stored.report == result.report and stored.analysis.verification is not None
    assert ResearchResult.model_validate_json(result.model_dump_json()) == result


async def test_invented_ids_trigger_one_bounded_revision(make_analysis_service):
    llm = ScriptedLLM(
        [
            analyst_reply([{"claim": GOOD_CLAIM, "source_ids": ["S42"]}]),
            analyst_reply([{"claim": GOOD_CLAIM, "source_ids": ["S43"]}]),
            analyst_reply([{"claim": GOOD_CLAIM, "source_ids": ["S44"]}]),
        ]
    )
    service, document_id = make_analysis_service(llm)

    result = await service.run(request(document_id))

    analysis = result.analysis
    assert len(llm.analyst_calls) == 2  # first attempt + one revision, never more
    assert [a.invented_source_ids for a in analysis.attempts] == [["S42"], ["S43"]]
    assert "S42" in llm.analyst_calls[1]  # the revision prompt carried the feedback
    assert analysis.status is AnalysisStatus.NO_VERIFIED_FINDINGS
    assert analysis.verification.claims[0].status is VerificationStatus.INVALID_REFERENCE
    findings = result.report.split("## Key findings (verified)")[1].split("## ")[0]
    assert "No finding was verified." in findings


async def test_revision_can_fix_a_rejected_claim(make_analysis_service):
    def reply(prompt):
        if "failed verification" in prompt:
            return citing_battery_page(prompt)
        return analyst_reply([{"claim": "The containers are painted green.", "source_ids": ["S1"]}])

    llm = ScriptedLLM(reply)
    service, document_id = make_analysis_service(llm)

    result = await service.run(request(document_id))

    assert len(result.analysis.attempts) == 2
    assert result.analysis.status is AnalysisStatus.COMPLETED
    assert "painted green" not in result.report.split("## Key findings")[1].split("## ")[0]


@pytest.mark.parametrize(("revisions", "calls"), [(0, 1), (1, 2), (2, 3)])
async def test_revision_limit_is_configurable_and_bounded(make_analysis_service, revisions, calls):
    llm = ScriptedLLM(["not json at all"])
    service, document_id = make_analysis_service(llm, analysis_max_revisions=revisions)

    result = await service.run(request(document_id))

    assert len(llm.analyst_calls) == calls
    assert result.analysis.status is AnalysisStatus.FAILED
    assert all(a.parse_error for a in result.analysis.attempts)
    assert "not valid JSON" in result.analysis.errors[0]


async def test_malformed_then_valid_output_recovers(make_analysis_service):
    llm = ScriptedLLM(
        lambda p: (
            citing_battery_page(p)
            if "JSON" in p.split("problems")[-1] and "previous reply" in p
            else '{"answer": "x", "findings": "S1"}'
        )
    )
    service, document_id = make_analysis_service(llm)

    result = await service.run(request(document_id))

    assert [bool(a.parse_error) for a in result.analysis.attempts] == [True, False]
    assert result.analysis.status is AnalysisStatus.COMPLETED


async def test_contradicted_claim_is_not_presented_as_fact(make_analysis_service):
    llm = ScriptedLLM(citing_battery_page, review="CONTRADICTED")
    service, document_id = make_analysis_service(llm, analysis_max_revisions=0)

    result = await service.run(request(document_id))

    claim = result.analysis.verification.claims[0]
    assert claim.status is VerificationStatus.CONTRADICTED
    assert result.analysis.status is AnalysisStatus.NO_VERIFIED_FINDINGS
    assert "**contradicted:**" in result.report


async def test_model_abstention(make_analysis_service):
    llm = ScriptedLLM(
        [analyst_reply([], answer="INSUFFICIENT_EVIDENCE", limitations=["No data on cost."])]
    )
    service, document_id = make_analysis_service(llm)

    result = await service.run(request(document_id))

    assert result.analysis.status is AnalysisStatus.INSUFFICIENT_EVIDENCE
    assert result.analysis.verification is None and llm.review_calls == 0
    assert "not sufficient to answer" in result.report


async def test_insufficient_evidence_skips_the_model(make_service):
    service = make_service()

    def must_not_be_called():
        raise AssertionError("no LLM call when evidence is insufficient")

    service._llm_factory = must_not_be_called

    result = await service.run(
        ResearchRequest(question="What is the best recipe for sourdough bread?", analyze=True)
    )

    assert result.status is RunStatus.INSUFFICIENT_EVIDENCE
    assert result.analysis.status is AnalysisStatus.INSUFFICIENT_EVIDENCE
    assert "Analysis skipped" in result.analysis.errors[0]
    assert result.report.startswith("# Research report")


async def test_model_unavailable_is_a_failed_analysis_not_a_crash(make_service, sample_pdf):
    service = make_service()
    attempts = []

    def unavailable():
        attempts.append(1)
        raise LLMLoadError("Could not load model 'Qwen/Qwen2.5-1.5B-Instruct': offline")

    service._llm_factory = unavailable
    info, _ = service.documents().ingest(sample_pdf, "report.pdf")

    result = await service.run(request(info.document_id))
    await service.run(request(info.document_id))

    assert result.status is RunStatus.COMPLETED and result.sources  # research still returned
    assert result.analysis.status is AnalysisStatus.FAILED
    assert "offline" in result.analysis.errors[0]
    assert "could not be completed" in result.report
    assert service.get(result.run_id) == result
    assert len(attempts) >= 2  # a failed creation is not cached; the next request retries


@pytest.mark.parametrize(
    ("llm", "message"),
    [
        (ScriptedLLM(["{}"], error=RuntimeError("CUDA error: device-side assert")), "CUDA error"),
        (ScriptedLLM(["{}"], delay=0.5), "timed out"),
    ],
)
async def test_inference_error_and_timeout(make_analysis_service, llm, message):
    service, document_id = make_analysis_service(llm, analysis_timeout_seconds=0.1)

    result = await service.run(request(document_id))

    assert result.analysis.status is AnalysisStatus.FAILED
    assert message in result.analysis.errors[-1]
    assert len(result.analysis.attempts) == 0  # an inference failure is never retried


async def test_review_timeout_falls_back_to_code_checks(make_analysis_service):
    llm = ScriptedLLM(citing_battery_page, review_delay=0.3)
    service, document_id = make_analysis_service(llm, analysis_timeout_seconds=0.2)

    started = time.perf_counter()
    result = await service.run(request(document_id))

    assert time.perf_counter() - started < 5
    assert any("review timed out" in e for e in result.analysis.errors)
    claim = result.analysis.verification.claims[0]
    assert claim.status is VerificationStatus.UNVERIFIED
    assert claim.review_status is ReviewStatus.FAILED
    assert "timed out" in claim.review_reason
    assert result.analysis.status is AnalysisStatus.NO_VERIFIED_FINDINGS


async def test_review_disabled_means_nothing_is_verified(make_analysis_service):
    llm = ScriptedLLM(citing_battery_page)
    service, document_id = make_analysis_service(llm, verifier_llm_review=False)

    result = await service.run(request(document_id))

    claim = result.analysis.verification.claims[0]
    assert claim.status is VerificationStatus.UNVERIFIED and claim.code_checks_passed
    assert claim.review_status is ReviewStatus.NOT_PERFORMED
    assert "disabled" in claim.review_reason
    assert llm.review_calls == 0
    assert "## Unverified observations" in result.report


def test_api_analyze_round_trip_and_default_off(make_settings, sample_pdf):
    llm = ScriptedLLM(citing_battery_page)
    client = TestClient(create_app(make_settings(), llm_factory=lambda: llm))
    document_id = client.post(
        "/documents", files={"file": ("report.pdf", sample_pdf, "application/pdf")}
    ).json()["document_id"]

    plain = client.post("/research", json={"question": QUESTION, "document_ids": [document_id]})
    assert plain.status_code == 200 and plain.json()["analysis"] is None
    assert plain.json()["report"] is None and llm.analyst_calls == []  # model never used

    analysed = client.post(
        "/research", json={"question": QUESTION, "document_ids": [document_id], "analyze": True}
    )
    body = analysed.json()
    assert analysed.status_code == 200
    assert body["analysis"]["status"] == "completed"
    assert body["analysis"]["verification"]["claims"][0]["status"] == "verified"
    assert body["report"].startswith("# Research report")
    assert client.get(f"/research/{body['run_id']}").json() == body


def test_legacy_run_without_analysis_fields_still_loads():
    legacy = {
        "run_id": "6f1e3c2a-1b2c-4d5e-8f90-123456789abc",
        "status": "completed",
        "mode": "demo",
        "search_provider": "demo",
        "request": {"question": "A question from an earlier release"},
        "plan": {"objective": "x", "queries": []},
        "sources": [],
        "metrics": {
            "duration_ms": 1,
            "search_calls": 1,
            "failed_queries": 0,
            "raw_hits": 0,
            "sources_returned": 0,
        },
    }

    result = ResearchResult.model_validate(legacy)

    assert result.analysis is None and result.report is None
    assert result.request.analyze is False


async def test_only_off_topic_claims_means_insufficient_evidence(make_analysis_service):
    def off_topic(prompt):
        # Like the real model: copy a sentence from a retrieved source that doesn't answer it.
        sid, text = re.findall(r'<source id="(S\d+)"[^>]*>\n(.*?)\n</source>', prompt, re.S)[0]
        claim = text.split(". ")[0].strip().rstrip(".") + "."
        return analyst_reply([{"claim": claim, "source_ids": [sid]}], answer=claim)

    service, document_id = make_analysis_service(ScriptedLLM(off_topic))

    result = await service.run(
        ResearchRequest(
            question="How many electric vehicles does the campus charge each day?",
            document_ids=[document_id],
            analyze=True,
        )
    )

    claim = result.analysis.verification.claims[0]
    assert claim.addresses_question is False
    assert claim.status is VerificationStatus.UNVERIFIED
    assert result.analysis.status is AnalysisStatus.INSUFFICIENT_EVIDENCE
    assert "not sufficient to answer" in result.report
