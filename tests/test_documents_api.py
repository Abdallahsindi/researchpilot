import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.rag.deps import MissingDependencyError
from tests.pdf_factory import build_pdf


@pytest.fixture
def client(make_settings) -> TestClient:
    return TestClient(create_app(make_settings()))


def upload(client, data: bytes, filename: str = "report.pdf"):
    return client.post("/documents", files={"file": (filename, data, "application/pdf")})


def test_upload_list_get_and_dedupe(client, sample_pdf):
    created = upload(client, sample_pdf)

    assert created.status_code == 201
    info = created.json()
    assert info["filename"] == "report.pdf"
    assert info["page_count"] == 4 and info["pages_with_text"] == 3
    assert info["chunk_count"] == 3
    assert any("OCR is not supported" in w for w in info["warnings"])

    duplicate = upload(client, sample_pdf, "other-name.pdf")
    assert duplicate.status_code == 200 and duplicate.json() == info

    assert [d["document_id"] for d in client.get("/documents").json()] == [info["document_id"]]
    assert client.get(f"/documents/{info['document_id']}").json() == info
    assert client.get("/documents/doc_000000000000").status_code == 404


@pytest.mark.parametrize(
    ("data", "filename", "status"),
    [
        (b"plain text", "notes.txt", 415),
        (b"%PDF-1.4 garbage", "broken.pdf", 422),
        (build_pdf([None]), "scan.pdf", 422),
    ],
)
def test_upload_rejects_bad_files(client, data, filename, status):
    response = upload(client, data, filename)

    assert response.status_code == status
    assert response.json()["detail"]


def test_upload_rejects_oversized_files(make_settings, sample_pdf):
    client = TestClient(create_app(make_settings(max_upload_mb=0.001)))

    response = upload(client, sample_pdf + b"\n" * 2000)

    assert response.status_code == 413


def test_document_search_returns_page_metadata(client, sample_pdf):
    document_id = upload(client, sample_pdf).json()["document_id"]

    response = client.post(
        "/documents/search",
        json={"query": "lithium iron phosphate battery", "top_k": 2, "document_ids": [document_id]},
    )

    assert response.status_code == 200
    top = response.json()[0]
    assert top["chunk"]["page_number"] == 2 and top["rank"] == 1
    assert top["chunk"]["chunk_id"].startswith(f"{document_id}:p2:")
    unknown = client.post(
        "/documents/search", json={"query": "battery", "document_ids": ["doc_000000000000"]}
    )
    assert unknown.status_code == 404


def test_research_combines_document_and_demo_sources(client, sample_pdf):
    document_id = upload(client, sample_pdf).json()["document_id"]

    response = client.post(
        "/research",
        json={
            "question": "Why choose lithium iron phosphate battery storage for a vector database?",
            "document_ids": [document_id],
        },
    )

    assert response.status_code == 200
    body = response.json()
    document_sources = [s for s in body["sources"] if s["source_type"] == "document"]
    demo_sources = [s for s in body["sources"] if s["source_type"] == "demo"]
    assert document_sources and demo_sources
    assert body["sources"][0]["source_type"] == "document"  # uploaded evidence first
    assert document_sources[0]["page_number"] == 2
    assert document_sources[0]["url"] is None and document_sources[0]["filename"] == "report.pdf"
    assert all(s["url"].startswith("https://demo.example/") for s in demo_sources)
    assert body["metrics"]["document_searches"] == 1
    assert body["metrics"]["document_chunks_retrieved"] >= 1


def test_research_with_unknown_or_malformed_document_ids(client):
    question = "A question with no criteria at all"
    unknown = client.post(
        "/research", json={"question": question, "document_ids": ["doc_000000000000"]}
    )
    malformed = client.post("/research", json={"question": question, "document_ids": ["../x"]})

    assert unknown.status_code == 404 and "doc_000000000000" in unknown.json()["detail"]
    assert malformed.status_code == 422


def test_uploaded_pdf_retrieval_reports_sources_and_diagnostics(client, sample_pdf):
    document_id = upload(client, sample_pdf).json()["document_id"]

    body = client.post(
        "/research",
        json={
            "question": "Why was lithium iron phosphate chosen for the battery storage?",
            "document_ids": [document_id],
        },
    ).json()

    document_sources = [s for s in body["sources"] if s["source_type"] == "document"]
    assert body["status"] == "completed"
    assert document_sources and document_sources[0]["page_number"] == 2
    assert {s["document_id"] for s in document_sources} == {document_id}

    assert body["document_index"]["indexed_chunks"] == 3
    assert body["document_index"]["selected_documents"] == [
        {
            "document_id": document_id,
            "filename": "report.pdf",
            "page_count": 4,
            "chunk_count": 3,
        }
    ]
    diagnostics = body["document_retrieval"][0]
    assert diagnostics["selected_document_ids"] == [document_id]
    assert diagnostics["accepted"] >= 1 and diagnostics["rejected"] >= 1
    assert diagnostics["candidates"][0]["score"] >= diagnostics["min_score"]
    assert body["metrics"]["document_candidates_retrieved"] == 3
    assert body["metrics"]["document_candidates_rejected"] >= 1


def test_unrelated_question_does_not_fabricate_document_evidence(make_settings, sample_pdf):
    # Hashing embeddings can collide on unrelated words; use a calibrated floor for this
    # model-free test rather than weakening the assertions.
    client = TestClient(create_app(make_settings(document_min_score=0.3)))
    document_id = upload(client, sample_pdf).json()["document_id"]

    body = client.post(
        "/research",
        json={
            "question": "What is the best recipe for sourdough bread starter?",
            "document_ids": [document_id],
        },
    ).json()

    assert body["status"] == "insufficient_evidence"
    assert not [s for s in body["sources"] if s["source_type"] == "document"]
    assert body["metrics"]["document_chunks_retrieved"] == 0
    assert "No passage in the selected documents met the similarity threshold" in " ".join(
        body["warnings"]
    )


def test_document_id_filtering_excludes_unselected_uploads(make_settings, sample_pdf):
    client = TestClient(create_app(make_settings(document_min_score=0.3)))
    battery_id = upload(client, sample_pdf).json()["document_id"]
    marine_pdf = build_pdf(
        ["Marine biology field notes about coral reefs and ocean currents."],
        metadata={"Title": "Marine notes"},
    )
    marine_id = upload(client, marine_pdf, "marine.pdf").json()["document_id"]

    battery = client.post(
        "/research",
        json={
            "question": "Which battery chemistry was chosen for safety?",
            "document_ids": [battery_id],
        },
    ).json()
    marine = client.post(
        "/research",
        json={
            "question": "Which battery chemistry was chosen for safety?",
            "document_ids": [marine_id],
        },
    ).json()

    assert battery["sources"] and all(
        s["document_id"] == battery_id for s in battery["sources"] if s["source_type"] == "document"
    )
    assert not [s for s in marine["sources"] if s["source_type"] == "document"]
    assert marine["status"] == "insufficient_evidence"


def test_rejected_candidates_and_reason_are_exposed(make_settings, sample_pdf):
    settings = make_settings(document_min_score=0.99, document_fallback_min_score=0.98)
    client = TestClient(create_app(settings))
    document_id = upload(client, sample_pdf).json()["document_id"]

    body = client.post(
        "/research",
        json={
            "question": "Why was lithium iron phosphate chosen for the battery storage?",
            "document_ids": [document_id],
        },
    ).json()

    diagnostics = body["document_retrieval"][0]
    assert body["status"] == "insufficient_evidence"
    assert diagnostics["accepted"] == 0 and diagnostics["rejected"] >= 1
    assert diagnostics["candidates"]
    assert all(not candidate["accepted"] for candidate in diagnostics["candidates"])
    assert "No candidate met the similarity threshold 0.99" in diagnostics["reason"]
    assert body["metrics"]["document_candidates_rejected"] >= 1


def test_minimum_source_requirement_can_abstain_on_a_single_passage(make_settings, sample_pdf):
    strict = TestClient(create_app(make_settings(min_sources=2, document_top_k=1)))
    document_id = upload(strict, sample_pdf).json()["document_id"]
    strict_body = strict.post(
        "/research",
        json={
            "question": "Why was lithium iron phosphate chosen?",
            "document_ids": [document_id],
        },
    ).json()

    permissive = TestClient(create_app(make_settings(min_sources=1, document_top_k=1)))
    permissive_id = upload(permissive, sample_pdf).json()["document_id"]
    permissive_body = permissive.post(
        "/research",
        json={
            "question": "Why was lithium iron phosphate chosen?",
            "document_ids": [permissive_id],
        },
    ).json()

    assert len(strict_body["sources"]) == 1
    assert strict_body["status"] == "insufficient_evidence"
    assert permissive_body["sources"]
    assert permissive_body["status"] == "completed"


def test_missing_optional_dependency_returns_503(make_settings):
    def unavailable():
        raise MissingDependencyError("The optional package 'faiss' is required ...")

    client = TestClient(create_app(make_settings(), library_factory=unavailable))

    response = client.get("/documents")

    assert response.status_code == 503 and "faiss" in response.json()["detail"]
    # Research without documents does not need the library and still works.
    ok = client.post("/research", json={"question": "Should a small team use pgvector?"})
    assert ok.status_code == 200
