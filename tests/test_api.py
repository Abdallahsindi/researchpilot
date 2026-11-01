from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client(make_settings) -> TestClient:
    return TestClient(create_app(make_settings()))


def test_health_reports_demo_mode(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "mode": "demo", "search_provider": "demo"}


def test_research_round_trip(client):
    response = client.post(
        "/research",
        json={
            "question": "Should a small team use pgvector or a dedicated vector database?",
            "criteria": ["cost"],
            "constraints": ["Team of 3 engineers"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["sources"][0]["id"] == "S1"
    assert body["plan"]["constraints"] == ["Team of 3 engineers"]

    fetched = client.get(f"/research/{body['run_id']}")
    assert fetched.status_code == 200
    assert fetched.json() == body


def test_unknown_run_returns_404(client):
    assert client.get(f"/research/{uuid4()}").status_code == 404


def test_invalid_run_id_returns_422(client):
    assert client.get("/research/not-a-uuid").status_code == 422


@pytest.mark.parametrize(
    "payload",
    [{"question": "short"}, {"question": "A valid question here?", "criteria": [""]}, {}],
)
def test_invalid_request_returns_422(client, payload):
    assert client.post("/research", json=payload).status_code == 422
