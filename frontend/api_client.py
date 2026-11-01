"""HTTP client for the ResearchPilot FastAPI backend, used by the Streamlit UI.

The frontend talks to the backend only through these calls; it contains no AI logic.
Endpoints and fields match `app/api/routes.py` and `app/schemas/research.py`.
"""

import os

import httpx

DEFAULT_API_URL = "http://localhost:8000"
# Mirrors ResearchRequest validation so the UI can explain problems before sending.
QUESTION_MIN, QUESTION_MAX = 10, 1000
MAX_ITEMS, MAX_ITEM_CHARS = 10, 200
# An analyze=true request can include a first-time model download and load.
RESEARCH_TIMEOUT_SECONDS = 1800


def api_url_from_env() -> str:
    return os.environ.get("RESEARCHPILOT_API_URL", DEFAULT_API_URL).rstrip("/")


def make_http_client(base_url: str) -> httpx.Client:
    """Creates the HTTP client. Tests replace this to point the UI at an in-process app."""
    return httpx.Client(base_url=base_url)


class ApiError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        return response.text or response.reason_phrase
    if isinstance(detail, list):  # FastAPI validation errors
        return "; ".join(
            f"{'.'.join(str(p) for p in item.get('loc', [])[1:])}: {item.get('msg')}"
            for item in detail
        )
    return str(detail)


class ResearchPilotClient:
    def __init__(self, base_url: str | None = None, *, client: httpx.Client | None = None):
        self.base_url = (base_url or api_url_from_env()).rstrip("/")
        self._client = client or make_http_client(self.base_url)

    def _request(self, method: str, path: str, *, timeout: float = 30, **kwargs) -> httpx.Response:
        try:
            response = self._client.request(method, path, timeout=timeout, **kwargs)
        except httpx.ConnectError as exc:
            raise ApiError(
                f"Cannot reach the ResearchPilot API at {self.base_url}. Start it with "
                "`uv run uvicorn app.main:app` or set RESEARCHPILOT_API_URL."
            ) from exc
        except httpx.TimeoutException as exc:
            raise ApiError(f"The API did not respond within {timeout:.0f} seconds.") from exc
        except httpx.HTTPError as exc:
            raise ApiError(f"Request to the API failed: {exc}") from exc
        if response.status_code >= 400:
            raise ApiError(
                f"HTTP {response.status_code}: {_detail(response)}", response.status_code
            )
        return response

    def health(self) -> dict:
        return self._request("GET", "/health", timeout=5).json()

    def list_documents(self) -> list[dict]:
        return self._request("GET", "/documents").json()

    def upload_document(self, filename: str, data: bytes) -> tuple[dict, bool]:
        """Returns (DocumentInfo, created). created is False if the same file already existed."""
        response = self._request(
            "POST",
            "/documents",
            files={"file": (filename, data, "application/pdf")},
            timeout=300,  # the first upload loads the embedding model
        )
        return response.json(), response.status_code == 201

    def research(self, payload: dict) -> dict:
        return self._request(
            "POST", "/research", json=payload, timeout=RESEARCH_TIMEOUT_SECONDS
        ).json()

    def get_run(self, run_id: str) -> dict:
        return self._request("GET", f"/research/{run_id}").json()


def split_lines(text: str) -> list[str]:
    """One item per line (commas are kept: criteria like 'cost, upfront' stay whole)."""
    items: list[str] = []
    for line in text.splitlines():
        item = line.strip().lstrip("-•* ").strip()
        if item and item.lower() not in {i.lower() for i in items}:
            items.append(item)
    return items


def build_research_payload(
    question: str,
    *,
    goals: str = "",
    constraints: str = "",
    criteria: str = "",
    document_ids: list[str] | None = None,
    analyze: bool = True,
) -> tuple[dict, list[str]]:
    """Build a POST /research body. Returns (payload, problems); send only if no problems."""
    question = question.strip()
    payload = {
        "question": question,
        "goals": split_lines(goals),
        "constraints": split_lines(constraints),
        "criteria": split_lines(criteria),
        "document_ids": list(document_ids or []),
        "analyze": analyze,
    }
    problems = []
    if not question:
        problems.append("Enter a research question before pressing Analyze.")
    elif not QUESTION_MIN <= len(question) <= QUESTION_MAX:
        problems.append(
            f"The question must be {QUESTION_MIN}-{QUESTION_MAX} characters "
            f"(currently {len(question)})."
        )
    for field in ("goals", "constraints", "criteria"):
        items = payload[field]
        if len(items) > MAX_ITEMS:
            problems.append(f"At most {MAX_ITEMS} {field} are allowed ({len(items)} given).")
        long = [i for i in items if len(i) > MAX_ITEM_CHARS]
        if long:
            problems.append(f"Each of the {field} must be at most {MAX_ITEM_CHARS} characters.")
    return payload, problems
