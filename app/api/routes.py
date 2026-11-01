from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile, status
from pydantic import BaseModel, Field

from app.rag.models import DocumentInfo, RetrievedChunk
from app.rag.pdf import (
    DocumentError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from app.rag.retrieval import DocumentLibrary, UnknownDocumentError
from app.rag.vector_store import VectorStoreError
from app.schemas.research import ResearchRequest, ResearchResult
from app.services.documents import DocumentsUnavailableError
from app.services.research import ResearchService

router = APIRouter()


def get_service(request: Request) -> ResearchService:
    return request.app.state.research_service


Service = Annotated[ResearchService, Depends(get_service)]


def get_library(service: Service) -> DocumentLibrary:
    try:
        return service.documents()
    except DocumentsUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


Library = Annotated[DocumentLibrary, Depends(get_library)]


def _unknown_documents(exc: UnknownDocumentError) -> HTTPException:
    return HTTPException(
        status.HTTP_404_NOT_FOUND, f"Unknown document id(s): {', '.join(exc.document_ids)}"
    )


@router.get("/health")
def health(service: Service) -> dict:
    return {
        "status": "ok",
        "mode": "demo" if service.settings.demo_mode else "live",
        "search_provider": service.provider.name,
    }


@router.post("/research", response_model=ResearchResult)
async def create_research(body: ResearchRequest, service: Service) -> ResearchResult:
    try:
        return await service.run(body)
    except UnknownDocumentError as exc:
        raise _unknown_documents(exc) from exc
    except DocumentsUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


@router.get("/research/{run_id}", response_model=ResearchResult)
def get_research(run_id: UUID, service: Service) -> ResearchResult:
    result = service.get(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Research run not found")
    return result


@router.post("/documents", response_model=DocumentInfo, status_code=status.HTTP_201_CREATED)
def upload_document(
    library: Library, response: Response, file: Annotated[UploadFile, File()]
) -> DocumentInfo:
    """Upload a PDF. Re-uploading identical bytes returns the existing document (200)."""
    # Read one byte past the limit so oversized files are rejected without loading them fully.
    data = file.file.read(library.max_file_bytes + 1)
    try:
        info, created = library.ingest(data, file.filename or "upload.pdf")
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    except FileTooLargeError as exc:
        raise HTTPException(413, str(exc)) from exc
    except DocumentError as exc:
        raise HTTPException(422, str(exc)) from exc
    except VectorStoreError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    if not created:
        response.status_code = status.HTTP_200_OK
    return info


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents(library: Library) -> list[DocumentInfo]:
    return library.list_documents()


@router.get("/documents/{document_id}", response_model=DocumentInfo)
def get_document(document_id: str, library: Library) -> DocumentInfo:
    info = library.get_document(document_id)
    if info is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return info


class DocumentSearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_ids: list[str] | None = None


@router.post("/documents/search", response_model=list[RetrievedChunk])
def search_documents(body: DocumentSearchRequest, library: Library) -> list[RetrievedChunk]:
    try:
        return library.retrieve(body.query, body.top_k, body.document_ids)
    except UnknownDocumentError as exc:
        raise _unknown_documents(exc) from exc
    except VectorStoreError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
