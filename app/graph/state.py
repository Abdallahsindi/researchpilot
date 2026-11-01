"""State passed between workflow nodes."""

import operator
from typing import Annotated, NotRequired, TypedDict

from app.schemas.analysis import AnalysisResult, VerificationReport
from app.schemas.research import (
    DocumentSearchOutcome,
    ResearchPlan,
    ResearchRequest,
    RunStatus,
    SearchOutcome,
    SearchQuery,
    Source,
)


class ResearchState(TypedDict):
    request: ResearchRequest
    plan: NotRequired[ResearchPlan]
    # Parallel search nodes each append their outcome; the reducer merges them.
    search_outcomes: Annotated[list[SearchOutcome], operator.add]
    document_outcomes: Annotated[list[DocumentSearchOutcome], operator.add]
    sources: NotRequired[list[Source]]
    status: NotRequired[RunStatus]
    warnings: NotRequired[list[str]]
    # Analysis stages (only when request.analyze is true)
    analysis: NotRequired[AnalysisResult]
    verification: NotRequired[VerificationReport]
    revision_feedback: NotRequired[str | None]
    analyst_failed: NotRequired[bool]  # the last Analyst call raised or timed out
    report: NotRequired[str]


class SearchTask(TypedDict):
    """Input sent to one parallel `search` node."""

    query: SearchQuery


class DocumentSearchTask(TypedDict):
    """Input sent to one parallel `document_search` node."""

    query: SearchQuery
    document_ids: list[str]
