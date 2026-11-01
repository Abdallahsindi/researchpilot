"""Models for the Analyst, Verifier, and Report Writer stages of a research run."""

from enum import StrEnum

from pydantic import BaseModel, Field


class AnalystFinding(BaseModel):
    """One claim proposed by the Analyst, with the run source IDs it relies on."""

    claim: str = Field(min_length=1, max_length=600)
    source_ids: list[str] = Field(default_factory=list, max_length=6)
    criterion: str | None = Field(
        default=None, description="Which of the user's criteria this finding informs, if any."
    )


class AnalystOutput(BaseModel):
    """The Analyst's structured reply, validated before anything else reads it."""

    answer: str = Field(min_length=1, max_length=1500)
    findings: list[AnalystFinding] = Field(default_factory=list, max_length=8)
    recommendation: str | None = Field(default=None, max_length=1500)
    limitations: list[str] = Field(default_factory=list, max_length=8)
    insufficient_evidence: bool = False


class AnalysisAttempt(BaseModel):
    """One Analyst call. Kept for every attempt so revisions are auditable."""

    attempt: int
    raw_output: str | None = None
    output: AnalystOutput | None = None
    parse_error: str | None = None
    invented_source_ids: list[str] = Field(
        default_factory=list, description="IDs the model used that are not sources of this run."
    )
    feedback: str | None = Field(
        default=None, description="Verifier feedback that this attempt was asked to address."
    )
    duration_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ReviewJudgement(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_INFO = "not_enough_info"
    UNAVAILABLE = "unavailable"  # review disabled, failed, timed out, or unparseable


class ReviewStatus(StrEnum):
    """Why the optional LLM review of a claim did or did not run."""

    COMPLETED = "completed"  # a review ran and produced a usable judgement
    SKIPPED = "skipped"  # deliberately not run (off-topic, budget, or an earlier check failed)
    FAILED = "failed"  # attempted, but errored, timed out, or gave no usable judgement
    NOT_PERFORMED = "not_performed"  # disabled or unavailable, so no review was attempted
    NOT_RECORDED = "not_recorded"  # default for results persisted before this field existed


class SourceReview(BaseModel):
    """LLM-assisted judgement of one (claim, cited source) pair."""

    source_id: str
    judgement: ReviewJudgement
    raw_output: str | None = None


class VerificationStatus(StrEnum):
    VERIFIED = "verified"  # references valid, code checks pass, and the review agrees
    UNVERIFIED = "unverified"  # references valid but support could not be confirmed
    UNSUPPORTED = "unsupported"  # cited text does not back the claim
    CONTRADICTED = "contradicted"  # a cited source contradicts the claim, or sources conflict
    INVALID_REFERENCE = "invalid_reference"  # no valid citation, or metadata does not match


class ClaimVerification(BaseModel):
    claim: str
    criterion: str | None = None
    status: VerificationStatus
    explanations: list[str] = Field(default_factory=list)
    cited_source_ids: list[str] = Field(description="Valid source IDs of this run.")
    invalid_source_ids: list[str] = Field(default_factory=list)
    lexical_overlap: float | None = Field(
        default=None, description="Share of the claim's content words found in cited text."
    )
    missing_numbers: list[str] = Field(default_factory=list)
    code_checks_passed: bool = Field(
        default=False,
        description="References, metadata, numbers, and word overlap all passed (no LLM).",
    )
    reviews: list[SourceReview] = Field(default_factory=list)
    review_status: ReviewStatus = Field(
        default=ReviewStatus.NOT_RECORDED,
        description="Whether the LLM review completed, was skipped, failed, or was not performed.",
    )
    review_reason: str | None = Field(
        default=None, description="The reason for review_status, set by the Verifier."
    )
    addresses_question: bool = Field(
        default=True,
        description="False if the claim shares no term with the question or criteria.",
    )
    synthetic_only: bool = Field(
        default=False, description="Every cited source is a Demo-mode synthetic entry."
    )


class VerificationReport(BaseModel):
    claims: list[ClaimVerification] = Field(default_factory=list)
    answer_check: ClaimVerification | None = None
    recommendation_check: ClaimVerification | None = None
    llm_review_enabled: bool = False
    needs_revision: bool = False
    feedback: str | None = None

    def count(self, status: VerificationStatus) -> int:
        return sum(1 for c in self.claims if c.status is status)


class AnalysisStatus(StrEnum):
    COMPLETED = "completed"  # at least one verified finding
    NO_VERIFIED_FINDINGS = "no_verified_findings"  # analysis ran, but nothing passed
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"  # skipped, or the Analyst abstained
    FAILED = "failed"  # model unavailable, error, timeout, or malformed after retries


class AnalysisMetrics(BaseModel):
    analyst_calls: int = 0
    review_calls: int = 0
    llm_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class AnalysisResult(BaseModel):
    status: AnalysisStatus
    model_id: str | None = None
    attempts: list[AnalysisAttempt] = Field(default_factory=list)
    verification: VerificationReport | None = None
    errors: list[str] = Field(default_factory=list)
    metrics: AnalysisMetrics = Field(default_factory=AnalysisMetrics)

    @property
    def final_output(self) -> AnalystOutput | None:
        for attempt in reversed(self.attempts):
            if attempt.output is not None:
                return attempt.output
        return None
