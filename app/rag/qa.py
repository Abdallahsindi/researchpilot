"""Evidence-grounded question answering over retrieved PDF chunks.

How citations stay trustworthy:
1. Retrieved chunks are shown to the model as passages with short IDs: E1, E2, ...
2. The model must reply in a fixed format: an ANSWER line and an EVIDENCE line listing the
   passage IDs it used. (Small models follow this template far more reliably than inline
   citations; that was observed when testing Qwen2.5-1.5B on the sample report.)
3. Code parses the IDs and resolves each one to the chunk actually retrieved for this
   question. Filename and page number come from that chunk's stored metadata, never from
   text the model wrote. IDs that were not in the prompt are discarded and reported.
4. Each answer sentence is checked against the cited passages: if most of its content words
   do not appear in them, the sentence is flagged as weakly supported. Each cited passage is
   also checked on its own; one that shares little wording with the answer is kept visible
   but marked as not supporting it (small models tend to over-cite).
5. An answer with no valid citation, or with no supported sentence, is withheld.

Steps 3-4 prove the cited passages were retrieved and share the answer's wording. They do
not prove logical entailment. The multi-agent workflow adds the implemented
`app.agents.verifier` stage, which combines code checks with an optional LLM review. This
module remains the standalone document-QA path used by the notebooks and evaluation harness.
"""

import re
from enum import StrEnum
from time import perf_counter
from typing import Protocol

from pydantic import BaseModel, Field

from app.rag.models import RetrievedChunk
from app.services.llm import LLM

ABSTAIN_TOKEN = "INSUFFICIENT_EVIDENCE"
SUPPORT_THRESHOLD = 0.5
# On the sample report, passages that really supported an answer shared 56-100% of its
# content words; passages the model cited without using them shared 11-18%.
CITATION_SUPPORT_THRESHOLD = 0.3

SYSTEM_PROMPT = f"""You answer questions using only the evidence passages provided. \
Each passage has an ID such as E1.

Reply in exactly this format:
ANSWER: <one to three sentences using only facts from the passages>
EVIDENCE: <comma-separated IDs of the passages that contain those facts>

If the passages do not contain the answer, reply exactly:
ANSWER: {ABSTAIN_TOKEN}
EVIDENCE: none

Do not use outside knowledge. Only use passage IDs that appear below."""

FORMAT_REMINDER = "Reply with the ANSWER line and then the EVIDENCE line."

# Tolerates markdown bold ("**ANSWER:**"), which small models sometimes add.
_ANSWER_PREFIX_RE = re.compile(r"^\s*(?:\*\*)?ANSWER(?:\*\*)?\s*:(?:\*\*)?\s*", re.IGNORECASE)
_EVIDENCE_RE = re.compile(r"(?:^|\s)(?:\*\*)?EVIDENCE(?:\*\*)?\s*:(?:\*\*)?(.*)$", re.DOTALL)
_ID_RE = re.compile(r"\bE(\d+)\b", re.IGNORECASE)
_INLINE_RE = re.compile(r"\[\s*(E\d+(?:\s*[,;]\s*E\d+)*)\s*\]", re.IGNORECASE)
# Split after end punctuation followed by whitespace, so "1.2 MW" stays one sentence.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOPWORDS = frozenset(
    "the and for was were are has have had with that this from which its their they them "
    "than then also been being into about over under such who what when where why how "
    "can could would should will not but all any each per our your his her she him".split()
)


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]: ...


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNSUPPORTED = "unsupported"


class Citation(BaseModel):
    label: str
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    score: float
    excerpt: str
    overlap: float = Field(description="Share of the answer's content words in this passage.")
    supports_answer: bool


class SentenceSupport(BaseModel):
    sentence: str
    overlap: float = Field(description="Share of the sentence's content words in cited text.")
    supported: bool


class QAMetrics(BaseModel):
    retrieval_ms: float
    generation_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0


class QAResult(BaseModel):
    question: str
    status: AnswerStatus
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    invalid_citations: list[str] = Field(
        default_factory=list, description="IDs the model used that were not in the prompt."
    )
    evidence_support: float | None = Field(
        default=None,
        description="Share of answer sentences whose content words mostly appear in the cited "
        "passages. A lexical heuristic, not proof of entailment.",
    )
    sentence_support: list[SentenceSupport] = Field(default_factory=list)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    evidence_labels: dict[str, str] = Field(
        default_factory=dict, description="Prompt passage ID -> chunk_id."
    )
    raw_output: str | None = None
    prompt: list[dict[str, str]] | None = None
    model_id: str | None = None
    warnings: list[str] = Field(default_factory=list)
    metrics: QAMetrics

    @property
    def supporting_citations(self) -> list[Citation]:
        return [c for c in self.citations if c.supports_answer]

    @property
    def cited_pages(self) -> list[tuple[str, int]]:
        return sorted({(c.filename, c.page_number) for c in self.supporting_citations})


def build_prompt(question: str, evidence: list[RetrievedChunk]) -> list[dict[str, str]]:
    """Chat messages containing the question and only the selected evidence."""
    passages = "\n\n".join(
        f"Passage E{i}:\n{item.chunk.text}" for i, item in enumerate(evidence, start=1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{passages}\n\nQuestion: {question}\n\n{FORMAT_REMINDER}"},
    ]


def parse_response(raw: str) -> tuple[str, list[str]]:
    """Split a model reply into (answer text, passage IDs in the order cited)."""
    evidence_match = _EVIDENCE_RE.search(raw)
    evidence = evidence_match.group(1) if evidence_match else ""
    answer = raw[: evidence_match.start()] if evidence_match else raw
    answer = _ANSWER_PREFIX_RE.sub("", answer)

    labels: list[str] = []
    inline = " ".join(_INLINE_RE.findall(answer))
    for number in _ID_RE.findall(f"{evidence} {inline}"):
        label = f"E{int(number)}"
        if label not in labels:
            labels.append(label)
    return answer.strip(), labels


def content_words(text: str) -> set[str]:
    words = _WORD_RE.findall(text.lower())
    return {w for w in words if w not in _STOPWORDS and (len(w) >= 3 or w[0].isdigit())}


def check_support(answer: str, cited_texts: list[str]) -> list[SentenceSupport]:
    """Lexical support check: does each sentence's wording come from the cited passages?"""
    cited_words = content_words(" ".join(cited_texts))
    results = []
    for sentence in _SENTENCE_SPLIT_RE.split(_INLINE_RE.sub("", answer)):
        sentence = sentence.strip()
        words = content_words(sentence)
        if not words:
            continue
        overlap = len(words & cited_words) / len(words)
        results.append(
            SentenceSupport(
                sentence=sentence,
                overlap=round(overlap, 3),
                supported=overlap >= SUPPORT_THRESHOLD,
            )
        )
    return results


def _remove_invalid_inline(answer: str, valid: set[str]) -> str:
    def replace(match: re.Match) -> str:
        kept = [f"E{int(n)}" for n in _ID_RE.findall(match.group(1))]
        return "".join(f"[{label}]" for label in kept if label in valid)

    return re.sub(r" {2,}", " ", _INLINE_RE.sub(replace, answer)).strip()


def resolve_answer(raw_output: str, evidence: list[RetrievedChunk]) -> dict:
    """Validate the model's passage IDs against the retrieved evidence and decide the status."""
    label_map = {f"E{i}": item for i, item in enumerate(evidence, start=1)}
    answer, labels = parse_response(raw_output)
    valid = [label for label in labels if label in label_map]
    invalid = [label for label in labels if label not in label_map]
    abstained = ABSTAIN_TOKEN in answer.upper().replace(" ", "_")
    warnings: list[str] = []
    if invalid:
        warnings.append(f"Discarded passage IDs that were not in the prompt: {', '.join(invalid)}.")
    result = {"invalid_citations": invalid, "warnings": warnings}

    if abstained and not valid:
        warnings.append("The model abstained after reading the retrieved evidence.")
        return {
            **result,
            "status": AnswerStatus.INSUFFICIENT_EVIDENCE,
            "answer": "The retrieved passages do not contain enough evidence to answer this.",
        }
    if not valid:
        warnings.append("No valid citation; the raw output is unverified.")
        return {
            **result,
            "status": AnswerStatus.UNSUPPORTED,
            "answer": "The model's answer did not cite any retrieved passage, so it cannot be "
            "checked against the document and is withheld. See raw_output.",
        }

    answer = _remove_invalid_inline(answer, set(valid))
    if abstained:
        warnings.append("The model reported that part of the question is not covered.")
    support = check_support(answer, [label_map[label].chunk.text for label in valid])
    supported = [s for s in support if s.supported]
    ratio = round(len(supported) / len(support), 3) if support else None
    answer_words = content_words(_INLINE_RE.sub("", answer))
    citations = []
    for label in valid:
        chunk = label_map[label].chunk
        overlap = (
            len(answer_words & content_words(chunk.text)) / len(answer_words)
            if answer_words
            else 0.0
        )
        citations.append(
            Citation(
                label=label,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                score=round(label_map[label].score, 4),
                excerpt=chunk.text[:240],
                overlap=round(overlap, 3),
                supports_answer=overlap >= CITATION_SUPPORT_THRESHOLD,
            )
        )
    loose = [f"{c.label} (page {c.page_number})" for c in citations if not c.supports_answer]
    if loose:
        warnings.append(
            f"Cited but sharing little wording with the answer: {', '.join(loose)}; "
            "not counted as support."
        )
    result.update(citations=citations, evidence_support=ratio, sentence_support=support)

    if support and not supported:
        warnings.append("None of the answer's sentences are supported by the cited passages.")
        return {
            **result,
            "status": AnswerStatus.UNSUPPORTED,
            "answer": "The model cited passages, but its wording does not match them, so the "
            "answer is withheld. See raw_output and sentence_support.",
        }
    weak = [s.sentence for s in support if not s.supported]
    if weak:
        warnings.append(f"{len(weak)} sentence(s) are weakly supported by the cited passages.")
    return {**result, "status": AnswerStatus.ANSWERED, "answer": answer}


def answer_question(
    question: str,
    retriever: Retriever,
    llm: LLM,
    *,
    top_k: int = 4,
    max_new_tokens: int = 256,
    include_prompt: bool = True,
) -> QAResult:
    started = perf_counter()
    evidence = retriever.retrieve(question, top_k=top_k)
    metrics = QAMetrics(retrieval_ms=round((perf_counter() - started) * 1000, 1))

    if not evidence:
        # Nothing cleared the similarity threshold: abstain without calling the model.
        return QAResult(
            question=question,
            status=AnswerStatus.INSUFFICIENT_EVIDENCE,
            answer="No passage in the indexed documents is relevant enough to answer this.",
            warnings=["No chunk met the retrieval similarity threshold; the LLM was not called."],
            model_id=llm.model_id,
            metrics=metrics,
        )

    messages = build_prompt(question, evidence)
    generation = llm.generate(messages, max_new_tokens=max_new_tokens)
    metrics = metrics.model_copy(
        update={
            "generation_ms": generation.duration_ms,
            "prompt_tokens": generation.prompt_tokens,
            "completion_tokens": generation.completion_tokens,
            "llm_calls": 1,
        }
    )
    return QAResult(
        question=question,
        retrieved=evidence,
        evidence_labels={f"E{i}": item.chunk.chunk_id for i, item in enumerate(evidence, 1)},
        raw_output=generation.text,
        prompt=messages if include_prompt else None,
        model_id=llm.model_id,
        metrics=metrics,
        **resolve_answer(generation.text, evidence),
    )


def render_markdown(result: QAResult) -> str:
    """Human-readable answer with citations resolved to filename and page."""
    lines = [f"**Question:** {result.question}", f"**Status:** `{result.status.value}`", ""]
    lines.append(result.answer)
    if result.supporting_citations:
        lines += ["", "**Sources (resolved from retrieved chunks, not from model text):**"]
        lines += [
            f"- {c.label} → {c.filename}, page {c.page_number} — `{c.chunk_id}` "
            f"(similarity {c.score:.3f}, answer-word overlap {c.overlap:.0%})"
            for c in result.supporting_citations
        ]
    loose = [c for c in result.citations if not c.supports_answer]
    if loose:
        lines += ["", "*Also cited by the model, but not supporting the answer:* "]
        lines[-1] += ", ".join(
            f"{c.label} → page {c.page_number} ({c.overlap:.0%} overlap)" for c in loose
        )
    if result.evidence_support is not None:
        lines += ["", f"**Evidence support:** {result.evidence_support:.0%} of sentences"]
    if result.warnings:
        lines += ["", *[f"> ⚠️ {w}" for w in result.warnings]]
    return "\n".join(lines)
