"""Deterministic research planner.

The planner creates one query for the core question, one for each distinct requested part,
one per comparison criterion, and one per goal. It deliberately uses no LLM: query planning
is fast, reproducible, and testable offline. A future extension could generate the same
`ResearchPlan` shape with a model.
"""

import re

from app.schemas.research import ResearchPlan, ResearchRequest, SearchQuery

# Split a multi-part question before a new question/instruction verb, either after a comma
# or after "and". This keeps "signed and unsigned integers" together because "unsigned" is
# not a clause opener.
_PART_SPLIT_RE = re.compile(
    r"(?:;|,\s*and\s+|,\s*|\s+and\s+)(?=(?:what|how|why|which|when|where|who|explain|"
    r"describe|define|list|compare|contrast|discuss|give|state|summari[sz]e|outline|"
    r"identify)\b)",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_META_INSTRUCTION_RE = re.compile(
    r"\b(?:support each|cite|citation|if the document|if the source|clearly state|"
    r"use only|do not use|reply|answer with)\b",
    re.IGNORECASE,
)
_INSTRUCTION_PREFIX_RE = re.compile(
    r"^(?:explain|describe|define|list|compare|contrast|discuss|summari[sz]e|outline|"
    r"identify|state|give)\s+(?:the\s+)?",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(
    r"\b(?:difference(?:s)?\s+between|distinguish(?:es|ed)?(?:\s+between)?)\s+"
    r"(.+?)\s+and\s+(.+?)\s*$",
    re.IGNORECASE,
)
_TOPIC_WORD_RE = re.compile(r"[a-z0-9]+")
_QUESTION_FILLER = frozenset(
    {
        "according",
        "answer",
        "between",
        "briefly",
        "citation",
        "citations",
        "cite",
        "clearly",
        "component",
        "components",
        "could",
        "define",
        "describe",
        "detail",
        "did",
        "does",
        "difference",
        "differences",
        "discuss",
        "document",
        "documents",
        "each",
        "enough",
        "evidence",
        "explain",
        "give",
        "how",
        "identify",
        "inside",
        "lecture",
        "list",
        "main",
        "mention",
        "outline",
        "page",
        "pages",
        "pdf",
        "provide",
        "question",
        "relevant",
        "source",
        "sources",
        "state",
        "step",
        "steps",
        "support",
        "tell",
        "that",
        "the",
        "their",
        "these",
        "this",
        "those",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
    }
)


def _has_topic(part: str) -> bool:
    words = {w for w in _TOPIC_WORD_RE.findall(part.casefold()) if len(w) >= 3}
    return bool(words - _QUESTION_FILLER)


def _comparison_parts(part: str) -> list[str]:
    """Split an explicit comparison into a query for each side when that is useful."""
    match = _COMPARISON_RE.search(part)
    if not match:
        return [part]
    left, right = (group.strip(" \t\r\n,;.?!") for group in match.groups())
    if not left or not right:
        return [part]
    # "signed and unsigned integers" describes two kinds of integer. Give the short first
    # side the shared head noun so it is a useful retrieval query on its own.
    left_words, right_words = left.split(), right.split()
    if len(left_words) == 1 and len(right_words) > 1:
        left = f"{left} {right_words[-1]}"
    return [left, right]


def _question_parts(question: str) -> list[str]:
    """Deterministically split a question into distinct requested parts."""
    parts: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(question.strip()):
        if _META_INSTRUCTION_RE.search(sentence) and not re.search(
            r"\b(?:what|how|why|which|when|where|who|explain|describe|define|list|"
            r"compare|discuss|identify)\b",
            sentence,
            re.IGNORECASE,
        ):
            continue
        for clause in _PART_SPLIT_RE.split(sentence):
            clause = clause.strip(" \t\r\n,;.")
            clause = re.sub(r"^(?:and|also)\s+", "", clause, flags=re.IGNORECASE)
            clause = _INSTRUCTION_PREFIX_RE.sub("", clause).strip()
            for part in _comparison_parts(clause):
                if _has_topic(part):
                    parts.append(part)
    return parts


def plan_research(request: ResearchRequest, max_queries: int) -> ResearchPlan:
    candidates = [(request.question, "Answer the core question")]
    parts = _question_parts(request.question)
    if len(parts) > 1:
        candidates += [
            (part, f"Answer requested part {index}") for index, part in enumerate(parts, start=1)
        ]
    candidates += [
        (f"{request.question} {criterion}", f"Evidence on criterion: {criterion}")
        for criterion in request.criteria
    ]
    candidates += [
        (f"{request.question} {goal}", f"Evidence for goal: {goal}") for goal in request.goals
    ]

    queries: list[SearchQuery] = []
    seen: set[str] = set()
    for text, purpose in candidates:
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        queries.append(SearchQuery(id=f"Q{len(queries) + 1}", text=text, purpose=purpose))
        if len(queries) == max_queries:
            break

    return ResearchPlan(
        objective=request.question,
        queries=queries,
        goals=request.goals,
        constraints=request.constraints,
        criteria=request.criteria,
    )
