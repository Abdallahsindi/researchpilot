"""Grounded QA tests. The LLM is a scripted fake: these tests check the citation and
abstention logic around the model, not the quality of a real model's answers."""

import pytest

from app.rag.chunking import chunk_document
from app.rag.embeddings import HashingEmbeddings
from app.rag.evaluation import EvalCase, evaluate_qa, evaluate_retrieval, summarize
from app.rag.pdf import parse_pdf
from app.rag.qa import (
    ABSTAIN_TOKEN,
    AnswerStatus,
    answer_question,
    build_prompt,
    check_support,
    parse_response,
    render_markdown,
)
from app.rag.retrieval import VectorRetriever
from app.rag.vector_store import FaissVectorStore
from tests.conftest import FakeLLM


@pytest.fixture
def retriever(sample_pdf) -> VectorRetriever:
    embedder = HashingEmbeddings()
    store = FaissVectorStore(embedder.dimension)
    chunks = chunk_document(parse_pdf(sample_pdf, "report.pdf"))
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    return VectorRetriever(embedder, store, min_score=0.1)


BATTERY_Q = "Why was lithium iron phosphate chosen for the battery?"


def test_valid_citations_resolve_to_retrieved_pages(retriever):
    llm = FakeLLM(
        "ANSWER: The battery uses lithium iron phosphate cells, chosen for safety.\nEVIDENCE: E1"
    )

    result = answer_question(BATTERY_Q, retriever, llm, top_k=2)

    assert result.status is AnswerStatus.ANSWERED
    assert result.answer == "The battery uses lithium iron phosphate cells, chosen for safety."
    assert [c.label for c in result.citations] == ["E1"]
    citation = result.citations[0]
    assert citation.chunk_id == result.retrieved[0].chunk.chunk_id
    assert (citation.filename, citation.page_number) == ("report.pdf", 2)
    assert result.evidence_labels["E1"] == citation.chunk_id
    assert result.evidence_support == 1.0
    assert result.metrics.llm_calls == 1 and result.metrics.prompt_tokens == 100


def test_fabricated_ids_are_discarded_and_reported(retriever):
    llm = FakeLLM("ANSWER: Lithium iron phosphate was chosen for safety.\nEVIDENCE: E1, E7, E9")

    result = answer_question(BATTERY_Q, retriever, llm, top_k=2)

    assert result.status is AnswerStatus.ANSWERED
    assert result.invalid_citations == ["E7", "E9"]
    assert [c.label for c in result.citations] == ["E1"]
    assert any("E7, E9" in w for w in result.warnings)


def test_inline_fabricated_ids_are_removed_from_answer(retriever):
    llm = FakeLLM("Lithium iron phosphate was chosen for safety [E1][E7].")

    result = answer_question(BATTERY_Q, retriever, llm, top_k=2)

    assert result.status is AnswerStatus.ANSWERED
    assert "[E7]" not in result.answer and "[E1]" in result.answer
    assert result.invalid_citations == ["E7"]


def test_answer_without_valid_citations_is_withheld(retriever):
    llm = FakeLLM("ANSWER: The battery is made by Tesla (page 7).\nEVIDENCE: none")

    result = answer_question(BATTERY_Q, retriever, llm, top_k=2)

    assert result.status is AnswerStatus.UNSUPPORTED
    assert result.citations == []
    assert "Tesla" not in result.answer
    assert result.raw_output == llm.output  # kept for transparency, clearly unverified


def test_cited_but_unrelated_answer_is_withheld(retriever):
    llm = FakeLLM("ANSWER: The containers are painted green and guarded by robots.\nEVIDENCE: E1")

    result = answer_question(BATTERY_Q, retriever, llm, top_k=2)

    assert result.status is AnswerStatus.UNSUPPORTED
    assert result.evidence_support == 0.0
    assert result.sentence_support[0].supported is False


def test_weakly_supported_sentences_are_flagged(retriever):
    llm = FakeLLM(
        "ANSWER: Lithium iron phosphate cells were chosen for safety. "
        "The supplier is Acme Robotics of Mars.\nEVIDENCE: E1"
    )

    result = answer_question(BATTERY_Q, retriever, llm, top_k=2)

    assert result.status is AnswerStatus.ANSWERED
    assert result.evidence_support == 0.5
    assert [s.supported for s in result.sentence_support] == [True, False]
    assert any("weakly supported" in w for w in result.warnings)


def test_model_abstention_is_reported(retriever):
    llm = FakeLLM(f"ANSWER: {ABSTAIN_TOKEN}\nEVIDENCE: none")

    result = answer_question("What colour are the battery containers?", retriever, llm)

    assert result.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert result.citations == [] and result.metrics.llm_calls == 1


def test_partial_abstention_keeps_cited_part(retriever):
    llm = FakeLLM(
        f"ANSWER: Lithium iron phosphate was chosen for safety. The supplier is {ABSTAIN_TOKEN}."
        "\nEVIDENCE: E1"
    )

    result = answer_question(BATTERY_Q, retriever, llm, top_k=2)

    assert result.status is AnswerStatus.ANSWERED
    assert any("not covered" in w for w in result.warnings)


def test_no_relevant_evidence_skips_the_llm(retriever):
    llm = FakeLLM("should never be called")

    result = answer_question("sourdough bread starter recipe", retriever, llm)

    assert result.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert llm.calls == [] and result.retrieved == []


def test_prompt_contains_only_selected_evidence(retriever):
    llm = FakeLLM("ANSWER: x\nEVIDENCE: E1")

    answer_question(BATTERY_Q, retriever, llm, top_k=1)

    system, user = llm.calls[0]
    assert system["role"] == "system" and ABSTAIN_TOKEN in system["content"]
    assert "Passage E1:" in user["content"] and "Passage E2" not in user["content"]
    assert "lithium iron phosphate" in user["content"]
    assert "3.1 million" not in user["content"]  # unretrieved page is not in the prompt
    assert f"Question: {BATTERY_Q}" in user["content"]


def test_response_parsing_and_support_helpers():
    assert parse_response("ANSWER: X is 1.2 MW.\nEVIDENCE: E1, E3") == (
        "X is 1.2 MW.",
        ["E1", "E3"],
    )
    assert parse_response("**ANSWER:** Y [e2].\n**EVIDENCE:** E2") == ("Y [e2].", ["E2"])
    assert parse_response("Plain [E1; E4] text") == ("Plain [E1; E4] text", ["E1", "E4"])
    assert parse_response("The evidence: suggests X") == ("The evidence: suggests X", [])
    support = check_support("Capacity is 1.2 megawatts. Owned by aliens.", ["peak 1.2 megawatts"])
    assert [s.supported for s in support] == [True, False]
    assert "Question: Q?" in build_prompt("Q?", [])[1]["content"]


def test_render_markdown_shows_resolved_pages(retriever):
    llm = FakeLLM("ANSWER: Lithium iron phosphate was chosen for safety.\nEVIDENCE: E1")
    result = answer_question(BATTERY_Q, retriever, llm, top_k=1)

    text = render_markdown(result)

    assert "report.pdf, page 2" in text and result.citations[0].chunk_id in text


def test_evaluation_metrics_from_runs(retriever):
    cases = [
        EvalCase(id="battery", question=BATTERY_Q, answerable=True, expected_pages=[2]),
        EvalCase(id="none", question="sourdough bread starter recipe", answerable=False),
    ]
    llm = FakeLLM("ANSWER: Lithium iron phosphate was chosen for safety.\nEVIDENCE: E1, E5")

    retrieval = evaluate_retrieval(retriever, cases, top_k=2)
    qa = evaluate_qa(lambda q: answer_question(q, retriever, llm, top_k=2), cases)
    summary = summarize(retrieval, qa)

    assert retrieval[0].hit is True and retrieval[1].hit is None
    assert [r.correct_decision for r in qa] == [True, True]
    assert qa[0].cited_expected_page is True
    assert summary["retrieval_hit_rate"] == 1.0
    assert summary["decision_accuracy"] == 1.0
    assert summary["citation_validity"] == 0.5  # one valid (E1), one fabricated (E5)
    assert summary["mean_evidence_support"] == 1.0


def test_abstention_spelling_variants(retriever):
    # Qwen2.5-1.5B replied "Insufficient Evidence" in a real CPU run.
    for output in ("ANSWER: Insufficient Evidence\nEVIDENCE: none", "insufficient evidence"):
        result = answer_question("What colour are the containers?", retriever, FakeLLM(output))
        assert result.status is AnswerStatus.INSUFFICIENT_EVIDENCE


def test_over_cited_passage_is_marked_not_supporting(retriever):
    # Mirrors a real Qwen run: the correct passage plus an unrelated one listed as evidence.
    llm = FakeLLM("ANSWER: Lithium iron phosphate cells were chosen for safety.\nEVIDENCE: E1, E2")

    result = answer_question(BATTERY_Q, retriever, llm, top_k=2)

    assert result.status is AnswerStatus.ANSWERED
    by_label = {c.label: c for c in result.citations}
    assert by_label["E1"].supports_answer and by_label["E1"].page_number == 2
    assert not by_label["E2"].supports_answer
    assert [c.label for c in result.supporting_citations] == ["E1"]
    assert result.cited_pages == [("report.pdf", 2)]
    assert any("not counted as support" in w for w in result.warnings)
    assert "not supporting the answer" in render_markdown(result)
