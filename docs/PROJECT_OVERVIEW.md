# ResearchPilot project overview

## Purpose

ResearchPilot is a research and decision-support application that answers questions from
evidence instead of relying on a language model's memory. It accepts uploaded PDFs and uses
a configurable search provider, retrieves relevant passages, asks a local model to analyse
them, checks every citation, and produces a Markdown report. The report distinguishes three
things that are often mixed together:

- **Verified findings:** a real source of this run, matching PDF page metadata, numbers and
  wording present in the source, a relevant subject, no instruction-like source text, and a
  supporting LLM review.
- **Unverified observations:** plausible model claims that did not pass every check.
- **Model-written recommendations and limitations:** useful context, but not verified facts.

The application says so when the evidence is insufficient. It is designed for the case where
an answer that cannot be traced to a source is worse than no answer.

## What is implemented

- FastAPI backend with PDF upload, document listing, document search, research runs, and run
  retrieval.
- LangGraph workflow with parallel web and PDF retrieval, evidence assessment, and a bounded
  Analyst → Verifier → Report Writer loop.
- Page-level PDF extraction, overlapping chunks, semantic embeddings, and a local FAISS index,
  with an optional pgvector adapter.
- Local `Qwen/Qwen2.5-1.5B-Instruct` inference through Hugging Face Transformers. The model is
  loaded lazily when an analysis request first needs it.
- Deterministic citation, number, subject, metadata, and prompt-injection checks, plus an
  optional LLM review per claim and source.
- A deterministic Markdown report that separates verified findings, unverified observations,
  rejected claims, missing decision criteria, sources, and verification details.
- A Streamlit UI that uploads through the real API, displays registered documents, labels the
  sources a run actually used, shows verification reasons, and downloads the report.
- An offline Demo corpus for development and a synthetic PDF with questions and expected
  pages. The UI labels these as sample data; it does not present them as live web research.
- A local VS Code walkthrough notebook that imports the real modules, plus a separate Kaggle
  notebook that is preserved for the original demonstration.

## Technology and responsibilities

| Technology | Role |
| --- | --- |
| Python 3.11+ | Runtime |
| FastAPI + Pydantic | HTTP API, request validation, and result schemas |
| LangGraph | Typed state, parallel branches, conditional routing, and the revision loop |
| `pypdf` | Page-level PDF text extraction |
| `sentence-transformers/all-MiniLM-L6-v2` | 384-dimensional semantic embeddings |
| FAISS | Local cosine-similarity vector index |
| Qwen2.5-1.5B-Instruct + Transformers | Local Analyst generation and Verifier review |
| Streamlit | Browser interface; a thin client over the API |
| pytest + Ruff | Offline regression tests and lint/format checks |

Tavily search and Supabase pgvector adapters are implemented behind interfaces. They are not
verified against the real external services in this release.

## Key design decisions

### Retrieval before generation

Only retrieved passages are sent to the model. Each passage is labelled with a source ID, and
the model may cite only that ID. This gives code a finite set of valid references to check.

### Deterministic report rendering

The Report Writer does not call a model. It can only format data produced by the workflow, so
it cannot add a new claim, number, link, or confidence score. When a requested criterion has no
verified evidence, it withholds the model's recommendation and generates a conservative
statement naming the missing evidence.

### Verification is deliberately conservative

Word overlap alone never verifies a claim. A claim that cites a real page but cannot be
confirmed stays `unverified`. This produces fewer verified findings than a human reviewer might
accept, but it prevents a valid-looking citation from being presented as proof.

### Provenance is per run

The configured search mode and the sources actually used are separate. A Demo-mode
configuration does not mean a PDF-only run used synthetic data. The UI labels each run by its
actual evidence mix.

### Local-first operation

The default configuration needs no paid API or cloud service. Models run locally; tests replace
the LLM, embeddings, search provider, and vector store with offline stand-ins. Live search and
pgvector are optional adapters.

## Project structure

```
app/
  agents/       planner, analyst, verifier, report writer
  api/          FastAPI routes
  graph/        LangGraph state and workflow
  rag/          PDF, chunking, embeddings, vector stores, retrieval, QA evaluation
  schemas/      Pydantic models shared across the application
  services/     orchestration, documents, LLM loading, storage
  tools/        web-search adapters
frontend/       Streamlit UI, HTTP client, pure display helpers
notebooks/      local instructor walkthrough and preserved Kaggle notebook
sample_data/    synthetic test PDF, questions, expected pages
docs/           architecture, overview, demo script, Kaggle checklist
scripts/        sample data generation and end-to-end API demo
tests/          offline regression suite
data/           git-ignored runtime runs and document index
```

## Known limitations

- **Verification is heuristic.** Lexical overlap cannot detect every change of meaning, and a
  1.5B review model can miss a subject swap. "Verified" means the checks passed, not that the
  claim is objectively true.
- **Conservative recall.** Correct paraphrases can remain unverified when their wording does
  not overlap enough with the cited text.
- **Small model.** Qwen2.5-1.5B can produce malformed JSON, over-cite, or give an inconsistent
  recommendation. The schema and report logic contain those failures but do not eliminate
  them.
- **Retrieval scope.** Only the top `DOCUMENT_TOP_K` chunks per planned query and the top
  `ANALYSIS_MAX_SOURCES` sources reach the agents. A relevant passage outside that window is
  invisible. Fixed-size chunks can also stop mid-sentence; the verifier rejects a claim that
  ends in a dangling fragment, but it cannot recover the continuation that was not retrieved.
- **No OCR.** Image-only PDF pages are reported and skipped.
- **External integrations.** Live Tavily search and real Supabase pgvector are not verified.
  Kaggle execution of the preserved notebook is not verified for this release.
- **One model at a time.** Analysis requests share a serialized model, so concurrent requests
  queue. A timed-out generation may continue in its worker thread until it finishes.

## Evaluation approach

The offline suite checks workflow routing, schema validation, citation and injection checks,
revision limits, persistence, the API, the UI, and notebook structure. Real-model runs on the
synthetic PDF are reported separately with their actual timings and statuses. The sample
evaluation set is small and was used during prompt tuning, so it is a sanity check rather than
a benchmark.
