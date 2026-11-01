# ResearchPilot architecture

ResearchPilot is a FastAPI service with a Streamlit client. A request is planned, runs web
and PDF searches in parallel, and then optionally enters a bounded multi-agent analysis
stage. The model proposes claims; code and a second model review decide whether those claims
may be presented as verified.

## End-to-end data flow

```mermaid
flowchart LR
    subgraph Client
        UI[Streamlit UI<br/>frontend/streamlit_app.py]
    end

    subgraph API[FastAPI service]
        ROUTES[app/api/routes.py]
        SERVICE[ResearchService<br/>app/services/research.py]
        STORE[(Run store<br/>data/runs/*.json)]
        DOCS[(Document library<br/>data/documents/)]
    end

    subgraph Graph[LangGraph workflow]
        PLAN[Planner<br/>deterministic queries]
        WEB[search xN<br/>Tavily or Demo corpus]
        DOC[document_search xN<br/>uploaded PDFs]
        COLLECT[collect_sources<br/>deduplicate and rank]
        ASSESS[assess_evidence]
        ANALYST[Analyst<br/>Qwen]
        VERIFIER[Verifier<br/>code checks + LLM review]
        WRITER[Report Writer<br/>deterministic Markdown]
    end

    subgraph RAG[PDF retrieval]
        PDF[load_pdf<br/>page-level text]
        CHUNK[chunk_document<br/>500 chars, 100 overlap]
        EMBED[SentenceTransformer<br/>all-MiniLM-L6-v2]
        FAISS[(FAISS index<br/>cosine similarity)]
    end

    UI -->|POST /documents| ROUTES
    UI -->|POST /research analyze=true| ROUTES
    UI -->|GET /research/{run_id}| ROUTES
    ROUTES --> SERVICE
    SERVICE --> DOCS
    DOCS --> PDF --> CHUNK --> EMBED --> FAISS
    SERVICE --> PLAN
    PLAN --> WEB
    PLAN --> DOC
    DOC --> FAISS
    WEB --> COLLECT
    DOC --> COLLECT
    COLLECT --> ASSESS
    ASSESS -->|analyze=false or insufficient evidence| WRITER
    ASSESS -->|verified route| ANALYST
    ANALYST --> VERIFIER
    VERIFIER -->|bounded revision| ANALYST
    VERIFIER --> WRITER
    WRITER --> SERVICE
    SERVICE --> STORE
    STORE --> ROUTES
    ROUTES -->|ResearchResult + report| UI
```

The graph is compiled in `app/graph/workflow.py`. The typed state is in
`app/graph/state.py`; the API schema and persisted result are in
`app/schemas/research.py`.

## Request lifecycle

1. **The UI calls the real API.** `POST /documents` registers a PDF and returns its
   `document_id`; `POST /research` sends the question, goals, constraints, criteria,
   `document_ids`, and `analyze`. The UI never invents a progress state or a result.
2. **Planning.** `app/agents/planner.py` builds one query for the question, one per
   criterion, and one per goal. It is deterministic and does not call a model.
3. **Parallel retrieval.** LangGraph fans out every query to the configured search provider
   and to `document_search` when document IDs were supplied. Web results and PDF chunks are
   deduplicated and ranked.
4. **Evidence assessment.** A run is `completed`, `insufficient_evidence`, or `failed`.
   With `analyze=false`, the run stops here and returns sources only. With
   `analyze=true`, insufficient evidence goes directly to the Report Writer, which
   explains that no findings are presented.
5. **Analyst.** The local Qwen model receives only the selected sources, wrapped in
   `<source id="S1">` blocks with a warning that source text is untrusted. It returns JSON:
   an answer, findings with source IDs, a recommendation, and limitations. Pydantic validates
   the shape, and invented source IDs are removed.
6. **Verifier.** For every claim, code checks the source ID, the stored PDF chunk and page
   metadata, sentence completeness, numbers, wording overlap, subject relevance, and
   instruction-like text. If LLM review is enabled, the model reviews each (claim, source)
   pair. `verified` requires both the code checks and a supporting review. Rejected findings
   can trigger at most the configured number of revisions.
7. **Report Writer.** The report is rendered deterministically. Only `verified` claims are
   key findings. Unverified observations and rejected claims are labelled separately. The
   Analyst's recommendation is shown only when every requested criterion has verified
   evidence; otherwise a conservative statement names the missing evidence.
8. **Persistence and display.** The full `ResearchResult` is written to
   `data/runs/<run_id>.json`. The UI re-fetches it with `GET /research/{run_id}` and renders
   the Sources, Agent analysis, Verification, and Report tabs.

## PDF ingestion and retrieval

`app/rag/pdf.py` validates the file and extracts text page by page. `app/rag/chunking.py`
splits each page into overlapping character chunks, never across page boundaries, so every
chunk keeps a real `page_number`. `app/rag/embeddings.py` supports the semantic MiniLM model
and an offline lexical baseline for tests. `app/rag/retrieval.py` persists the document
library and rebuilds the FAISS index when its manifest no longer matches the configured
chunks or embedding model. Each run records the indexed chunk count, selected documents,
raw candidates, similarity scores, and threshold rejection reason. A below-threshold
candidate may only be used by the conservative fallback when it also shares distinctive
subject wording with the question; framing words such as "results" or "second" do not
qualify.

## Source provenance

Every source has a `source_type`:

| Type | Meaning |
| --- | --- |
| `document` | A chunk of an uploaded PDF, with filename, page, and chunk ID |
| `web` | A real page returned by the configured live provider |
| `demo` | A synthetic entry from the offline Demo corpus; never real evidence |

Demo mode is a search configuration, not proof that a run used synthetic sources. A run can
use uploaded PDFs only; the UI labels it **Local PDF Research** and does not warn about
synthetic sources. If a run actually contains `demo` sources, the UI and report say so.
Live web search requires `DEMO_MODE=false` and `TAVILY_API_KEY`; it has not been tested
against the real Tavily service in this release.

## Main modules

| Path | Responsibility |
| --- | --- |
| `app/agents/` | Planner, Analyst prompt/parser, Verifier, and Report Writer |
| `app/graph/` | LangGraph state, nodes, fan-out, and conditional routing |
| `app/rag/` | PDF extraction, chunking, embeddings, vector stores, retrieval, QA evaluation |
| `app/services/` | Research orchestration, document library construction, LLM loading, run storage |
| `app/schemas/` | Pydantic models shared by the API, workflow, UI, and stored runs |
| `app/api/` | FastAPI routers |
| `frontend/` | Streamlit presentation and a thin HTTP client |
| `tests/` | Offline logic, API, workflow, UI, notebook, and sample-data checks |

## Trust boundaries

- Uploaded PDF text and web snippets are untrusted data. They are delimited, closing tags
  inside them are neutralised, and prompts say not to follow their instructions.
- The model never supplies filenames or page numbers as authoritative metadata. Code resolves
  cited IDs to stored chunks.
- An LLM review is evidence for verification, not proof. The report says so explicitly.
- Secrets are read from environment variables or `.env`, never stored in run data or the UI.
