# ResearchPilot

ResearchPilot is a local, evidence-grounded research assistant. It accepts a question,
plans the research, retrieves passages from uploaded PDFs and a configured search provider,
asks a local language model to analyse the evidence, checks every citation in code, and
produces a Markdown report. The report keeps verified findings separate from unverified
observations and model-written recommendations.

This repository is the final university-presentation release. It runs locally with
`Qwen/Qwen2.5-1.5B-Instruct`, does not require a paid API, and uses an offline synthetic corpus
for Demo mode.

## What it does

- Upload PDFs through a real FastAPI endpoint.
- Extract text page by page, chunk it with overlap, embed it, and retrieve relevant passages
  with FAISS.
- Plan deterministic queries for the question, criteria, and goals.
- Run web/search and document retrieval in parallel through LangGraph.
- Use the local Qwen model as an Analyst that returns structured findings with source IDs.
- Verify references, PDF page metadata, numbers, wording, subject relevance, and
  instruction-like source text in code; optionally ask the model to review each claim.
- Route rejected findings through a bounded revision, then stop safely.
- Render a deterministic report with verified findings, unverified observations, rejected
  claims, missing criteria, source details, and a downloadable Markdown file.
- Show the whole workflow in a Streamlit dashboard and an instructor walkthrough notebook.

## Evidence and verification model

A **valid citation** means the cited ID belongs to the run and, for a PDF, matches the stored
chunk and page. It is not a verified fact.

A claim is **verified** only when all of these hold:

1. Every cited ID is a real source of this run.
2. PDF citations match the stored document, filename, page, chunk text, and metadata.
3. URLs or page numbers written in the claim belong to a cited source.
4. The claim is a complete sentence, not a chunk fragment that stops mid-sentence.
5. Every number in the claim appears in the cited text.
6. Enough of the claim's content words appear in the cited text.
7. A subject from the question appears in both the claim and the cited text.
8. The cited source does not contain instruction-like injected text.
9. The LLM review judges the claim supported.

Word overlap alone never verifies a claim. Without LLM review, a claim can pass the code
checks and still remain `unverified`.

The report uses these outcomes:

| Status | Meaning |
| --- | --- |
| `verified` | Code checks passed and the review agreed |
| `unverified` | References are valid, but support could not be confirmed |
| `unsupported` | The cited text does not support the claim |
| `contradicted` | A source contradicts the claim, or sources conflict |
| `invalid_reference` | No valid citation, or PDF metadata does not match |

When a requested criterion has no verified finding, the Report Writer withholds the Analyst's
recommendation and replaces it with a deterministic statement naming the missing evidence. It
does not infer that the missing criterion is satisfactory.

## Technology

| Technology | Role |
| --- | --- |
| FastAPI + Pydantic | API, validation, and typed results |
| LangGraph | Parallel branches, conditional routing, revision loop |
| `pypdf` | Page-level PDF extraction |
| `sentence-transformers/all-MiniLM-L6-v2` | 384-dimensional embeddings |
| FAISS | Local cosine-similarity index; optional pgvector adapter |
| Qwen2.5-1.5B-Instruct + Transformers | Local Analyst and Verifier review |
| Streamlit | Browser interface over the API |
| pytest + Ruff | Offline tests and lint/format checks |

## Repository layout

```
app/
  agents/       planner, analyst, verifier, report writer
  api/          FastAPI routes
  graph/        LangGraph state, nodes, and routing
  rag/          PDF, chunking, embeddings, vector stores, retrieval, evaluation
  schemas/      Pydantic models shared by API, workflow, UI, and storage
  services/     orchestration, document library, LLM loading, run storage
  tools/        search providers (Tavily and offline Demo)
frontend/       Streamlit UI, HTTP client, pure display helpers
notebooks/      local walkthrough notebook and preserved Kaggle notebook
sample_data/    synthetic test PDF, questions, expected pages
docs/           architecture, project overview, demo script, recording notes, Kaggle checklist
scripts/        sample-data generator and end-to-end API demo
tests/          offline regression suite
data/           git-ignored runtime runs and document index
```

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --all-groups
```

This installs the API, tests, embeddings, FAISS, PyTorch/Transformers, Streamlit, and Jupyter.
The first real analysis downloads about 90 MB of embedding weights and about 3.1 GB of model
weights into the Hugging Face cache. Later runs reuse the cache.

If `uv` is installed but not on your `PATH`, either add its user binary directory or call it
explicitly:

```bash
# macOS example; adjust the Python version if yours differs
export PATH="$HOME/Library/Python/3.9/bin:$PATH"
# or
~/Library/Python/3.9/bin/uv sync --all-groups
```

`uv.lock` records the resolved environment. The three `requirements*.txt` files are generated
pinned exports for the base, ML, and UI environments; each file header records the `uv export`
command that regenerates it.

The project virtual environment can also run everything directly after a sync:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
.venv/bin/streamlit run frontend/streamlit_app.py
```

Copy `.env.example` to `.env` only if you want to change settings. Demo mode, local FAISS, and
the default local model work without a `.env` file.

## Run the application

Use two terminals from the repository root.

**Terminal 1 — backend:**

```bash
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://localhost:8000/docs> for the API documentation. The model is not loaded at
startup.

**Terminal 2 — frontend:**

```bash
.venv/bin/streamlit run frontend/streamlit_app.py
```

Open <http://localhost:8501>. The UI talks to `http://localhost:8000` by default; set
`RESEARCHPILOT_API_URL` to use another backend.

The API sets `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` before importing FAISS or PyTorch.
That prevents the macOS `SIGSEGV` (exit code `139`) that can occur when their OpenMP worker
pools collide during local model loading. Other entry points such as notebooks or custom
scripts that use both libraries should set `OMP_NUM_THREADS=1` in their environment too.

## Use the sample PDF

The sample document is:

```
sample_data/pdfs/researchpilot_test_report.pdf
```

It is a **five-page synthetic educational test report** about a fictional college battery
pilot. It is not a real-world study. The companion questions are in:

```
sample_data/pdfs/researchpilot_test_questions.md
```

The questions include an answerable summary, a specific reported result, safety arrangements,
cost/efficiency, and unanswerable questions. Expected pages were checked against the generated
PDF by `scripts/make_test_report.py`.

In the UI:

1. Select the PDF under **Documents**.
2. Press **Upload 1 selected file** and wait for the green confirmation. The UI shows the
   backend's real `document_id`, page count, and chunk count.
3. Confirm the document is checked under **Registered documents**.
4. Enter the question, add criteria such as `efficiency`, `safety`, and `cost` one per line,
   and press **Analyze**.
5. Use the example-question button to load the synthetic battery decision question.

The UI never treats a browser-selected file as uploaded. A document ID is always the value
returned by `POST /documents`; it is never hard-coded.

### API alternative

```bash
curl -s -F "file=@sample_data/pdfs/researchpilot_test_report.pdf" \
  http://localhost:8000/documents
```

Capture the returned `document_id` and use it in the next request:

```bash
UPLOAD=$(curl -s -F "file=@sample_data/pdfs/researchpilot_test_report.pdf" \
  http://localhost:8000/documents)
DOCUMENT_ID=$(printf '%s' "$UPLOAD" | .venv/bin/python -c \
  'import json, sys; print(json.load(sys.stdin)["document_id"])')
printf 'document_id=%s\n' "$DOCUMENT_ID"

RESEARCH=$(curl -s http://localhost:8000/research \
  -H 'content-type: application/json' \
  -d "$(cat <<JSON
{
  "question": "Based on the pilot results, should the college reuse the lithium iron phosphate battery at its second campus?",
  "criteria": ["efficiency", "safety", "cost"],
  "constraints": ["limited capital budget"],
  "document_ids": ["$DOCUMENT_ID"],
  "analyze": true
}
JSON
)")
RUN_ID=$(printf '%s' "$RESEARCH" | .venv/bin/python -c \
  'import json, sys; print(json.load(sys.stdin)["run_id"])')
printf 'run_id=%s\n' "$RUN_ID"

curl -s "http://localhost:8000/research/$RUN_ID"
```

## End-to-end demonstration script

The script starts its own API with a temporary data directory, uploads the sample PDF, runs an
analysis-enabled question using the returned document ID, fetches the stored run, prints the
verification results and report, then asks a question the PDF cannot answer:

```bash
.venv/bin/python scripts/demo_api.py --start
```

To use an already-running backend instead:

```bash
.venv/bin/python scripts/demo_api.py --base-url http://127.0.0.1:8000
```

The first analysis can take minutes while the model loads. The script prints genuine API
errors if the model cannot load; it does not simulate results.

## Walkthrough notebook

Open `notebooks/researchpilot_project_walkthrough.ipynb` in VS Code, select the project
`.venv` kernel, and choose **Run All**. It covers the problem, architecture, device detection,
PDF extraction, chunking/embeddings/FAISS, Qwen loading, citation-checked question answering,
the real FastAPI workflow in-process, a verified report, an insufficient-evidence case, the
test suite, limitations, and likely questions.

The notebook finds the repository root from the kernel's working directory; it does not depend
on a machine-specific absolute path. It uses the local sample PDF and loads the real model when
the model section runs. The preserved Kaggle notebook is
`notebooks/researchpilot_kaggle_demo.ipynb`; Kaggle execution is not required for this release.

## Tests, lint, and validation

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python - <<'PY'
import nbformat
for path in (
    "notebooks/researchpilot_project_walkthrough.ipynb",
    "notebooks/researchpilot_kaggle_demo.ipynb",
):
    nbformat.validate(nbformat.read(path, as_version=4))
    print(path, "valid")
PY
```

Current result: **218 tests passed**, **all Ruff checks passed**, and **69 files already
formatted**. Tests run offline with scripted LLM replies, hashing embeddings, and an offline
search provider; they verify logic and contracts, not real model quality.

## API summary

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service status, search mode, and provider name |
| `POST` | `/documents` | Upload a PDF; identical bytes return the existing document |
| `GET` | `/documents` | List registered documents |
| `GET` | `/documents/{document_id}` | Document metadata and warnings |
| `POST` | `/documents/search` | Similarity search over stored chunks |
| `POST` | `/research` | Run the workflow; `analyze=true` enables the agents |
| `GET` | `/research/{run_id}` | Fetch a persisted run and report |

Upload errors are explicit: `415` for a non-PDF, `413` for an oversized file, `422` for a
password-protected or textless PDF, and `503` when the optional document stack is unavailable.

## Search provenance and Demo mode

`DEMO_MODE=true` (the default) uses an offline synthetic corpus at `demo.example`. These
entries are sample data, never live web research. The configured search provider and the
sources actually used in a run are separate:

- A PDF-only run is labelled **Local PDF Research** in the UI and has no synthetic-source
  warning.
- If a run actually contains Demo sources, the UI and report label them as synthetic and not
  real evidence.
- Live web search requires `DEMO_MODE=false` and `TAVILY_API_KEY`. The live Tavily adapter is
  implemented but has not been verified against the real service in this release.

## Configuration

The defaults are in `app/config.py`; `.env.example` documents the common settings.

| Variable | Default | Notes |
| --- | --- | --- |
| `DEMO_MODE` | `true` | Offline synthetic corpus |
| `TAVILY_API_KEY` | empty | Required only for live search |
| `EMBEDDING_PROVIDER` | `sentence-transformers` | `hashing` is an offline lexical baseline |
| `DOCUMENT_TOP_K` | `3` | Chunks retrieved per planned query |
| `DOCUMENT_MIN_SCORE` | `0.25` | Cosine-similarity floor |
| `DOCUMENT_FALLBACK_MIN_SCORE` | `0.18` | Lower floor used only when a candidate also shares distinctive subject wording; set equal to the main floor to disable |
| `LLM_MODEL_ID` | `Qwen/Qwen2.5-1.5B-Instruct` | Local model |
| `LLM_DEVICE` | `auto` | `cuda`, `mps`, or `cpu` |
| `ANALYSIS_MAX_SOURCES` | `6` | Sources given to the agents |
| `ANALYSIS_MAX_REVISIONS` | `1` | Bounded retry rounds |
| `VERIFIER_LLM_REVIEW` | `true` | Without it, claims cannot reach `verified` |

Runtime runs and uploaded documents are written under `data/`, which is git-ignored. Secrets
belong in `.env` or the environment, never in code, notebooks, or run data.

## Known limitations

- **Verified is not proof.** Lexical checks and a 1.5B reviewer can miss a changed meaning or a
  subject swap. Read the cited page before relying on a finding.
- **Conservative recall.** Correct paraphrases often remain `unverified`.
- **Small model.** Qwen2.5-1.5B can ignore JSON formatting or over-cite. Invalid output is
  rejected rather than shown as fact.
- **Retrieval scope.** Only the configured top-k chunks and top sources reach the agents. A
  relevant page outside that window is invisible.
- **No OCR.** Scanned pages are reported and skipped.
- **Untested integrations.** Live Tavily search, real Supabase pgvector, and Kaggle execution
  are not verified in this release.
- **One model at a time.** Analysis requests are serialized; concurrent requests queue.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — Mermaid data-flow diagram and module responsibilities.
- [Project overview](docs/PROJECT_OVERVIEW.md) — purpose, design decisions, limitations.
- [Final demo script](docs/FINAL_DEMO_SCRIPT.md) — five-minute presentation and instructor Q&A.
- [Notebook recording notes](docs/NOTEBOOK_RECORDING_NOTES.md) — speaking cues for the local walkthrough recording.
- [Kaggle checklist](docs/KAGGLE_CHECKLIST.md) — optional, preserved notebook setup.
