# ResearchPilot — 5-minute presentation script

This script uses observed results from a real local run on this machine. Timings and model
outputs vary by machine and run; do not present the rehearsal numbers as guarantees.

## Before the session (about 10 minutes, once)

From the repository root:

```bash
uv sync --all-groups
```

If `uv` is not on your `PATH`, use `.venv/bin/...` for the commands below or call the installed
`uv` executable directly. The first real analysis downloads about 90 MB of embedding weights
and about 3.1 GB of model weights; later runs reuse the cache.

Start the backend and frontend in two terminals:

```bash
# terminal 1
OMP_NUM_THREADS=1 .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000

# terminal 2
OMP_NUM_THREADS=1 .venv/bin/streamlit run frontend/streamlit_app.py
```

Open <http://localhost:8501> and <http://localhost:8000/docs>. Warm the model with:

```bash
.venv/bin/python scripts/demo_api.py --base-url http://127.0.0.1:8000
```

The sample PDF is synthetic educational test data:
`sample_data/pdfs/researchpilot_test_report.pdf`. The questions and expected pages are in
`sample_data/pdfs/researchpilot_test_questions.md`.

The full local walkthrough notebook
`notebooks/researchpilot_project_walkthrough.ipynb` (VS Code, kernel `.venv`, Run All) is a
complete alternative or backup.

---

## 0:00–0:30 · The problem

> Language models write fluent answers but cannot show where a fact came from, and they
> sometimes invent sources. For research and decisions, every claim needs to be tied to a real
> page, and the system has to say “I don’t know” when the evidence is missing. ResearchPilot
> plans the research, retrieves evidence from uploaded PDFs and a configured search provider,
> has a local LLM analyse it, checks every citation in code, and writes a report in which only
> verified claims count as findings.

## 0:30–1:15 · Architecture and stack

Draw the pipeline on the slide or point to `docs/ARCHITECTURE.md`:

```
POST /research → plan → search ×N (web) + document_search ×N (PDFs)
                       → collect_sources → assess_evidence
                       → analyst → verifier → report_writer → END
```

- **FastAPI + Pydantic** for the API and typed results.
- **LangGraph** for parallel retrieval, conditional routing, and the bounded revision loop.
- **pypdf + MiniLM + FAISS** for page-level PDF retrieval.
- **Qwen2.5-1.5B-Instruct** running locally for the Analyst and claim review.
- **Streamlit** as a thin client over the real API.
- **pytest + Ruff** for the offline checks.

Say clearly: “Verified means the source exists and the checks passed. It is not a guarantee of
truth, and the report labels unverified observations separately.”

## 1:15–2:45 · Live demonstration

1. Show the header: **Backend connected** and **Search configured: Demo corpus (offline)**.
   Explain that this is the configured search provider, not a claim that this run will use
   synthetic sources.
2. Under **Documents**, upload `sample_data/pdfs/researchpilot_test_report.pdf`.
3. Point to the confirmation: the real `document_id`, **5 pages**, and **16 chunks**. Explain
   that a file selected in the browser is not counted until the backend accepts it.
4. Keep the document checked under **Registered documents (1)**.
5. Click **Load example question for the sample PDF**.
6. Open **Analysis settings** and show the criteria: `efficiency`, `safety`, `cost`, and the
   constraint `limited capital budget`.
7. Click **Analyze**.
8. When results appear, point to the **Local PDF Research** badge. This run used four PDF
   passages and no synthetic sources, so there is no synthetic-source warning.
9. Show the metrics and the three verified findings: thermal stability, the two minor alarms,
   and the container/fire-suppression arrangements.
10. Open **Comparisons and trade-offs** and then the **Recommendation** section.

The important sentence to read aloud:

> **Insufficient verified evidence for a final decision.** The verified findings do not cover
> efficiency and cost. Further evidence is needed for efficiency and cost before the decision
> can be made. The absence of verified evidence must not be read as satisfactory performance
> on these criteria.

Point out that this recommendation was generated deterministically from the verification
results. The Analyst’s own recommendation was withheld because it discussed safety and cost
without explicitly covering the missing efficiency evidence.

## 2:45–3:35 · How verification works

Open the **Verification** tab and click a claim card.

- The claim cites `S1: researchpilot_test_report.pdf, p. 4`.
- Code checks that the source ID exists in this run and that the stored PDF chunk, filename,
  page, and text match.
- It checks numbers, wording overlap, subject relevance, instruction-like text, and that the
  claim is a complete sentence.
- It shows the LLM review status and reason.
- A claim is `verified` only when the code checks and the review agree. A valid citation by
  itself stays `unverified`.

This is the central engineering point: the model proposes findings, but it does not decide
what the report calls verified.

## 3:35–4:20 · When the document cannot answer

Run a second question:

```text
How many electric vehicles does the campus charge each day?
```

Use the same document and no criteria. The PDF has no such information. The run returns
`analysis.status = insufficient_evidence` and states that no findings are presented. Nothing
is invented to fill the gap.

If the demo is short on time, run this through the API instead:

```bash
.venv/bin/python scripts/demo_api.py --base-url http://127.0.0.1:8000
```

The script also prints the insufficient-evidence run and the genuine API errors if a model
cannot load.

## 4:20–5:00 · Tests, limitations, and close

- **218 offline tests passed** in about 4.4 s on the final validation run. They replace the
  LLM, embeddings, and search with offline stand-ins: no API keys and no model download.
- Ruff reports **all checks passed** and **69 files already formatted**.
- The real-model rehearsal produced **3 verified findings, 0 unverified, 0 rejected**, in
  **11.2 s**; the insufficient-evidence run took **1.2 s**. These are observed results, not
  benchmarks.
- Limitations: verification is conservative but heuristic; the 1.5B reviewer can miss meaning
  changes; retrieval only sees the configured top-k chunks; scanned PDFs are not OCR’d; live
  Tavily and real Supabase pgvector are implemented but not verified against the services.

Close with:

> The system is useful because it refuses to turn a plausible model sentence into a finding
> without a traceable source. The report shows what is verified, what is merely observed, and
> what evidence is still missing.

---

## Likely questions

**“Why not just ask ChatGPT?”**
A general chat model can read a PDF, but it does not normally provide an auditable per-claim
verification pipeline or a deterministic report that withholds unsupported recommendations.
ResearchPilot makes the evidence path and the failure states visible.

**“Is the local model good enough?”**
It is a 1.5B model, so it is deliberately constrained by retrieval, schema validation, and
verification. It can produce bad or incomplete claims; the system labels or rejects them
rather than trusting them. A larger model can be used through `LLM_MODEL_ID`.

**“What does verified mean?”**
It means the citation and metadata exist, the numbers and wording are present, the subject is
relevant, the source has no instruction-like injection, and the LLM review agreed. It is not
a formal proof; the report says to read the cited page.

**“Why does the recommendation say evidence is missing when the PDF has efficiency data?”**
Retrieval did not surface a chunk that supported an efficiency claim in that run, so the
criterion was not covered. The report is intentionally conservative: an unsurfaced fact is not
a verified finding. A larger top-k or a different query would be needed to retrieve it.

**“Why is Demo mode on?”**
It provides an offline, reproducible corpus so the application and tests do not need a paid
search API. Demo sources are labelled synthetic. A PDF-only run is labelled **Local PDF
Research** and has no synthetic-source warning.

**“Can it handle many PDFs or concurrent users?”**
The library supports up to 20 document IDs per request, but analysis uses one local model at a
time, so concurrent analysis requests queue. The document store is local FAISS; pgvector is an
optional adapter.

**“What is next?”**
Larger-model evaluation, sentence-aware chunking, a broader independently written evaluation
set, OCR, and verification of the live search and pgvector adapters.

## If something fails

- **Backend offline:** check terminal 1, `http://localhost:8000/health`, and the API docs.
- **Model still loading:** the first analysis can take minutes; the UI waits for the finished
  run rather than showing fake progress.
- **macOS exit code 139:** restart both processes with `OMP_NUM_THREADS=1` as shown above.
- **No verified findings:** show the Verification tab and explain that this is a deliberate
  outcome, not a UI error. Then use the walked-through insufficient-evidence example.
- **Need a backup:** run the local notebook end to end; it imports the same modules and uses
  the same sample PDF.
