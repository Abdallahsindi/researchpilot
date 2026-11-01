# Notebook recording notes

These are speaking cues for a 5–7 minute screen recording of
`notebooks/researchpilot_project_walkthrough.ipynb`. They follow the notebook's actual section
order. This is a guide, not a script to memorise: explain each point in your own words and skip
anything that makes the recording run long.

The 5-minute application demonstration in `docs/FINAL_DEMO_SCRIPT.md` is a separate recording.
This file is only for the local walkthrough notebook.

## Before recording

- Open the repository in VS Code and select the `.venv` kernel.
- Run all cells once before recording. With the models already cached, the full run takes
  roughly 30–60 seconds on this Mac. The last local run used Apple MPS with float16.
- Keep the executed outputs on screen while recording. The committed notebook must stay
  output-free, so do not save the executed version back over it. Use a scratch copy or clear all
  outputs before committing.
- If a cell fails, check the kernel first. The notebook prints a warning when it is not running
  with the project's `.venv`.

The numbers below are from the last local run. Talk about what the output means rather than
promising exact timings, because model output and timings can vary.

---

## 0:00–0:30 · Opening

**Show:** the title and the opening paragraphs.

**Explain:** ResearchPilot is not just a chatbot over a PDF. The problem is traceability: a
fluent answer is not useful for research unless I can see which page supports it, and the system
has to admit when the document does not contain the answer. The notebook runs the project's real
code, not a separate mock demo.

**Transition:** "The answer has to be checked, so the first thing to look at is where each check
happens in the architecture."

## 0:30–1:00 · Section 1: The problem

**Show:** the paragraph under "The problem."

**Explain:** retrieval limits what the model can see; the LangGraph stages separate planning,
analysis, verification, and reporting; the code checks resolve citations back to stored text.
The important distinction is between "the model said it" and "the document supports it."

**Transition:** "That separation is easier to see in the graph itself."

## 1:00–1:45 · Section 2: Architecture

**Show:** the ASCII graph.

**Explain:**

- The request enters through `POST /research`; web and PDF searches run in parallel, then
  `collect_sources` and `assess_evidence` decide whether analysis should continue.
- The Planner is deliberately ordinary Python. It creates one query for the core question, one
  per criterion, and one per goal. It is not an LLM-powered planner.
- `all-MiniLM-L6-v2` is the embedding model. It only finds relevant chunks.
  `Qwen2.5-1.5B-Instruct` is the generative model that writes findings and performs the optional
  language review.
- The Report Writer is also plain code. It formats results that have already been checked; it
  does not make another model call.
- Conditional edges handle the important cases: sources only, insufficient evidence, normal
  analysis, or one bounded revision.

**Do not claim:** that Tavily or Supabase pgvector were tested against live services. The
architecture paragraph and the limitations section say they are implemented but not verified live.

**Transition:** "Before running any of that, I need the right Python environment and device."

## 1:45–2:10 · Section 3: Environment and device

**Show:** the project root, `.venv` path, and device output.

**Explain:** the notebook discovers the project root instead of hard-coding a path, and it warns
if the selected kernel is not the project's `.venv`. The device selector does not load the
model; it only chooses where inference will run. On this machine it chose Apple MPS with
float16. The first run needs the model download; later runs use the cache.

**Transition:** "With the environment ready, I can work through the document side."

## 2:10–3:00 · Sections 4 and 5: PDF, chunks, embeddings, and FAISS

**Show:** the five-page PDF table, then the 16-chunk retrieval table.

**Explain:**

- The sample PDF is synthetic. The facts are invented, so a correct answer should come from the
  document rather than Qwen's memory.
- `load_pdf()` extracts text page by page, so every later chunk keeps its page number.
- Chunking is about 500 characters with 100 characters of overlap. The chunk ID encodes the page
  (`…:p2:c0`), which is how the citation can be resolved later.
- MiniLM creates a 384-dimensional unit vector per chunk. FAISS compares the question vector with
  those chunks and returns the closest ones.
- In this run the top match was page 2 at similarity 0.547. The important point is that retrieval
  is still model-free at this stage: Qwen has not written anything yet.

**What this does not prove:** a good retrieval score does not prove the answer will be correct.
It only means the passage is a plausible match.

**Transition:** "Retrieval finds the evidence. Now I load the model that writes the answer."

## 3:00–3:25 · Section 6: Running Qwen locally

**Show:** the loaded model, device, and tokenizer output.

**Explain:** `HuggingFaceLLM` is the project's only model-loading code. Construction is lazy; the
expensive work happens in `load()` or on the first `generate()` call. Qwen is used without
fine-tuning, with greedy decoding, so it does not sample randomly. There is no fallback to fake
output if loading fails. It took a couple of seconds in the final run.

**Transition:** "Now I can put retrieval and Qwen together in the standalone RAG path."

## 3:25–4:15 · Section 7: Standalone RAG and checked citations

**Show:** the answerable efficiency answer, the unanswerable manufacturer answer, then the
hand-written `E1`/`E9` check.

**Explain:**

- This is the standalone path: `answer_question()` uses the retriever and Qwen directly. It is
  not the LangGraph workflow yet.
- The prompt labels up to three retrieved passages as `E1`, `E2`, and `E3`. The model must return
  an `ANSWER:` line and an `EVIDENCE:` line.
- Code maps each `E` label back to the chunk that was actually in the prompt. The filename and
  page come from that stored chunk, not from anything Qwen wrote.
- The efficiency answer resolved to page 3 and reported 90.5%. The manufacturer question
  abstained because the PDF does not contain that information.
- The hand-written reply is deliberately not a model result. It cites `E1`, which is real, and
  `E9`, which does not exist. The output shows that `E1` is kept and `E9` is discarded.

**What this does not prove:** the citation checks are lexical heuristics. They catch invented
IDs and weakly supported wording, but they do not prove that the answer is true. That is why the
full workflow adds more checks.

**Transition:** "The standalone function answers one question. The real application coordinates
several stages, so now I switch to the actual FastAPI app."

## 4:15–5:35 · Section 8: The real FastAPI and LangGraph workflow

**Show, in this order:** health response and graph edges; upload response; research status and
source table; Analyst JSON and verification table; stored report and recommendation.

**Explain:**

- `TestClient` sends requests through the real ASGI application. I do not start a separate
  server, but the routes, schemas, document library, and graph are the real ones.
- The run uses a temporary data directory and Demo mode, so it starts clean and needs no API key.
  The already loaded Qwen instance is passed through `llm_factory`, so the app does not load a
  second copy. It does load its own embedding model, as the server would.
- The graph output shows the possible paths: skip analysis, abstain, go to the Analyst, revise,
  or go to the Report Writer.
- The Analyst receives source blocks, treats their text as untrusted, and returns structured
  JSON. Pydantic validates it, and invented source IDs are removed.
- The Verifier runs code checks first: valid references, stored document metadata, fabricated
  page or URL locators, complete sentences, numbers, word overlap, subject checks, injection
  checks, and relevance. Then the model reviews claims against their sources. `verified`
  requires both the code checks and a supporting review.
- In this run the request completed in under 6 seconds with 4 LLM calls and 3 verified findings,
  with 0 unverified and 0 rejected claims. Point out that the report recommendation is withheld
  because no verified finding covers efficiency and cost. That is deterministic report-writing
  code, not another model call.
- The API stores the run, and fetching it again returns the same object. That is the persistence
  check.

**If it comes up:** the Analyst may attach odd criterion labels to its findings. The report does
not trust the label by itself; it only counts a criterion when verified evidence actually
supports it. That is why the recommendation can be withheld even though the model wrote a
confident recommendation.

**What this does not prove:** this is one synthetic PDF and one request. Live Tavily search,
Supabase pgvector, and Kaggle execution are outside what this notebook verifies.

**Transition:** "A good research system also has to stop when the evidence is missing."

## 5:35–6:00 · Section 9: Insufficient evidence

**Show:** `analysis: insufficient_evidence`, the executive summary, and the absence of findings.

**Explain:** the question asks how many electric vehicles the campus charges each day. The PDF
does not discuss electric vehicles. The system should not turn a real but unrelated sentence
into an answer. This is the relevance failure that motivated the check: a citation can be
correct and still not answer the question. The report explains what is missing instead of
inventing a number.

**What this does not prove:** one abstention example does not prove the workflow catches every
irrelevant claim.

**Transition:** "The last thing I want to show is how I check the surrounding logic."

## 6:00–6:25 · Section 10: Automated tests

**Show:** the pytest and Ruff lines at the bottom of the notebook.

**Explain:** the last local run was 218 tests passed in about 4.4 seconds, and Ruff passed. The
tests use scripted model replies, not Qwen, so they check routing, schemas, citation and injection
checks, revision limits, API behaviour, persistence, and the UI. They do not measure model
quality. Sections 6–9 are the real-model sanity check.

**Transition:** "That leaves the boundaries of what the project actually establishes."

## 6:25–6:50 · Section 11: Limitations

**Show:** the limitations bullets.

**Explain:** "verified" is not a guarantee of truth; the checks are heuristics. The system is
conservative by design, so a correct paraphrase can remain unverified. The 1.5B model is small
and can produce inconsistent JSON. Retrieval only sees the configured top-k chunks. Live Tavily,
Supabase pgvector, and Kaggle were not tested in this release, and there is no OCR for scanned
PDFs. The evaluation set is small and was used during prompt tuning, so it is not a benchmark.

**Transition:** "Those are also the areas I expect to be asked about."

## 6:50–7:00 · Section 12 and closing

**Show:** the questions section, then the final paragraph.

**Explain:** be ready for the likely questions:

- Why a small model? It runs locally in about 3.1 GB, and the verification layer is there because
  the model is not trusted on its own.
- Why not rely on word overlap? "LFP was not chosen for safety" shares almost every word with the
  source. Overlap alone never makes a claim verified.
- What happens if the model cannot load? The run reports a failed analysis with the real error;
  a request without `analyze` never loads the model.
- Why LangGraph? The state and routing are explicit, searches run in parallel, and each stage
  can be tested separately.
- Why no LLM in the Report Writer? It only formats checked results, so plain code is safer and
  cannot add new claims.

**Close:** "The point is not that Qwen can produce a fluent paragraph. It is that the system can
show which pages the answer came from, separate verified findings from unverified ones, and
refuse to answer when the evidence is not there."
