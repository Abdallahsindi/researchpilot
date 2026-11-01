"""End-to-end API demo for presentations.

    uv run python scripts/demo_api.py --start          # starts its own server (temp data dir)
    uv run python scripts/demo_api.py --base-url http://127.0.0.1:8000   # use a running server

Steps: start the API (no model loaded), upload the sample PDF, run an analysis-enabled
research request with the returned document id, fetch the stored run, print the report and
verification results, then ask a question the PDF cannot answer.

The first analysis request loads Qwen2.5-1.5B-Instruct (about 3.1 GB download the first time;
needs `uv sync --group rag --group llm`). If the model cannot load, the genuine error from the
API is shown; nothing is simulated.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
# The synthetic test report and its question file (see sample_data/pdfs/).
CASES = json.loads((ROOT / "sample_data/pdfs/researchpilot_test_report_cases.json").read_text())
SAMPLE_PDF = ROOT / "sample_data" / "pdfs" / CASES["pdf"]
ANSWERABLE = CASES["ui_demo_request"]
UNANSWERABLE = next(c["question"] for c in CASES["cases"] if not c["answerable"])
# Strings the model loader prints; used to show the server did not load the LLM at startup.
MODEL_LOAD_MARKERS = ("Loading weights", "Loading Qwen", "Model loaded")


def heading(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def start_server(port: int, log_path: Path, data_dir: Path) -> subprocess.Popen:
    env = {**os.environ, "DATA_DIR": str(data_dir)}
    if sys.platform == "darwin":
        # FAISS followed by PyTorch can trip a libomp crash on macOS; keep the demo stable.
        env.setdefault("OMP_NUM_THREADS", "1")
    log = open(log_path, "w")  # noqa: SIM115 - kept open for the server's lifetime
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)],
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def wait_for_health(client: httpx.Client, timeout: float = 60) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        try:
            return client.get("/health").raise_for_status().json()
        except httpx.HTTPError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.25)


def research(client: httpx.Client, body: dict) -> dict:
    started = time.perf_counter()
    response = client.post("/research", json=body)
    if response.status_code != 200:
        print(f"POST /research failed: HTTP {response.status_code}: {response.text}")
        response.raise_for_status()
    run = response.json()
    print(f"POST /research -> 200 in {time.perf_counter() - started:.1f}s, run_id={run['run_id']}")
    return run


def show_analysis(run: dict, sources: dict) -> None:
    analysis = run["analysis"]
    print(f"research status: {run['status']}    analysis status: {analysis['status']}")
    print(f"model: {analysis['model_id']}    LLM calls: {run['metrics']['llm_calls']}")
    for error in analysis["errors"]:
        print(f"analysis error: {error}")
    for attempt in analysis["attempts"]:
        print(
            f"Analyst attempt {attempt['attempt']}: parse_error={attempt['parse_error']}, "
            f"invented source ids removed={attempt['invented_source_ids']}"
        )
    verification = analysis["verification"]
    if verification:
        print("\nClaim verification (valid citation != verified fact):")
        for claim in verification["claims"]:
            refs = []
            for sid in claim["cited_source_ids"]:
                s = sources[sid]
                where = f"{s['filename']} p.{s['page_number']}" if s["page_number"] else s["url"]
                refs.append(f"{sid} [{s['source_type']}] {where}")
            print(f"  [{claim['status']}] {claim['claim']}")
            print(f"      cites: {'; '.join(refs) or 'nothing valid'}")
            print(f"      why:   {' '.join(claim['explanations'])}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--start", action="store_true", help="start a server for the demo")
    parser.add_argument("--port", type=int, default=8800)
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="researchpilot-demo-"))
    server = None
    base_url = args.base_url
    if args.start:
        base_url = f"http://127.0.0.1:{args.port}"
        log_path = workdir / "server.log"
        heading(f"1. Starting the API on {base_url} (data dir {workdir / 'data'})")
        started = time.perf_counter()
        server = start_server(args.port, log_path, workdir / "data")

    # Analysis can take minutes on first use (model download), so allow a long read timeout.
    client = httpx.Client(base_url=base_url, timeout=httpx.Timeout(10, read=1800))
    try:
        health = wait_for_health(client)
        if server:
            loaded = any(m in log_path.read_text() for m in MODEL_LOAD_MARKERS)
            print(f"healthy after {time.perf_counter() - started:.1f}s: {health}")
            print(f"LLM loaded during startup: {'YES' if loaded else 'no'}")
        else:
            heading(f"1. Using running API at {base_url}")
            print(health)

        heading("2. Upload the sample PDF (POST /documents)")
        with open(SAMPLE_PDF, "rb") as handle:
            upload = client.post(
                "/documents", files={"file": (SAMPLE_PDF.name, handle, "application/pdf")}
            )
        upload.raise_for_status()
        document = upload.json()
        print(
            f"HTTP {upload.status_code}: document_id={document['document_id']}, "
            f"{document['page_count']} pages, {document['chunk_count']} chunks"
        )

        heading("3. Analysis-enabled research request (POST /research, analyze=true)")
        body = {**ANSWERABLE, "document_ids": [document["document_id"]], "analyze": True}
        print(f"request: {body}")
        run = research(client, body)
        sources = {s["id"]: s for s in run["sources"]}

        heading("4. Retrieve the persisted run (GET /research/{run_id})")
        stored = client.get(f"/research/{run['run_id']}").raise_for_status().json()
        print(f"stored run identical to the response: {stored == run}")

        heading("5. Sources, verification, and limitations")
        for s in run["sources"]:
            ref = f"{s['filename']}, page {s['page_number']}" if s["filename"] else s["url"]
            print(f"  {s['id']} [{s['source_type']}] {ref}")
        show_analysis(run, sources)
        output = next(
            (a["output"] for a in reversed(run["analysis"]["attempts"]) if a["output"]), None
        )
        if output and output["limitations"]:
            print("\nLimitations stated by the Analyst (model-written, not verified):")
            for item in output["limitations"]:
                print(f"  - {item}")
        if run["report"]:
            report_path = workdir / "report.md"
            report_path.write_text(run["report"])
            heading(f"6. Markdown report (also saved to {report_path})")
            print(run["report"])

        heading("7. A question the PDF cannot answer")
        run = research(
            client,
            {"question": UNANSWERABLE, "document_ids": [document["document_id"]], "analyze": True},
        )
        show_analysis(run, {s["id"]: s for s in run["sources"]})
        summary = run["report"].split("## Executive summary")[1].split("## ")[0].strip()
        print(f"\nExecutive summary from the report:\n{summary}")
        return 0
    except httpx.HTTPError as exc:
        print(f"\nDemo stopped: {type(exc).__name__}: {exc}")
        if server:
            print(f"Server log: {log_path}")
        return 1
    finally:
        client.close()
        if server:
            server.terminate()
            server.wait(timeout=30)


if __name__ == "__main__":
    sys.exit(main())
