# Kaggle run checklist

The notebook `notebooks/researchpilot_kaggle_demo.ipynb` has been executed end to end
**locally** (Apple M4 Pro, MPS float16, via `nbclient`). It has **not yet been run on
Kaggle.** Work through this list once on Kaggle and note the results before presenting them.

## Prepare (on your computer)

- [ ] From the repository root, package the shared code and sample data:
      `zip -r researchpilot-code.zip app notebooks/sample_data -x '*/__pycache__/*'`
- [ ] Check that the zip contains `app/rag/qa.py` and
      `notebooks/sample_data/researchpilot_sample_report.pdf`: `unzip -l researchpilot-code.zip`

## Set up on Kaggle

- [ ] *Create → New Dataset*, upload `researchpilot-code.zip`, name it `researchpilot-code`.
- [ ] *Create → New Notebook → File → Import Notebook*, upload the `.ipynb`.
- [ ] *Add Input → Your Datasets → researchpilot-code*.
- [ ] Optional: *Add Input → Upload* your own PDF (otherwise the bundled sample is used).
- [ ] *Settings → Internet → On* (needed for `pip install pypdf faiss-cpu` and for downloading
      about 90 MB (MiniLM) and about 3.1 GB (Qwen2.5-1.5B)). Kaggle may require phone
      verification before Internet can be enabled.
- [ ] *Settings → Accelerator → GPU T4 x2* (or leave on CPU; slower but supported).
- [ ] Optional: *Add-ons → Secrets → `HF_TOKEN`*. Never paste tokens into cells.

## Run and record

- [ ] *Run All*.
- [ ] Section 2 prints `ResearchPilot code: /kaggle/input/...`, not an error.
- [ ] Section 2 prints the device: expect `cuda (float16)` on a T4, or `cpu (float32)`. If a
      GPU is attached but CPU is chosen, the printed reason should explain why (e.g. the
      PyTorch build does not support that GPU).
- [ ] Section 3 shows the PDF you expected and the `document_id` returned by your own upload
      (the sample PDF has 4 pages; do not hard-code a document ID).
- [ ] Section 6 loads `Qwen/Qwen2.5-1.5B-Instruct` and prints the device and dtype.
- [ ] Section 7 shows an answer with page references and no `ERROR`.
- [ ] Section 8 shows the president question as `insufficient_evidence`, and the evaluation
      table and summary. Record the printed numbers; they may differ from the local run
      (float16 on CUDA vs MPS).
- [ ] The final check cell prints `Citation-validity and retrieval checks passed`.
- [ ] Note the session's hardware, total runtime, and any warnings.

## Before sharing

- [ ] No cell output contains a token or password.
- [ ] Update the README and `docs/FINAL_DEMO_SCRIPT.md` with "verified on Kaggle (date,
      hardware)" only after this checklist passes.

## If something fails

| Symptom | Likely cause / fix |
| --- | --- |
| `ResearchPilot code not found` | Dataset not attached, or the zip lacks `app/` at its top level |
| `pip install` fails | Internet is off |
| `LLMLoadError` | Internet off, or wrong `LLM_MODEL_ID`; the message lists the checks |
| Out of memory | Restart the kernel; use a GPU session; or set `LLM_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct` |
| `Python 3.11 or newer` assertion | The Kaggle image is older than expected; choose the latest environment |
