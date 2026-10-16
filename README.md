# pdf2fb2

PDF → FB2 converter. Pages are segmented with **PP-DocLayoutV3** (PaddlePaddle) and every
text/title/table/footnote segment is read with **Qianfan-OCR** (PyTorch/transformers, 192 languages).

PaddlePaddle GPU and PyTorch CUDA cannot share one environment, so the project is split into
four uv projects, each with its own `.venv`:

| Directory | venv contents | Role |
|-----------|---------------|------|
| `.` (root) | PyMuPDF, Pillow, dev/security tools | Orchestrator: rendering, FB2 assembly, checks |
| `workers/layout_worker` | `paddlepaddle-gpu` (cu129), `paddleocr` | PP-DocLayoutV3 segmentation |
| `workers/ocr_worker` | `torch`/`torchvision` (cu130, needs NVIDIA driver ≥ 580), `transformers`, `accelerate` | Qianfan-OCR recognition |
| `workers/readermarks_worker` | `torch` (cu130), `timm`, `segmentation-models-pytorch`, OpenCV | handwritten reader-mark removal after the first OCR pass, then rescan (see its README) |
| `workers/worker_protocol` | stdlib only (installed into all three) | JSON job-file contract |

The orchestrator starts each worker as a child process with that worker's own interpreter
(`workers/<name>/.venv/bin/python -m <name> --job-file …`), sequentially, so each model gets
the whole GPU and frees it when its process exits. No shell is involved.

## Setup

Model weights (already expected in `./models`):

```bash
hf download PaddlePaddle/PP-DocLayoutV3 --local-dir models/PP-DocLayoutV3
hf download baidu/Qianfan-OCR          --local-dir models/Qianfan-OCR
```

Environments:

```bash
scripts/sync_environments.sh
# equivalent to:
uv sync
uv sync --project workers/layout_worker
uv sync --project workers/ocr_worker
uv sync --project workers/readermarks_worker
```

Optional: point to other interpreters with `PDF2FB2_LAYOUT_PYTHON` / `PDF2FB2_OCR_PYTHON` / `PDF2FB2_MARKS_PYTHON`.

## Pipeline

1. Render PDF pages (PyMuPDF).
2. Segment every page with PP-DocLayoutV3 (pictures are cut out as images, never OCR'd).
3. Reader-mark detection (`workers/readermarks_worker`, checkpoints in `models/readermarks`): picture
   segments found by the layout model are blanked first (printed pictures are never taken for hand
   drawings), then handwritten notes, pen/pencil underlines and margin marks are detected and removed.
4. Pages with marks are re-segmented: the cleaned page goes through the layout model again (`rescan/`).
5. Final records = first-pass pictures + first-pass segments of unmarked pages + re-segmented
   cleaned pages; only then Qianfan-OCR reads every text segment, once → FB2 assembly and checks.

`--keep-marks` skips steps 3–4, `--debug-marks` writes `marks/<page>/debug.png` overlays.

## Usage

```bash
uv run pdf2fb2 book.pdf -o out --lang en      # or: uv run main.py book.pdf -o out
uv run pdf2fb2 --selftest -o out              # two-page generated test PDF

# faster OCR: serve Qianfan-OCR with vLLM (in its own env) and use the HTTP backend
vllm serve models/Qianfan-OCR --served-model-name Qianfan-OCR
uv run pdf2fb2 book.pdf --qianfan-url http://localhost:8000/v1 --workers 8
```

In a terminal the run shows one overall progress bar and, under it, the current step
(`extracting pages from pdf 3/20`, `segmenting page 7/40`, `parsing block 2/5 on page 3/10`).
When output is redirected, a line is printed every 10% instead.

Output directory: `pages/` (renders), `vis/` (segmentation drawings), `ocr_crops/`,
`layout_job.json`/`layout_result.json`, `ocr_job.json`/`ocr_results.jsonl`, `ocr_cache.json`
(re-runs skip already read segments; cleaned pages have their own keys), `marks/` + `marks_job.json`/`marks_result.json` (reader marks),
`rescan/` (cleaned pages with their own layout files), `segments.json` (final records, rescanned
pages flagged `"rescanned": true`) and `<name>.fb2`. Worker output (model loading, warnings,
tracebacks) goes to `layout_job.log` / `ocr_job.log` / `marks_job.log`.

Source layout (`src/pdf2fb2`): `interface/` (CLI, argument validation, self-test PDF), `pipeline/`
(orchestration, stage contracts, reader-mark rescan and page merging, checks), `model_stages/`
(layout, OCR, reader-mark worker stages), `worker_runtime/` (worker venvs and launcher), `progress/`,
`rendering/` (PDF rendering, crops, drawings), `segments/` (segment geometry, label roles),
`fb2_output/` (FB2 assembly).

## Checks (AGENTS.md §6)

```bash
uv run ruff check src workers main.py
uv run bandit -c pyproject.toml -r src workers main.py -ll
uv run semgrep scan --config p/default --config p/python --config p/secrets \
    --exclude node_modules --exclude .venv --exclude models --error
uv run pip-audit
uv run --project workers/layout_worker pip-audit
uv run --project workers/ocr_worker pip-audit   # torch/paddle wheels come from their own indexes and are skipped
uv run --project workers/readermarks_worker pip-audit
workers/readermarks_worker/.venv/bin/python -m pytest workers/readermarks_worker
shellcheck scripts/*.sh
trivy fs --scanners vuln,secret,misconfig --severity MEDIUM,HIGH,CRITICAL --exit-code 1 \
    --skip-dirs .venv,workers/layout_worker/.venv,workers/ocr_worker/.venv,models .
```
