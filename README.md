# pdf2fb2

PDF → FB2 converter. Pages are segmented with **PP-DocLayoutV3** (PaddlePaddle) and every
text/title/table/footnote segment is read with **Qianfan-OCR** (PyTorch/transformers, 192 languages).
Bold, italic and strikeout words are detected with **TexTAR** (ICDAR 2025) and kept in the FB2.

PaddlePaddle GPU and PyTorch CUDA cannot share one environment, so the project is split into
five uv projects, each with its own `.venv`:

| Directory | venv contents | Role |
|-----------|---------------|------|
| `.` (root) | PyMuPDF, Pillow, dev/security tools | Orchestrator: rendering, FB2 assembly, checks |
| `workers/layout_worker` | `paddlepaddle-gpu` (cu129), `paddleocr` | PP-DocLayoutV3 segmentation |
| `workers/ocr_worker` | `torch`/`torchvision` (cu130, needs NVIDIA driver ≥ 580), `transformers`, `accelerate` | Qianfan-OCR recognition |
| `workers/readermarks_worker` | `torch` (cu130), `timm`, `segmentation-models-pytorch`, OpenCV | handwritten reader-mark removal before OCR, then re-segmentation (see its README) |
| `workers/textar_worker` | `torch`/`torchvision` (cu130), `python-doctr` | word boxes (docTR DBNet) + TexTAR word attributes |
| `workers/worker_protocol` | stdlib only (installed into every worker) | JSON job-file contract |

The orchestrator starts each worker as a child process with that worker's own interpreter
(`workers/<name>/.venv/bin/python -m <name> --job-file …`), sequentially, so each model gets
the whole GPU and frees it when its process exits. No shell is involved.

## Setup

Model weights (already expected in `./models`):

```bash
hf download PaddlePaddle/PP-DocLayoutV3 --local-dir models/PP-DocLayoutV3
hf download baidu/Qianfan-OCR          --local-dir models/Qianfan-OCR
hf download Tex-TAR/model_tex-tar      --local-dir models/textar        # TexTAR-trained.pt
mkdir -p models/doctr && curl -L -o models/doctr/db_resnet50-79bd7d70.pt \
    'https://doctr-static.mindee.com/models?id=v0.7.0/db_resnet50-79bd7d70.pt&src=0'  # docTR word detector
```

Environments:

```bash
scripts/sync_environments.sh
# equivalent to:
uv sync
uv sync --project workers/layout_worker
uv sync --project workers/ocr_worker
uv sync --project workers/readermarks_worker
uv sync --project workers/textar_worker
```

Optional: point to other interpreters with `PDF2FB2_LAYOUT_PYTHON` / `PDF2FB2_OCR_PYTHON` / `PDF2FB2_MARKS_PYTHON` / `PDF2FB2_STYLE_PYTHON`.

## Pipeline

1. Render PDF pages (PyMuPDF).
2. Segment every page with PP-DocLayoutV3 (pictures are cut out as images, never OCR'd).
3. Reader-mark detection (`workers/readermarks_worker`, checkpoints in `models/readermarks`): picture
   segments found by the layout model are blanked first (printed pictures are never taken for hand
   drawings), then handwritten notes, pen/pencil underlines and margin marks are detected and removed.
4. Pages with marks are re-segmented: the cleaned page goes through the layout model again (`rescan/`).
5. Final records = first-pass pictures + first-pass segments of unmarked pages + re-segmented
   cleaned pages; only then Qianfan-OCR reads every text segment, once.
6. Text styles (`workers/textar_worker`): docTR's DBNet finds the word boxes of every final page
   (the detector TexTAR was trained with), TexTAR classifies each word in context windows of 125
   neighbouring words (normal / bold / italic / bold+italic and none / underline / strikeout), and
   the words inside each text segment are stored in `segments.json` (`words`).
7. The words are aligned with the OCR tokens of their segment (dynamic programming on character
   count vs box width, so merged or split boxes do not shift the styles) and the text gets
   `<b>`/`<i>`/`<s>` tags, which become `<strong>`/`<emphasis>`/`<strikethrough>` in the FB2.
   Rules: italic from score 0.4, bold from 0.55, strikeout from 0.7; bold+italic counts as italic
   on pages where italic words dominate 3:1; specks are ignored; a style covering a whole heading
   or caption is dropped; tables, the book title and numbers are never styled; FB2 has no
   underline, so underlines are not applied. The unstyled OCR text stays in `ocr_text`.
8. FB2 assembly and checks.

`--keep-marks` skips steps 3–4, `--debug-marks` writes `marks/<page>/debug.png` overlays,
`--no-styles` skips steps 6–7 (`--textar-model-dir`, `--doctr-model-dir` change the weight folders).

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
`rescan/` (cleaned pages with their own layout files), `style_job.json`/`style_result.json` (word styles), `segments.json` (final records, rescanned
pages flagged `"rescanned": true`) and `<name>.fb2`. Worker output (model loading, warnings,
tracebacks) goes to `layout_job.log` / `ocr_job.log` / `marks_job.log` / `style_job.log`.

Source layout (`src/pdf2fb2`): `interface/` (CLI, argument validation, self-test PDF), `pipeline/`
(orchestration, stage contracts, reader-mark rescan and page merging, checks), `model_stages/`
(layout, OCR, reader-mark and text-style worker stages), `text_styling/` (word order, token alignment,
style clean-up and markup), `worker_runtime/` (worker venvs and launcher), `progress/`,
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
uv run --project workers/textar_worker pip-audit
workers/readermarks_worker/.venv/bin/python -m pytest workers/readermarks_worker
(cd workers/textar_worker && uv run pytest)
uv run pytest                                    # root unit_tests/
shellcheck scripts/*.sh
trivy fs --scanners vuln,secret,misconfig --severity MEDIUM,HIGH,CRITICAL --exit-code 1 \
    --skip-dirs .venv,workers/layout_worker/.venv,workers/ocr_worker/.venv,models .
```
