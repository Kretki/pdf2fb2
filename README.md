# pdf2fb2

PDF/DjVu → FB2 converter. Pages are segmented with **PP-DocLayoutV3** (PaddlePaddle) and every
text/title/table/footnote segment is read with **Qianfan-OCR** (PyTorch/transformers, 192 languages).
Bold, italic and strikeout words are detected with **TexTAR** (ICDAR 2025) and kept in the FB2.
A fine-tuned multilingual encoder (**token verifier**, XLM-RoBERTa) labels every OCR token as correct,
misread or not on the page, and the flagged lines are read again. Before the FB2 is written, a local
**Qwen3.5-9B** examines the words the verifier flags (or, without a verifier, every token of the text), respells
misread words and removes what does not belong to the book (stray symbols, page numbers, running heads taken for
headings).

PaddlePaddle GPU and PyTorch CUDA cannot share one environment, so the project is split into
seven uv projects, each with its own `.venv`:

| Directory | venv contents | Role |
|-----------|---------------|------|
| `.` (root) | PyMuPDF, Pillow, dev/security tools | Orchestrator: rendering, FB2 assembly, checks |
| `workers/layout_worker` | `paddlepaddle-gpu` (cu129), `paddleocr` | PP-DocLayoutV3 segmentation |
| `workers/ocr_worker` | `torch`/`torchvision` (cu130, needs NVIDIA driver ≥ 580), `transformers`, `accelerate` | Qianfan-OCR recognition |
| `workers/readermarks_worker` | `torch` (cu130), `timm`, `segmentation-models-pytorch`, OpenCV | handwritten reader-mark removal before OCR, then re-segmentation (see its README) |
| `workers/textar_worker` | `torch`/`torchvision` (cu130), `python-doctr` | word boxes (docTR DBNet) + TexTAR word attributes |
| `workers/proofread_worker` | `torch` (cu130), `transformers`, `bitsandbytes`, `flash-linear-attention` | Qwen3.5-9B (4-bit, ~8 GB VRAM) token probabilities and deletion tests |
| `workers/verifier_worker` | `torch` (cu130), `transformers` | token verifier: pipeline jobs and its fine-tuning (see its README) |
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
hf download Qwen/Qwen3.5-9B           --local-dir models/qwen3.5-9B    # proofreading language model
hf download FacebookAI/xlm-roberta-base --local-dir models/xlm-roberta-base  # only to train the token verifier
```

`models/token_verifier` is not downloaded: the admin fine-tunes it on the test corpus
(`workers/verifier_worker/README.md`). Until it exists, the verification step is skipped.

Environments:

```bash
scripts/sync_environments.sh
# equivalent to:
uv sync
uv sync --project workers/layout_worker
uv sync --project workers/ocr_worker
uv sync --project workers/readermarks_worker
uv sync --project workers/textar_worker
uv sync --project workers/proofread_worker
uv sync --project workers/verifier_worker
```

Optional: point to other interpreters with `PDF2FB2_LAYOUT_PYTHON` / `PDF2FB2_OCR_PYTHON` / `PDF2FB2_MARKS_PYTHON` / `PDF2FB2_STYLE_PYTHON` /
`PDF2FB2_PROOFREAD_PYTHON` / `PDF2FB2_VERIFY_PYTHON`.

Qianfan-OCR's `config.json` has no `transformers_version`, so transformers warns that the tokenizer may have
Mistral's broken split regex. It does not: the tokenizer is Qwen2's, and `fix_mistral_regex=True` would split
mixed-case words (`McDonald` → `Mc` `Donald`) unlike in training. The OCR worker passes `fix_mistral_regex=False`,
which only silences the warning; tokenization is unchanged.

## Pipeline

1. Render PDF pages (PyMuPDF).
2. Segment every page with PP-DocLayoutV3 (pictures are cut out as images, never OCR'd).
   Handwriting check: every printed page keeps its content inside one vertical frame (per page parity,
   from the content tops to the content bottoms of the pages, the 5% most extreme pages ignored), so text
   segments sticking out of it by more than 2% of the page height (e.g. a note written above the running
   head) are relabelled `handwritten` at once. Body text shares its column edges and headings sit at one horizontal position,
   so text segments whose left/right edge (headings: center) lies outside the 2-sigma interval of
   their group (same role, same page parity, at least 5 segments) are inspected by the reader-mark
   model alone, with the rest of the page blanked (`outliers/`). Outliers with marks are relabelled
   `handwritten` (`original_label` keeps the layout label) and are neither OCR'd nor put in the book.
3. Reader-mark detection (`workers/readermarks_worker`, checkpoints in `models/readermarks`): picture
   segments found by the layout model are blanked first (printed pictures are never taken for hand
   drawings), then handwritten notes, pen/pencil underlines and margin marks are detected and removed.
4. Pages with marks are re-segmented: the cleaned page goes through the layout model again (`rescan/`).
   Rescan segments lying mostly inside a first-pass `handwritten` segment are dropped (a hand-lettered note
   the cleaner did not erase is found again, but stays handwritten).
5. Final records = first-pass pictures + first-pass segments of unmarked pages + re-segmented
   cleaned pages. Picture recovery: OpenCV measures the median glyph height of every segment
   (`glyph_height` in `segments.json`); the median over the plain paragraphs is the usual font size
   (normal range: their 5th..95th percentile). Pages with body text outside 0.5x..2x of it, or with 8+
   letter-like components of 2.5x+ that height outside every segment and inside the printed frame, are
   segmented again with a low layout threshold (`picture_rescan/`, `--picture-threshold`, default 0.05).
   The most probable picture proposal covering the region (and more probable than the text it would
   replace) becomes an `image` segment that also takes in the text it covers and adjacent text whose
   glyphs are outside the normal range (`recovered_from` keeps their labels). Statistics and outcomes:
   `picture_rescan/font_statistics.json`.
   Residual text: with every segment blanked, letters of the usual size (0.5x..2x) still on the page are
   grouped into lines and blocks. Pieces in the band where the book prints its running heads or page
   numbers (median position of the `header` / `number` segments, per page parity) are added as such;
   blocks of 12+ glyphs right under a picture become captions (`figure_title`); other blocks inside the
   printed frame are missed paragraphs: their pages are segmented again (`residual_rescan/`,
   `--residual-threshold`, default 0.3) and each block gets the most probable text-like proposal covering
   it on free page area, or its own box (a title label is kept for one-line blocks only). Report:
   `residual_rescan/residual_text.json`. A `header` lying more than 3% of the page height outside the
   running-head band is a chapter title set lower on an opening page, not a running head: it becomes
   `paragraph_title` (running heads are not OCR'd and never reach the FB2). Only then Qianfan-OCR reads
   every text segment, once; spaced ellipses (`. . .`) become `…`.
   Layout review after OCR: short texts at the top/bottom of pages are read as page numbers (arabic,
   roman, letter-prefixed; OCR confusions such as `IIO`/`x11` tolerated) and every numbering run
   (kind, prefix, page index − value) seen on 3+ pages is a numbering of the book, so several numberings
   per book work; candidates on a run are relabelled `number` (`page_number_value`, `numbering`). Short
   (≤ 10th percentile of paragraph length), centred `text` on sparse pages (text coverage ≤ 30% of the
   median page) that stands apart (5+ glyph heights from other text) or is set larger than the usual font
   becomes `paragraph_title` (`original_label` keeps the layout label). So does the first text of a page
   when it is short, one line (≤ 3 glyph heights tall), centred and 3+ glyph heights above the next text
   (a chapter or preface title over its first paragraph).
6. Text styles (`workers/textar_worker`): docTR's DBNet finds the word boxes of every final page
   (the detector TexTAR was trained with), TexTAR classifies each word in context windows of 125
   neighbouring words (normal / bold / italic / bold+italic and none / underline / strikeout), and
   the words inside each text segment are stored in `segments.json` (`words`).
   Token verification (`workers/verifier_worker`, `models/token_verifier`, skipped while that folder has no
   `verifier_config.json`): every unit of the segment texts (whitespace tokens; Chinese/Japanese characters) is
   labelled `correct` / `misread` / `inserted` (not on the page), with the neighbouring segments of the page as
   context and page evidence: Qianfan-OCR's per-character confidence (`ocr_confidence` in `segments.json`), the
   word boxes (missing, shared, too narrow), optionally a second OCR read from a larger crop (`--second-read`,
   `second_read_text`) and word crops (`--verify-crops`). With `--judge`, units the verifier is unsure about are
   checked by Qwen3.5-9B as a vision-language model against the image of their line (`judge_lines/`).
   Inserted units with no detected word box under them are deleted. The text lines of misread units are cropped
   tightly and read again by Qianfan-OCR (`verify_rereads/`). The original and the new text are labelled again
   and the new text is kept only when its mean error probability is lower by 0.02 (`verify_report.json`; the old
   text stays in `pre_verify_text`).
7. The words are aligned with the OCR tokens of their segment (dynamic programming on character
   count vs box width, so merged or split boxes do not shift the styles) and the text gets
   `<b>`/`<i>`/`<s>` tags, which become `<strong>`/`<emphasis>`/`<strikethrough>` in the FB2.
   Rules: italic from score 0.4, bold from 0.55, strikeout from 0.7; bold+italic counts as italic
   on pages where italic words dominate 3:1; specks are ignored; a style covering a whole heading
   or caption is dropped; tables, the book title and numbers are never styled; FB2 has no
   underline, so underlines are not applied. The unstyled OCR text stays in `ocr_text`.
8. The book flow (paragraphs, headings, captions in reading order) is built.
9. Proofreading (`workers/proofread_worker`, `models/qwen3.5-9B`, 4-bit). Which words Qwen examines depends on
   the token verifier:
   * **flagged mode** (default once `models/token_verifier` exists): the verifier labels every word of the flow
     and Qwen tests only the words it flags as misread or not on the page. The whole text is not scored. A
     word's own probability comes from an extra `own` variant of its test. `proofread_report.json` lists
     probabilities only for the examined words, and the summary has `proofread_scope: flagged`, `flagged_words`
     and `flow_words`.
   * **full mode** (no verifier, `--no-verify`, or `--full-proofread`): every token of the flow text gets its
     probability from the preceding text (`proofread_report.json`: per word and per token), and every word whose
     least probable token is below e^-9 is tested.

   In both modes, suspicious spots are tested by scoring the following words with and without them: the
   selected words, every heading, every paragraph of at most 6 words, and every paragraph break after an
   unfinished sentence.
   * A word or short paragraph is deleted only when it is far better explained as OCR noise than as
     text (log-odds = continuation gain − ln 60 per character − its own log-probability + prior,
     prior −6 for letters, −1 for symbols; delete above 3). A misread word that still fits keeps a
     high probability for its length and stays unchanged; lone hyphens and note links are never cut.
   * A heading is deleted when the text reads on without it (continuation gain, +2 inside an
     unfinished sentence, +3 when the heading text repeats, +12 when it heads 10+ pages: a running
     head); when the sentence runs through it, the paragraphs on both sides are re-joined, which also
     removes the wrong section split.
   * A paragraph break is removed when the joined text is more probable by 2 nats or more.
   * A word split by a line-break hyphen that OCR kept with a space (`дискути- ровали`, `grand - father`,
     `the -tall`) is scored in three spellings (as read, `grand-father`, `grandfather`) with the same
     context and following words; a hyphen stuck to the next word (`the -tall`) may be a speck, so there
     separate words (`the tall`) are a fourth spelling. The most probable spelling is written, keeping style
     tags (report action `fix_hyphen`). Without proofreading (`--no-proofread`) such words stay as read.
   * Every tested word (flagged by the verifier, or below e^-9 in full mode) also gets up to 6 candidate
     spellings within two edits: OCR confusion pairs
     (`с`/`е`, `ы`/`ые`, `rn`/`m`, `cl`/`d`, long `s` read as `f`, Latin/Cyrillic look-alikes) and forms the book
     itself prints at least twice. Words the book prints three times or more are not respelled. Each spelling is
     scored with the same context and following words as the hyphen spellings. A candidate replaces the word
     when it wins by 2 nats after a penalty of 2 nats per unit of edit cost (a confusion costs 0.5, any other
     edit 1), keeping style tags (report action `respell_word`, summary `respelled_words`).
10. FB2 rendering and checks.

`--keep-marks` skips the handwriting check and steps 3–4, `--debug-marks` writes `marks/<page>/debug.png` overlays,
`--no-styles` skips steps 6–7 (`--textar-model-dir`, `--doctr-model-dir` change the weight folders),
`--no-proofread` skips step 9 (`--proofread-model-dir` changes the language model folder),
`--no-verify` skips the token verification of step 6 and makes step 9 score the whole text
(`--verify-model-dir` changes the verifier folder), `--full-proofread` keeps the verifier's re-reads but lets
Qwen score the whole text, `--second-read`, `--verify-crops` and `--judge` (`--judge-model-dir`) add verifier
evidence (see `workers/verifier_worker/README.md`), `--no-ocr-confidence` stops collecting OCR confidences,
`--no-picture-recovery` skips the picture recovery of step 5, `--no-residual-text` the residual text,
`--no-layout-review` the label statistics before and after OCR (running heads, page numbers, headings).

## Usage

```bash
uv run pdf2fb2 input/book.pdf --lang en       # results in output/book/ (or: uv run main.py input/book.pdf)
uv run pdf2fb2 book.djvu -o some/dir          # explicit result folder for one book
uv run pdf2fb2 --batch                        # every book under input/ -> output/<sub-folders>/<book name>/
uv run pdf2fb2 --batch --input-dir shelf --output-root converted
uv run pdf2fb2 --selftest                     # two-page generated test PDF, in output/selftest/

# faster OCR: serve Qianfan-OCR with vLLM (in its own env) and use the HTTP backend
vllm serve models/Qianfan-OCR --served-model-name Qianfan-OCR
uv run pdf2fb2 book.pdf --qianfan-url http://localhost:8000/v1 --workers 8
```

**Input formats.** `.pdf` is read directly. `.djvu`/`.djv` is first converted losslessly to
`<result folder>/source/<book name>.pdf` with DjVuLibre's `ddjvu` (install it with `sudo apt install djvulibre-bin`);
the converted file is reused while it is newer than the DjVu file.

**Batch mode.** `--batch` converts every `.pdf`/`.djvu`/`.djv` under `--input-dir` (default `input/`, sub-folders
included; hidden files and links leading outside it are skipped) into `--output-root` (default `output/`), one
folder per book named after the file and mirroring the sub-folders. Books of one folder with the same name
(`a.pdf`, `a.djvu`) get `a_pdf/`, `a_djvu/`. A failing book does not stop the batch; one summary line per book is
printed at the end and the exit code is 1 if any book failed or did not pass its checks. `-o`, `--title`,
`--author` and a book path name a single book and are rejected with `--batch`.

**Languages.** `--lang ru` declares a single-language book (as before). `--lang ja,en,la` declares several,
main language first: every paragraph, title, caption and note gets its language from its letters' scripts
(kana → Japanese, Han only → Chinese, Cyrillic, Greek, ...) and, for languages sharing a script, from their
frequent short words; paragraphs not in the main language get `xml:lang`, and a foreign run inside a paragraph
(two words or more, in a script the paragraph's language does not use: an English quotation in Japanese text)
is wrapped in `<style name="foreign" xml:lang="..">`. `--lang auto` picks among all built-in languages
(`ar de el en es fr he hi it ja ko la pl pt ru th uk zh`) and writes the language with the most text to `<lang>`.
Chinese/Japanese/Thai lines and page-broken paragraphs are joined without a space. Qianfan-OCR reads all of these
scripts with the same prompt, so OCR needs no language setting.

In a terminal the run shows one overall progress bar and, under it, the current step
(`extracting pages from pdf 3/20`, `segmenting page 7/40`, `parsing block 2/5 on page 3/10`).
When output is redirected, a line is printed every 10% instead.

Output directory: `pages/` (renders), `vis/` (segmentation drawings), `ocr_crops/`,
`layout_job.json`/`layout_result.json`, `ocr_job.json`/`ocr_results.jsonl`, `ocr_cache.json`
(re-runs skip already read segments; cleaned pages have their own keys), `marks/` + `marks_job.json`/`marks_result.json` (reader marks),
`rescan/` (cleaned pages with their own layout files), `style_job.json`/`style_result.json` (word styles),
`proofread_job.json`/`proofread_result.json`/`proofread_report.json` (token probabilities, edits and reasons),
`verify_job.json`/`verify_report.json`/`verify_rereads/` (token verifier labels, re-read lines), `segments.json` (final records, rescanned
pages flagged `"rescanned": true`) and `<name>.fb2`. Worker output (model loading, warnings,
tracebacks) goes to `layout_job.log` / `ocr_job.log` / `marks_job.log` / `style_job.log` / `proofread_job.log` /
`verify_job.log`.

Source layout (`src/pdf2fb2`): `interface/` (CLI, argument validation, book discovery and batch runs, pipeline
wiring, self-test PDF), `document_sources/` (input formats: PDF, DjVu → PDF), `text_languages/` (scripts,
language profiles and detection, multilingual FB2 markup), `pipeline/`
(orchestration, stage contracts, reader-mark rescan and page merging, checks), `model_stages/`
(layout, OCR, reader-mark and text-style worker stages), `text_styling/` (word order, token alignment,
style clean-up and markup), `proofreading/` (flow blocks, test selection, edit decisions, flow edits, report), `worker_runtime/` (worker venvs and launcher), `progress/`,
`rendering/` (PDF rendering, crops, drawings), `segments/` (segment geometry, label roles),
`fb2_output/` (FB2 assembly), `text_verification/` (token-verifier units, flags, re-read regions and the feedback
loop), `benchmark/` (test-corpus benchmark and verifier data export, run by `pdf2fb2-bench`).

## Time estimates and speed metrics

Before converting, `pdf2fb2` (and `pdf2fb2-bench run`, for the whole cycle of every sampled book) prints the
estimated duration per book and in total for the chosen stages, then waits: **Enter** starts, **Esc** aborts
(exit code 130). `--yes` prints the estimate and starts at once; with redirected input the run starts as well.

The estimate and the live progress bar come from `metrics/speed_metrics.json` (one per machine, git-ignored,
rewritten after every conversion): seconds per page of the page stages, seconds per word of OCR and verification,
seconds per inspected word of proofreading (kept separately for full and flagged mode), the startup time of
every stage (model loading), the words on a text page and the share of words the verifier flags as mistakes.
Every figure is a moving mean plus its mean deviation, and every page is taken for a text-only page, so the shown
time errs on the long side. Without the file, built-in figures are used. The bar is time-based
(elapsed / (elapsed + time left)), so a long proofreading step at the end keeps a realistic time left.

CPU work runs in parallel: page rendering in worker processes, glyph measurement, OCR crops and segment drawings
in threads. `PDF2FB2_CPU_WORKERS=N` limits the pool size (1 = sequential). The model workers still run one after
another, so each one gets the whole GPU.

## Benchmark and token verifier (admin)

`input/tests/catalog.csv` marks every corpus book `correct`, `mixed` or `wrong` (column `ocr_state`). Only
`correct` books are benchmarked or used for training. `uv run pdf2fb2-bench` converts sampled pages of a locked
book split with this pipeline and reports CER / WER / inserted / missed rates per language, before (`ocr`) and
after (`final`) proofreading. The hand-proofread `old_russian` books are the ground truth. The step-by-step
guide (download, data export, fine-tuning, threshold tuning, evaluation, before/after benchmark) is
`workers/verifier_worker/README.md`.

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
uv run --project workers/proofread_worker pip-audit
uv run --project workers/verifier_worker pip-audit
workers/readermarks_worker/.venv/bin/python -m pytest workers/readermarks_worker
(cd workers/textar_worker && uv run pytest)
(cd workers/proofread_worker && uv run pytest)
(cd workers/verifier_worker && uv run pytest)
uv run pytest                                    # root unit_tests/
shellcheck scripts/*.sh
trivy fs --scanners vuln,secret,misconfig --severity MEDIUM,HIGH,CRITICAL --exit-code 1 \
    --skip-dirs .venv,workers/layout_worker/.venv,workers/ocr_worker/.venv,models .
```
