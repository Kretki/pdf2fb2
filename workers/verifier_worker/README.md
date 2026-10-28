# verifier worker — token verifier (admin guide: data, fine-tuning, testing)

The token verifier is a multilingual encoder (default **XLM-RoBERTa base**, 278M parameters) fine-tuned for
**token classification**. It does not generate text. It reads the OCR text of a segment and gives every unit
(a whitespace token; for Chinese/Japanese, one character) three probabilities:

| label | meaning | what the pipeline does |
|---|---|---|
| `correct` | read as printed | nothing |
| `misread` | wrongly parsed (`rnodern`, `врсмя`, `fame` for `same`) | the unit's text line is cropped and read again by Qianfan-OCR |
| `inserted` | not on the page: text the OCR model wrote from context (a run-on continuation, a repeated line), or a reader's note | deleted when no word box lies under it; otherwise its line is read again |

After the re-reads the segment is labelled again, and the new text is kept only when its mean error probability
is lower (`verify_report.json` lists every decision). The pipeline runs this between word-box detection and word
styles. It is **skipped automatically until `models/token_verifier/config.json` exists**, which is the output of
the training below.

The verifier also decides **where Qwen3.5-9B looks**. Once `models/token_verifier` exists, Qwen no longer scores
every token of the book. Before proofreading, the verifier labels every word of the book flow, and Qwen runs
only:

* deletion and respelling tests of the **flagged words** (`misread` / `inserted`);
* the structural tests that the verifier cannot judge: headings, paragraphs of at most 6 words, paragraph
  breaks after an unfinished sentence, and words split by a line-break hyphen.

Each of these is a short context-plus-continuation test, so most of the book never goes through Qwen.
`proofread_report.json` / the conversion summary show `proofread_scope: flagged` with `flagged_words` out of
`flow_words`. Without a verifier, with `--no-verify`, or with `--full-proofread`, Qwen scores the whole text as
before (`proofread_scope: full`).

The model is trained on **every language of the catalog** (english, spanish, french, german, russian,
old_russian, japanese, and any language added to `catalog.csv` later), not only on the `old_russian` books.
Hand-proofread `old_russian` is only the most trustworthy *test* reference.

Nothing is downloaded or trained automatically. The admin runs every step below.

## 1. Setup (once)

```bash
uv sync --project workers/verifier_worker            # torch cu130 (RTX 50-series) + transformers
hf download FacebookAI/xlm-roberta-base --local-dir models/xlm-roberta-base   # ~1.1 GB, base model for training
```

The pipeline only needs `models/token_verifier/` (the fine-tuned result). The base model is used for training
only. A larger alternative is `FacebookAI/xlm-roberta-large` (560M). Set `paths.base_model` in a config file
(section 6), and use `batch_size` 8 with `gradient_accumulation` 4.

VRAM: base model training needs about 4 GB at batch size 16 in bf16. Inference needs under 1 GB. pdf2fb2 runs
one worker at a time, so the verifier never shares the GPU with Qianfan-OCR or Qwen3.5.

## 2. Corpus and splits

The corpus is `input/tests/` (see `input/tests/README.txt`). `catalog.csv` has the column **`ocr_state`**:

| ocr_state | meaning | used |
|---|---|---|
| `correct` | reference text is right; `old_russian` ones are hand-proofread Wikisource transcriptions (true ground truth), the others readable machine OCR | yes |
| `mixed` | readable, but with a systematic error: long s read as f, garbled scripts, garbage pages, two orthographies in one file | no |
| `wrong` | unusable reference (Fraktur garbage, Greek read as Latin, wrong language, manuscript) | no |

Only `correct` books are read by any command, from **all languages** of the catalog. When a book's state is
re-checked, edit its `ocr_state` cell; the commands pick up the change on the next run. A new language needs no
code change: add its books to `input/tests/<language>/` and to `catalog.csv` (with `language`, `language_code`
for `pdf2fb2 --lang`, the scan path, a reference path and `ocr_state`), then run `pdf2fb2-bench splits`. The split is by **book** (pages of one book never land in two
splits) and per language. About 15 % of each language goes to test, 15 % to valid, and the rest to train.
Half of the `old_russian` gold books go to test.

```bash
uv run pdf2fb2-bench splits          # creates benchmark_data/splits.json once and prints the counts
```

`benchmark_data/splits.json` is **locked**. Keep it, so that every later run, export and evaluation uses the same
books. New `correct` books are added to it automatically, and existing assignments never change. Delete the file
only to start a completely new experiment.

## 3. Baseline benchmark (before training)

```bash
uv run pdf2fb2-bench run --split test --name baseline --pages 10 -- --no-verify
```

This converts 10 sampled body pages of every test book with the normal pipeline. Pages are seeded by book key,
so every run reads the same pages. Options after `--` are passed to `pdf2fb2`.

Before converting, every `pdf2fb2-bench run` prints the estimated time of the whole cycle (every book, per book
and in total, for the stages the options after `--` select) and waits: **Enter** starts, **Esc** aborts. Put
`--yes` before `--` to start without waiting (for scripted runs, e.g. the step 4 commands run one after another).
The estimate comes from `metrics/speed_metrics.json`, which every converted book of the run updates (see the main
README, "Time estimates and speed metrics"). The run writes
`benchmark_runs/baseline/report.md`, with CER / WER / inserted-word / missed-word rates per language at two
levels:

* `ocr`: `segments.json` after OCR (and after the verifier feedback, when it ran), before proofreading;
* `final`: the FB2 text after Qwen3.5 proofreading (respellings, hyphen fixes, deletions).

How scoring works:

* The output page is located inside its reference by word trigrams. This skips running heads, page numbers
  and, for whole-book transcripts, the rest of the book.
* Both sides are normalised in the same way: line-break hyphens joined, Markdown and wiki markup removed, quotes
  and dashes unified, long s read as s.
* Chinese and Japanese are scored per character.

`unlocated` counts pages whose text matches nothing in the reference. A high count means bad OCR or a wrong
reference.

Useful options:

* `--languages old_russian`: gold pages only. This is the trustworthy number. The other languages compare with
  library OCR, which has its own errors, so use them as a relative regression measure.
* `--reuse`: score existing conversions again without converting.
* `uv run pdf2fb2-bench score --name baseline`: re-score after a scoring change.

A run of 10 pages × ~20 test books takes roughly 1–2 hours on an RTX 5070 Ti, most of it Qianfan-OCR and Qwen3.5;
the printed estimate is deliberately on the long side.

## 4. Training data

Two kinds of labelled data, both JSON lines in `benchmark_data/`:

```bash
# (a) synthetic: reference text of correct books, corrupted like OCR corrupts text
uv run pdf2fb2-bench export-references --split train --chunks-per-language 600
uv run pdf2fb2-bench export-references --split valid --chunks-per-language 150
workers/verifier_worker/.venv/bin/python -m token_verifier synth --split train
workers/verifier_worker/.venv/bin/python -m token_verifier synth --split valid

# (b) real: pipeline output of all languages aligned with the reference
uv run pdf2fb2-bench run --split train --name train_pages --pages 20 -- --no-verify --no-proofread
uv run pdf2fb2-bench export-aligned --name train_pages        # -> aligned_train.jsonl
uv run pdf2fb2-bench run --split valid --name valid_pages --pages 20 -- --no-verify --no-proofread
uv run pdf2fb2-bench export-aligned --name valid_pages        # -> aligned_valid.jsonl
uv run pdf2fb2-bench export-aligned --name baseline           # -> aligned_test.jsonl (step 3 run)
```

Synthetic corruptions are:

* OCR confusion pairs (`с/е`, `ы/ые`, `rn/m`, `cl/d`, long `s`→`f`; the shared table is in
  `workers/worker_protocol/src/worker_protocol/ocr_confusions.py`);
* character drops, duplications and swaps;
* inserted spans: repetition loops, a passage of the same book from elsewhere, and short reader notes.

Both kinds cover every catalog language and are balanced between languages:

* **Synthetic:** `--chunks-per-language` is one budget per language, shared by its books. English (many books)
  and French (few books) contribute about the same amount.
* **Real:** examples are oversampled per language towards the language with the most examples, by at most
  `training.language_balance_cap` (8).

Real examples teach the model Qianfan's actual error distribution. Their labels come from the alignment with the
reference, so with machine-OCR references a few labels are wrong. Synthetic data, the `old_russian` gold books
and `real_repeat` together balance this.

To restrict an experiment to some languages, pass `--languages english,russian` to `run`. The reference export
always takes all languages of the split.

## 5. Fine-tune, tune thresholds, evaluate

```bash
cd workers/verifier_worker
.venv/bin/python -m token_verifier train           # synthetic_train + aligned_train (x3); best epoch on valid
.venv/bin/python -m token_verifier tune            # thresholds on aligned_valid (train already does this)
.venv/bin/python -m token_verifier eval --split valid
.venv/bin/python -m token_verifier eval --split test   # final report only; never tune on it
```

`train` writes `models/token_verifier/`: weights, tokenizer, `thresholds.json` (decision thresholds of `misread`
and `inserted`, chosen for the best F1 on validation) and `window_settings.json`. Epoch progress is printed. Only
the epoch with the best validation F1 is kept.

`eval` writes `benchmark_data/verifier_reports/verifier_<split>.{md,json}`, with precision / recall / F1 of
`misread` and `inserted` over all languages and per language. On the test split, the `old_russian` rows are the
ground-truth number.

## 6. Settings

Defaults are in `src/token_verifier/default_settings.json`. To change them, pass a JSON file with only the keys
to change, e.g.:

```json
{"training": {"epoch_count": 6, "learning_rate": 2e-05}, "paths": {"base_model": "models/xlm-roberta-large"}}
```

```bash
.venv/bin/python -m token_verifier train --config my_settings.json
```

| key | default | effect |
|---|---|---|
| `synthesis.misread_rate` | 0.06 | share of units corrupted per synthetic example |
| `synthesis.insert_rate` | 0.3 | share of synthetic examples with an inserted span |
| `training.window_units` / `max_tokens` | 96 / 256 | units per window / sub-word limit |
| `training.class_weights` | 1, 3, 3 | loss weights of correct, misread, inserted (errors are rare) |
| `training.real_repeat` | 3 | repetitions of real examples per epoch |
| `training.language_balance_cap` | 8 | largest oversampling of a language's real examples (1 = no balancing) |
| `training.epoch_count` | 4 | epochs |

## 7. Benchmark after training

```bash
uv run pdf2fb2-bench run --split test --name verified --pages 10      # verifier on (models/token_verifier exists)
```

Compare `benchmark_runs/verified/report.md` with `benchmark_runs/baseline/report.md`:

* The `ocr` level shows what the feedback re-reads changed.
* The `final` level shows everything, including the gated Qwen proofreading.

To compare the gated proofreading with the old whole-text proofreading on the same pages, run the test split
once more with `-- --full-proofread`. Each book's `conversion/proofread_report.json` and the printed statistics
show `proofread_scope` and how many words Qwen examined.

Inserted-word rate should fall (hallucinated text removed). CER should fall wherever lines were re-read
successfully. For each book, `benchmark_runs/<name>/<book>/conversion/verify_report.json` shows which units were
flagged and which re-reads were accepted.

To switch the verifier off for a conversion, use `--no-verify` (Qwen then scores the whole text). To keep
the verifier's re-reads but let Qwen score everything, use `--full-proofread`. To use another model
directory, use `--verify-model-dir`.

## Layout

| Path | Content |
|---|---|
| `src/token_verifier/datasets/` | JSONL examples, noise synthesis, unit → sub-word windows |
| `src/token_verifier/networks/model_loading.py` | local-only loading of base / fine-tuned model, thresholds |
| `src/token_verifier/training/` | epoch loop (AdamW, warm-up, bf16), training service |
| `src/token_verifier/evaluation/` | metrics, threshold tuning, reports |
| `src/token_verifier/inference/` | batched prediction, pipeline job (`VerifyJob`, `worker_protocol/verify_job_schema.py`) |
| `src/pdf2fb2/benchmark/` (orchestrator) | catalog, splits, sampling, references, scoring, exports |
| `src/pdf2fb2/text_verification/` (orchestrator) | units, flags, regions, re-read loop, report |

Tests: `cd workers/verifier_worker && .venv/bin/python -m pytest` (tiny local tokenizer and random tiny encoder;
nothing is downloaded).
