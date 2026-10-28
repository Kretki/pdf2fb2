# verifier worker — token verifier (admin guide: data, fine-tuning, testing)

The token verifier is a multilingual encoder (default **XLM-RoBERTa base**, 278M parameters) fine-tuned for
**token classification**. It does not generate text. It reads the OCR text of a segment and gives every unit
(a whitespace token; for Chinese/Japanese, one character) three probabilities:

| label | meaning | what the pipeline does |
|---|---|---|
| `correct` | read as printed | nothing |
| `misread` | wrongly parsed (`rnodern`, `врсмя`, `fame` for `same`) | the unit's text line is cropped and read again by Qianfan-OCR |
| `inserted` | not on the page: text the OCR model wrote from context (a run-on continuation, a repeated line), or a reader's note | deleted when no word box lies under it; otherwise its line is read again |

After the re-reads the original and the re-read text are labelled again side by side, and the new text is kept
only when its mean error probability is lower (`verify_report.json` lists every decision). The pipeline runs this
between word-box detection and word styles. It is **skipped automatically until
`models/token_verifier/verifier_config.json` (or `ensemble.json`) exists**, which is the output of the training
below. A verifier trained with an older version must be trained again.

Besides the text, every unit can carry **page evidence**, all of it language-independent:

| evidence | where it comes from | pipeline switch |
|---|---|---|
| neighbouring segments of the page | segment order | always |
| OCR confidence (lowest / mean token log-probability) | Qianfan-OCR scores, `segment["ocr_confidence"]` | on; `--no-ocr-confidence` turns it off |
| word box missing / shared / too narrow | docTR word boxes (style stage) | on whenever word styles run |
| second OCR read disagrees | a second read from a slightly larger crop, `segment["second_read_text"]` | `--second-read` (doubles OCR time) |
| language-model plausibility | masked-LM head of `models/xlm-roberta-base` | on when the verifier was trained with `paths.language_model` |
| word crop embedding (image-and-text) | `models/dinov2-small` | `--verify-crops`, for a verifier trained with `paths.visual_model` |

Units the verifier is unsure about (error probability 0.15–0.6) can be judged by **Qwen3.5-9B as a
vision-language model** against the image of their printed line (`--judge`); its answer is averaged with the
verifier's probability before deletions and re-reads are planned.

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
scripts/sync_environments.sh                                   # all venvs (verifier: torch cu130 + transformers)
hf download FacebookAI/xlm-roberta-base --local-dir models/xlm-roberta-base   # ~1.1 GB, if not present yet
scripts/download_verifier_assets.sh fonts                      # rendering fonts (SIL OFL) -> models/fonts/
scripts/download_verifier_assets.sh mdeberta byt5              # optional alternative backbones (safetensors)
```

Models used by the verifier and where they come from:

| model | role | needed for |
|---|---|---|
| `models/xlm-roberta-base` | default backbone; masked-LM plausibility scorer | training (backbone and `paths.language_model`) |
| `models/dinov2-small` | word-crop embeddings | training/inference with `paths.visual_model` (already present) |
| `models/qwen3.5-9B` | vision-language judge (`--judge`) | pipeline only (already present) |
| `models/mdeberta-v3-base` | alternative sub-word backbone | optional (`paths.base_model`) |
| `models/byt5-small` | byte-level backbone: sees every letter, any script | optional (`paths.base_model`, `backbone_kind: byte`) |
| `models/fonts/*` | rendering fonts for rendered training pages | `render-references` (falls back to system fonts) |

The pipeline only needs `models/token_verifier/` (the fine-tuned result) plus the evidence models it was trained
with. The base model is used for training only.

VRAM (RTX 5070 Ti, 16 GB): xlm-roberta-base training with plausibility features, R-Drop and EMA needs about 8 GB
at batch size 16 in bf16; add about 1 GB for crop embeddings. Inference needs about 2 GB. The judge loads
Qwen3.5-9B in 4 bit (about 7 GB). pdf2fb2 runs one worker at a time.

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

All examples are JSON lines in `benchmark_data/`. Besides `words` and `labels`, real and rendered examples carry
the page evidence the pipeline sends at prediction time (`context_before`, `context_after`, `features`, and with
`--crops` the word `crops`), so training and use see the same inputs.

### 4a. Real pipeline output (aligned with the references)

```bash
# convert sampled pages; OCR confidences are collected by default, --second-read adds the second read
uv run pdf2fb2-bench run --yes --split train --name train_pages --pages 60 -- --no-verify --no-proofread --second-read
uv run pdf2fb2-bench run --yes --split valid --name valid_pages --pages 20 -- --no-verify --no-proofread --second-read
uv run pdf2fb2-bench run --yes --split test  --name baseline    --pages 10 -- --no-verify --second-read
uv run pdf2fb2-bench export-aligned --name train_pages --crops   # -> aligned_train.jsonl (+ verify_crops/)
uv run pdf2fb2-bench export-aligned --name valid_pages --crops   # -> aligned_valid.jsonl
uv run pdf2fb2-bench export-aligned --name baseline --crops      # -> aligned_test.jsonl
```

More pages give more real examples, which is the most valuable data: `--pages 60` on the train split triples the
earlier amount. Runs made before this version have no OCR confidences and no second read; convert them again
(without `--reuse`) to get these features. Word boxes and crops come from any run with word styles.

The labels come from aligning the output with the reference. Most references are machine OCR, so the export
aligns every segment separately, drops pages that barely match their reference and segments that are mostly
`inserted`, and labels case-, punctuation- and footnote-marker-only differences `ignore`. Two kinds of page are
always kept: pages and segments with a **repetition loop** (a real `inserted` error of the OCR model) and every page
of a **hand-proofread** reference (`text_quality` `hand_proofread_*` in `catalog.csv`).

### 4b. Rendered pages (exact labels, real pipeline errors)

The reference text of the train books is set in period fonts (Noto Serif, EB Garamond, IM Fell, Old Standard,
PT Serif, Fraktur for German, Noto Serif CJK), printed to page images, degraded like a scan (blur, noise, skew,
contrast, JPEG) and read by the real pipeline. The words of every page are known exactly, so the labels are exact
and the errors are the pipeline's own. Only `references_train.jsonl` is rendered, so nothing leaks from valid or
test.

```bash
uv run pdf2fb2-bench export-references --split train --chunks-per-language 600
uv run pdf2fb2-bench render-references --split train --pages-per-language 200   # -> rendered_train/<language>/
uv run pdf2fb2-bench run-rendered --split train -- --no-verify --no-proofread --second-read
uv run pdf2fb2-bench export-rendered --split train --crops                       # -> rendered_train.jsonl
```

### 4c. Synthetic corruption of reference text

```bash
uv run pdf2fb2-bench export-references --split valid --chunks-per-language 150
workers/verifier_worker/.venv/bin/python -m token_verifier synth --split train
workers/verifier_worker/.venv/bin/python -m token_verifier synth --split valid
```

Synthetic examples imitate real pipeline output. Their kinds and proportions follow statistics of the training
split's real output, never its text:

* every chunk is cut into segments of log-uniform length (3..150 units, median about 20, like real segments),
  each with its neighbouring reference units as context;
* misread units: OCR confusion pairs (`с/е`, `ы/ые`, `rn/m`, `cl/d`, long `s`→`f`, old orthography `ѣ/ъ/ь`;
  the shared table is in `workers/worker_protocol/src/worker_protocol/ocr_confusions.py`), dropped diacritics,
  letter drops, duplications and swaps, two words merged by a lost space and a word split by a space or a
  leftover hyphen (`misread` piece plus `inserted` piece);
* inserted spans, mostly one unit: footnote markers, stray letters and marks, short notes, repeated runs;
* reference units that look like OCR errors of the reference itself (`goo4`, `mo&t`, `6hosen`, `〓`, mixed
  scripts) are labelled `ignore`: most "correct" references are machine OCR, not proofread text.

Synthetic text has no page evidence; during training the evidence of real texts is also hidden at random
(`evidence_dropout`), so the model works with and without it.

### 4d. Relabelling the real data (confident learning)

```bash
workers/verifier_worker/.venv/bin/python -m token_verifier relabel     # -> aligned_train_relabelled.jsonl
```

The train books are split into folds (`relabel.fold_count`). A model trained without a fold predicts that fold's
units; a label the model contradicts with probability ≥ `relabel.relabel_confidence` is replaced (`ignore` units
never change). Most such units are errors of a machine-OCR reference. `train` uses the relabelled file when it
exists (`training.use_relabelled_real`). The command prints how many units changed, e.g. `misread->correct`.

## 5. Fine-tune, tune thresholds, evaluate

```bash
cd workers/verifier_worker
.venv/bin/python -m token_verifier train           # two stages, best (EMA) epoch on valid
.venv/bin/python -m token_verifier tune            # thresholds on aligned_valid (train already does this)
.venv/bin/python -m token_verifier eval --split valid
.venv/bin/python -m token_verifier eval --split test   # final report only; never tune on it
```

How `train` works:

* **Backbone:** `paths.base_model`, any of xlm-roberta-base / mdeberta-v3-base (sub-word) or byt5-small
  (bytes). The pretrained **word embeddings stay frozen** (`freeze_word_embeddings`), so the multilingual embedding
  space keeps similar words of different languages close and languages absent from the data still work; lower
  layers learn more slowly than upper ones (`layer_decay`).
* **Unit head:** every unit is the max and mean of *all* its sub-word (or byte) states, joined with its page
  evidence, language-model plausibility and optional crop embedding.
* **Stage `mixed`:** synthetic + real + rendered windows for `epoch_count` epochs.
* **Stage `real`:** real + rendered windows with `synthetic_replay_share` of synthetic windows for
  `real_stage_epochs` epochs at `real_stage_learning_rate`.
* **Regularisation:** R-Drop (`rdrop_weight`) and an EMA of the weights (`ema_decay`); validation and saving use
  the EMA weights.
* **Per epoch:** loss, thresholds (grid 0.005–0.95) and per-class / per-language precision, recall and F1 are
  printed; the best epoch is saved; a stage stops after `early_stop_patience` epochs without improvement.
* **Ensemble:** with `ensemble_members` > 1, members trained with different seeds go to `member_<n>/`,
  `ensemble.json` lists them, their probabilities are averaged and the ensemble's thresholds are tuned at the end.

`train` writes `models/token_verifier/`: `backbone/`, `verifier_head.safetensors`, `verifier_config.json`
(backbone kind, evidence models, window settings) and `thresholds.json`.

Example runs with other backbones:

```bash
cat > /tmp/mdeberta.json <<'JSON'
{"paths": {"base_model": "models/mdeberta-v3-base", "output_model": "models/token_verifier_mdeberta"}}
JSON
.venv/bin/python -m token_verifier train --config /tmp/mdeberta.json

cat > /tmp/byt5.json <<'JSON'
{"paths": {"base_model": "models/byt5-small", "output_model": "models/token_verifier_byt5"},
 "training": {"backbone_kind": "byte", "window_units": 48, "max_tokens": 1024, "batch_size": 8,
              "gradient_accumulation": 4}}
JSON
.venv/bin/python -m token_verifier train --config /tmp/byt5.json

cat > /tmp/ensemble.json <<'JSON'
{"training": {"ensemble_members": 3}}
JSON
.venv/bin/python -m token_verifier train --config /tmp/ensemble.json
```

Evaluate another output directory by passing the same config to `eval`, and use it in the pipeline with
`pdf2fb2 --verify-model-dir models/token_verifier_mdeberta`.

`eval` writes `benchmark_data/verifier_reports/verifier_<split>.{md,json}`, with precision / recall / F1 of
`misread` and `inserted` over all languages and per language. On the test split, the `old_russian` rows are the
ground-truth number.

## 6. Settings

Defaults are in `src/token_verifier/default_settings.json`. To change them, pass a JSON file with only the keys
to change (`--config my_settings.json`, accepted by every command).

| key | default | effect |
|---|---|---|
| `paths.base_model` | models/xlm-roberta-base | backbone to fine-tune |
| `paths.language_model` | models/xlm-roberta-base | masked-LM plausibility features (`null` disables) |
| `paths.visual_model` | models/dinov2-small | word-crop embeddings (`null` disables) |
| `synthesis.misread_rate` | 0.06 | share of units corrupted per synthetic segment |
| `synthesis.insert_rate` | 0.4 | share of synthetic segments with an inserted span |
| `synthesis.min_segment_units` / `max_segment_units` | 3 / 150 | length range of synthetic segments |
| `training.backbone_kind` | auto | `subword`, `byte` or detected from the checkpoint |
| `training.window_units` / `max_tokens` / `context_units` | 96 / 384 / 32 | units per window / piece limit / context units per side |
| `training.class_weights` | 1, 1.5, 1.5 | loss weights of correct, misread, inserted |
| `training.freeze_word_embeddings` | true | keep the pretrained multilingual embeddings fixed |
| `training.layer_decay` | 0.9 | learning-rate factor per layer below the top (1 disables) |
| `training.rdrop_weight` | 0.5 | R-Drop consistency weight (0 disables; halves training speed when on) |
| `training.ema_decay` | 0.999 | EMA of the weights (0 disables) |
| `training.evidence_dropout` | 0.3 | probability of hiding a training text's page evidence |
| `training.epoch_count` | 4 | epochs of stage `mixed` |
| `training.real_stage_epochs` / `real_stage_learning_rate` | 2 / 1e-5 | stage `real` (0 epochs skips it) |
| `training.synthetic_replay_share` | 0.15 | share of synthetic windows in stage `real` |
| `training.real_repeat` / `rendered_repeat` | 1 / 1 | repetitions of real / rendered examples per epoch |
| `training.language_balance_cap` | 3 | largest oversampling of a language's real examples (1 = no balancing) |
| `training.early_stop_patience` | 2 | epochs without a better validation score before a stage stops (0 = never) |
| `training.ensemble_members` | 1 | seeds trained and averaged |
| `training.use_relabelled_real` | true | prefer `aligned_train_relabelled.jsonl` |
| `relabel.fold_count` / `fold_epochs` / `relabel_confidence` | 4 / 1 / 0.9 | confident-learning relabelling |

## 7. Benchmark after training

```bash
uv run pdf2fb2-bench run --split test --name verified --pages 10 -- --second-read --verify-crops
uv run pdf2fb2-bench run --split test --name judged   --pages 10 -- --second-read --verify-crops --judge
```

Compare `benchmark_runs/<name>/report.md` with `benchmark_runs/baseline/report.md`:

* The `ocr` level shows what the feedback re-reads changed.
* The `final` level shows everything, including the gated Qwen proofreading.

Inserted-word rate should fall (hallucinated text removed). CER should fall wherever lines were re-read
successfully. For each book, `benchmark_runs/<name>/<book>/conversion/verify_report.json` shows which units were
flagged and which re-reads were accepted.

To switch the verifier off for a conversion, use `--no-verify` (Qwen then scores the whole text). To keep
the verifier's re-reads but let Qwen score everything, use `--full-proofread`. To use another model
directory, use `--verify-model-dir`.

## 8. Complete command sequence

```bash
# setup
scripts/sync_environments.sh
scripts/download_verifier_assets.sh fonts            # add "mdeberta byt5" for the optional backbones
uv run pdf2fb2-bench splits
# real data (with OCR confidence and second read)
uv run pdf2fb2-bench run --yes --split train --name train_pages --pages 60 -- --no-verify --no-proofread --second-read
uv run pdf2fb2-bench run --yes --split valid --name valid_pages --pages 20 -- --no-verify --no-proofread --second-read
uv run pdf2fb2-bench run --yes --split test  --name baseline    --pages 10 -- --no-verify --second-read
uv run pdf2fb2-bench export-aligned --name train_pages --crops
uv run pdf2fb2-bench export-aligned --name valid_pages --crops
uv run pdf2fb2-bench export-aligned --name baseline --crops
# rendered and synthetic data
uv run pdf2fb2-bench export-references --split train --chunks-per-language 600
uv run pdf2fb2-bench export-references --split valid --chunks-per-language 150
uv run pdf2fb2-bench render-references --split train --pages-per-language 200
uv run pdf2fb2-bench run-rendered --split train -- --no-verify --no-proofread --second-read
uv run pdf2fb2-bench export-rendered --split train --crops
workers/verifier_worker/.venv/bin/python -m token_verifier synth --split train
workers/verifier_worker/.venv/bin/python -m token_verifier synth --split valid
# relabel, train, evaluate
workers/verifier_worker/.venv/bin/python -m token_verifier relabel
workers/verifier_worker/.venv/bin/python -m token_verifier train
workers/verifier_worker/.venv/bin/python -m token_verifier eval --split valid
workers/verifier_worker/.venv/bin/python -m token_verifier eval --split test
# benchmark with the verifier
uv run pdf2fb2-bench run --split test --name verified --pages 10 -- --second-read --verify-crops --judge
# optional: free the exported training crops once training and evaluation are finished
rm -rf benchmark_data/verify_crops
```

Conversions delete their own verification crops (`verify_crops/`, `judge_lines/`, `second_read/ocr_crops/`) when a
book has been verified. The crops written by `export-aligned --crops` / `export-rendered --crops` are training
inputs: keep them until `relabel`, `train` and `eval` have run (re-export to recreate them).

## Layout

| Path | Content |
|---|---|
| `src/token_verifier/datasets/` | JSONL examples, noise synthesis, windows with context and evidence, batches |
| `src/token_verifier/features/` | page evidence vectors, language-model plausibility, crop embeddings |
| `src/token_verifier/networks/` | backbone adapters (sub-word / byte), unit head, saving and loading, ensembles |
| `src/token_verifier/training/` | parameter groups, epoch loop (R-Drop, EMA), two-stage schedule, relabelling |
| `src/token_verifier/evaluation/` | metrics, threshold tuning, reports |
| `src/token_verifier/inference/` | batched prediction, pipeline job (`VerifyJob`, `worker_protocol/verify_job_schema.py`) |
| `workers/worker_protocol/.../unit_features.py` | names and normalisation of the page evidence |
| `workers/proofread_worker/.../visual_judge.py` | vision-language judge (`proofread_worker.judge_entry`) |
| `src/pdf2fb2/benchmark/` (orchestrator) | catalog, splits, references, scoring, exports, rendered pages |
| `src/pdf2fb2/text_verification/` (orchestrator) | units, evidence, second read, judge, re-read loop, report |

Tests: `cd workers/verifier_worker && .venv/bin/python -m pytest` (tiny local tokenizer and random tiny encoders;
nothing is downloaded or loaded).
