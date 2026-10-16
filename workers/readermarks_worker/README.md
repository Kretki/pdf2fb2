# readermarks worker — handwritten reader-mark detection and removal

Removes reader marks (handwritten notes, pen/pencil underlines, circles, margin brackets/ticks,
highlighter, strike-throughs) from rendered pages. In pdf2fb2 it runs **after** the first layout
(PP-DocLayoutV3) + OCR (Qianfan-OCR) pass: picture segments are sent as `masked_regions` and blanked
with paper colour before detection; pages with marks are cleaned and then segmented and OCR'd again.
Printed text and printed rules are kept. Pages without marks pass through as a byte-identical copy. Design: STAGE_CLEAN.md (spot-then-segment after Pandey & Harit, 2021).

## Setup (RTX 50-series, offline)

```bash
uv sync --project workers/readermarks_worker      # torch cu130 wheels (sm_120)
workers/readermarks_worker/.venv/bin/python -m readermarks env-check
```

Weights used (local only, `HF_HUB_OFFLINE=1`, `local_files_only=True`):

| Model | Path | License |
|---|---|---|
| DINOv2 ViT-S/14 (frozen spotter backbone, configurable) | `models/dinov2-small` | Apache 2.0 |
| ConvNeXt-Tiny ImageNet (U-Net encoder) | `models/timm/convnext_tiny.fb_in22k_ft_in1k` | Apache 2.0 |
| Trained spotter / segmenter / thresholds | `models/readermarks/*.safetensors`, `thresholds.json` | own |

Exact package versions: `uv.lock` and `requirements.lock.txt`.

## Pipeline

1. **Stage 1 spotter** — page resized to 1344 px long side → frozen DINOv2 patch tokens →
   3×(3×3 conv, GroupNorm, GELU) head → 9 multi-label patch classes + page-level `has_marks` logit.
   Flagged regions = connected above-threshold mark patches, dilated by 2 patches.
2. **Stage 2 segmenter** — U-Net (ConvNeXt-Tiny) on 768×768 native-resolution crops of the flagged
   regions → `mark_any` + 7 mark classes + `printed_ink` + `printed_rule` (independent sigmoids).
3. **Post-processing** — components, class vote, geometric filter (fit residual, thickness variation,
   slope, overshoot past word boxes) that vetoes printed rules.
4. **Removal** — mark-only pixels (dilated mark mask minus printed ink) get the local paper colour
   (median of nearby non-ink pixels on a tile grid). No generative inpainting.

Outputs per page: `clean.png`, `marks_mask.png`, `masks/<class>.png`, `marks.json`
(class, bbox, confidence, contour, polyline, touched printed word boxes), optional `debug.png`
(marks red, vetoed printed lines green, flagged patches blue grid).

## Data layout

| Path | Content |
|---|---|
| `test_res/pages/page_NNNN.png` | real pages (200 dpi) |
| `src/readermarks/settings/real_page_labels.yaml` | page-level labels (pages 1–101) |
| `readermarks_data/splits.json` | locked split: synthetic bases / real validation / real test / unlabeled |
| `readermarks_data/synthetic/{train,validation,test}` | synthetic pages + bit-packed masks + manifest |
| `readermarks_data/features/dinov2-small/*` | fp16 feature cache |
| `readermarks_data/previews/train/preview_*.png` | 50 synthetic samples with masks |
| `readermarks_data/reports/real_{val,test}.{md,json}` | evaluation reports |

## Commands

```bash
cd workers/readermarks_worker
.venv/bin/python -m readermarks synth             # 2400/200/200 synthetic pages
.venv/bin/python -m readermarks preview           # 50 previews
.venv/bin/python -m readermarks cache-features
.venv/bin/python -m readermarks train-spotter
.venv/bin/python -m readermarks train-segmenter
.venv/bin/python -m readermarks tune              # thresholds on real validation pages only
.venv/bin/python -m readermarks eval --split real_val
.venv/bin/python -m readermarks eval --split real_test   # final report only
.venv/bin/python -m readermarks clean --input page.png --out out/ --debug
.venv/bin/python -m readermarks clean --input-dir pages/ --out out/ --batch
.venv/bin/python -m readermarks convert-labels --export labelstudio.json --out labels/
.venv/bin/python -m readermarks active-learning   # top-30 uncertain unlabeled pages → Label Studio tasks
.venv/bin/python -m pytest
```

In pdf2fb2 the stage runs automatically after the first OCR pass
(`uv run pdf2fb2 book.pdf -o out`); disable with `--keep-marks`, overlays with `--debug-marks`.
Evaluation, tuning and active learning blank the same picture regions, read from
`path_settings.layout_segments_file` (default `test_res/segments.json`).

## Results (2026-10-16, pictures blanked as in pdf2fb2, page 76 labelled as marked)

Data: 212 real pages at 200 dpi; page-level labels for pages 1–101 (41 marked, 60 unmarked). Locked split:
35 unmarked pages → synthetic training bases, 20 pages → real validation (threshold tuning),
46 pages → real test (30 marked + 16 unmarked), 111 unlabeled pages → active learning.
Tuned thresholds (`models/readermarks/thresholds.json`): page 0.4, patch 0.2, pixel 0.5, min 40 mark px.

| Metric | Target | Real val (20) | Real test (46) |
|---|---|---|---|
| page-level `has_marks` accuracy | ≥ 0.97 | 1.00 | 0.978 (30/30 marked found, 1 false positive) |
| unmarked page altered | never | 0 | 1 (the false positive) |
| full pipeline, marked page | ≤ 2 s | 0.10 s | 0.13 s |
| peak VRAM (inference / training) | ≤ 10 GB | 1.2 GB / 7.1 GB | 1.2 GB / 7.1 GB |

The test split was evaluated before (0.935, without picture blanking and with page 76 mislabelled);
this second evaluation follows the pipeline change, thresholds were tuned on validation pages only.

Synthetic test split: pixel `mark_any` IoU 0.62 (P 0.83, R 0.72); spotter `mark_any` patches at the
tuned permissive patch threshold 0.2 (a region gate only): P 0.30 / R 0.88 — low precision by design,
Stage 2 and the geometric filter make the final decision.

Known failure modes: coloured digitisation watermarks ("Digitized by the Internet Archive", test page 4)
are removed as marks. Weak synthetic classes: `strike_through` (merged into `hand_underline`),
`hand_text_inline`, `highlighter`; they need real labeled examples or stroke scans.
Full reports: `readermarks_data/reports/real_{val,test}.md`.
