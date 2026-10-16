#!/usr/bin/env bash
# sync_environments.sh - create/update all six virtual environments of pdf2fb2.
#
# Usage:
#   scripts/sync_environments.sh [extra uv sync arguments...]
#   e.g. scripts/sync_environments.sh --frozen
#
# Arguments:
#   Any arguments are passed unchanged to every `uv sync` call.
#
# Outputs:
#   .venv                         orchestrator (PyMuPDF, Pillow, dev/security tools)
#   workers/layout_worker/.venv   PaddlePaddle GPU + PaddleOCR (PP-DocLayoutV3)
#   workers/ocr_worker/.venv      PyTorch CUDA + transformers (Qianfan-OCR)
#   workers/readermarks_worker/.venv  PyTorch CUDA + timm/smp/OpenCV (reader-mark removal)
#   workers/textar_worker/.venv   PyTorch CUDA + docTR (TexTAR word styles)
#   workers/proofread_worker/.venv  PyTorch CUDA + transformers/bitsandbytes (Qwen3.5 proofreading)
#   uv.lock in each project directory. Exits non-zero on the first failing sync.
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for project_directory in "${repository_root}" \
                         "${repository_root}/workers/layout_worker" \
                         "${repository_root}/workers/ocr_worker" \
                         "${repository_root}/workers/readermarks_worker" \
                         "${repository_root}/workers/textar_worker" \
                         "${repository_root}/workers/proofread_worker"; do
  echo ">>> uv sync --project ${project_directory}"
  uv sync --project "${project_directory}" "$@"
done
