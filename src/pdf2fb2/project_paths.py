"""Fixed locations inside the repository (models, workers)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIRECTORY = PROJECT_ROOT / "models"
LAYOUT_MODEL_DIRECTORY = MODELS_DIRECTORY / "PP-DocLayoutV3"
QIANFAN_MODEL_DIRECTORY = MODELS_DIRECTORY / "Qianfan-OCR"
WORKERS_DIRECTORY = PROJECT_ROOT / "workers"
LAYOUT_WORKER_DIRECTORY = WORKERS_DIRECTORY / "layout_worker"
OCR_WORKER_DIRECTORY = WORKERS_DIRECTORY / "ocr_worker"
