"""Load the encoder and its tokenizer from a local directory (never from the network).

Training starts from the pretrained base model the admin downloaded (``models/xlm-roberta-base``) with a fresh
three-class token-classification head; the pipeline loads the fine-tuned directory ``models/token_verifier``,
which also holds ``thresholds.json``.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from worker_protocol.verify_job_schema import LABEL_NAMES

THRESHOLDS_FILE = "thresholds.json"
WINDOW_SETTINGS_FILE = "window_settings.json"
DEFAULT_THRESHOLDS = {"misread": 0.5, "inserted": 0.5}


@dataclass(frozen=True)
class LoadedVerifier:
    """Model ready for training or prediction.

    Attributes:
        text_tokenizer: Fast tokenizer.
        encoder_model: Token-classification model on its device.
        torch_device: Device of the model.
    """

    text_tokenizer: object
    encoder_model: torch.nn.Module
    torch_device: torch.device


def torch_device_of(device_name: str) -> torch.device:
    """Map ``gpu:N`` / ``cuda:N`` / ``cpu`` to a torch device (CPU when CUDA is unavailable).

    Args:
        device_name: Device text.

    Returns:
        Torch device.
    """
    cuda_name = device_name.replace("gpu", "cuda")
    return torch.device(cuda_name if cuda_name.startswith("cuda") and torch.cuda.is_available() else "cpu")


def load_verifier(model_directory: Path, device_name: str, new_head: bool = False) -> LoadedVerifier:
    """Load tokenizer and model.

    Args:
        model_directory: Local base model (``new_head=True``) or fine-tuned verifier directory.
        device_name: Device text.
        new_head: Create a fresh 3-class head (start of fine-tuning).

    Returns:
        The loaded verifier (in eval mode).

    Raises:
        OSError: If the directory does not hold a model.
    """
    label_mapping = {"id2label": dict(enumerate(LABEL_NAMES)),
                     "label2id": {label_name: label_position for label_position, label_name in enumerate(LABEL_NAMES)}}
    # B615: weights are loaded from a local directory only (local_files_only=True), never downloaded.
    text_tokenizer = AutoTokenizer.from_pretrained(str(model_directory), local_files_only=True)  # nosec B615
    model_options = {"num_labels": len(LABEL_NAMES), **label_mapping} if new_head else {}
    encoder_model = AutoModelForTokenClassification.from_pretrained(  # nosec B615
        str(model_directory), local_files_only=True, **model_options)
    target_device = torch_device_of(device_name)
    return LoadedVerifier(text_tokenizer, encoder_model.to(target_device).eval(), target_device)


def read_thresholds(model_directory: Path) -> dict[str, float]:
    """Decision thresholds tuned on the validation split.

    Args:
        model_directory: Fine-tuned verifier directory.

    Returns:
        ``{"misread": float, "inserted": float}`` (0.5 each when the file is missing).
    """
    thresholds_path = model_directory / THRESHOLDS_FILE
    if not thresholds_path.exists():
        return dict(DEFAULT_THRESHOLDS)
    stored_values = json.loads(thresholds_path.read_text(encoding="utf-8"))
    return {class_name: float(stored_values.get(class_name, default_value))
            for class_name, default_value in DEFAULT_THRESHOLDS.items()}


def read_window_settings(model_directory: Path, default_units: int, default_tokens: int) -> tuple[int, int]:
    """Window size the verifier was trained with.

    Args:
        model_directory: Fine-tuned verifier directory.
        default_units: Units per window when the file is missing.
        default_tokens: Sub-word limit when the file is missing.

    Returns:
        ``(window_units, max_tokens)``.
    """
    settings_path = model_directory / WINDOW_SETTINGS_FILE
    if not settings_path.exists():
        return default_units, default_tokens
    stored_values = json.loads(settings_path.read_text(encoding="utf-8"))
    return int(stored_values.get("window_units", default_units)), int(stored_values.get("max_tokens", default_tokens))
