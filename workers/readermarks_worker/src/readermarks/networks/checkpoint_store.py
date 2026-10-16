"""Save/load model checkpoints (safetensors + JSON metadata, no pickle) and tuned thresholds."""

import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

SPOTTER_FILE = "spotter.safetensors"
SEGMENTER_FILE = "segmenter.safetensors"
THRESHOLDS_FILE = "thresholds.json"
METADATA_KEY = "readermarks_meta"


def save_checkpoint(checkpoint_path: Path, torch_module: nn.Module, metadata: dict) -> None:
    """Store a module's weights with JSON-compatible metadata.

    Args:
        checkpoint_path: Destination ``.safetensors`` file; the directory is created.
        torch_module: Module to save.
        metadata: Architecture parameters (str/int/float/list only).

    Returns:
        None.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_state = {
        parameter_name: parameter_value.detach().contiguous().cpu()
        for parameter_name, parameter_value in torch_module.state_dict().items()
    }
    save_file(tensor_state, str(checkpoint_path), metadata={METADATA_KEY: json.dumps(metadata)})


def load_checkpoint(checkpoint_path: Path) -> tuple[dict, dict]:
    """Load weights and metadata (safetensors never executes code).

    Args:
        checkpoint_path: File written by ``save_checkpoint``.

    Returns:
        ``(state_dict, metadata)`` on the CPU.

    Raises:
        FileNotFoundError: If the checkpoint does not exist (model not trained yet).
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found; train the model first.")
    with safe_open(str(checkpoint_path), framework="pt") as checkpoint_file:
        metadata = json.loads(checkpoint_file.metadata()[METADATA_KEY])
    return load_file(str(checkpoint_path)), metadata


def save_thresholds(checkpoint_directory: Path, threshold_values: dict) -> Path:
    """Write tuned cleaning thresholds next to the checkpoints.

    Args:
        checkpoint_directory: Checkpoint directory.
        threshold_values: Mapping of ``CleaningSettings`` field -> value.

    Returns:
        Path of ``thresholds.json``.
    """
    thresholds_path = checkpoint_directory / THRESHOLDS_FILE
    thresholds_path.write_text(json.dumps(threshold_values, indent=1), encoding="utf-8")
    return thresholds_path


def load_thresholds(checkpoint_directory: Path) -> dict:
    """Read tuned thresholds, or an empty mapping when none were tuned.

    Args:
        checkpoint_directory: Checkpoint directory.

    Returns:
        Mapping of ``CleaningSettings`` field -> value.
    """
    thresholds_path = checkpoint_directory / THRESHOLDS_FILE
    if not thresholds_path.is_file():
        return {}
    return json.loads(thresholds_path.read_text(encoding="utf-8"))
