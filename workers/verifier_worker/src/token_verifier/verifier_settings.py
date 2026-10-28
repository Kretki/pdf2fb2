"""Settings of the verifier commands: ``default_settings.json`` merged with an optional ``--config`` JSON file.

Relative paths are resolved against the pdf2fb2 project root (two levels above ``workers/verifier_worker``).
"""

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SETTINGS_FILE = Path(__file__).with_name("default_settings.json")
MAX_SETTINGS_BYTES = 1_000_000


@dataclass(frozen=True)
class PathSettings:
    """Where models and data live.

    Attributes:
        base_model: Pretrained encoder downloaded by the admin (e.g. ``models/xlm-roberta-base``).
        output_model: Fine-tuned verifier used by the pipeline (``models/token_verifier``).
        data_directory: Exported and synthetic JSONL files (``benchmark_data``).
        reports_directory: Evaluation reports.
    """

    base_model: Path
    output_model: Path
    data_directory: Path
    reports_directory: Path


@dataclass(frozen=True)
class SynthesisSettings:
    """Corruption of reference text into labelled examples.

    Attributes:
        examples_per_chunk: Corrupted copies made of every reference chunk.
        misread_rate: Probability that a unit is misread.
        insert_rate: Probability that an example receives an inserted span.
        max_insert_units: Longest inserted span.
        random_seed: Seed of the generator.
    """

    examples_per_chunk: int
    misread_rate: float
    insert_rate: float
    max_insert_units: int
    random_seed: int


@dataclass(frozen=True)
class TrainingSettings:
    """Fine-tuning hyper-parameters.

    Attributes:
        device_name: Torch device (``cuda:0``).
        max_tokens: Sub-word tokens per window.
        window_units: Units per training window.
        batch_size: Windows per step.
        gradient_accumulation: Steps per optimizer update.
        learning_rate: Peak AdamW learning rate.
        weight_decay: AdamW weight decay.
        warmup_ratio: Share of updates with linear warm-up.
        epoch_count: Passes over the training data.
        class_weights: Loss weights of correct, misread, inserted.
        real_repeat: Times every real (pipeline) example is repeated per epoch.
        language_balance_cap: Largest oversampling factor that brings a language's real examples towards the
            language with the most of them (1 disables balancing).
        mixed_precision: Use bfloat16 autocast on CUDA.
        random_seed: Seed of shuffling and initialisation.
    """

    device_name: str
    max_tokens: int
    window_units: int
    batch_size: int
    gradient_accumulation: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    epoch_count: int
    class_weights: tuple[float, float, float]
    real_repeat: int
    mixed_precision: bool
    random_seed: int
    language_balance_cap: int = 8


@dataclass(frozen=True)
class VerifierSettings:
    """All settings.

    Attributes:
        path_settings: Paths.
        synthesis_settings: Synthetic data.
        training_settings: Fine-tuning.
    """

    path_settings: PathSettings
    synthesis_settings: SynthesisSettings
    training_settings: TrainingSettings


def project_path(path_text: str) -> Path:
    """Resolve a settings path.

    Args:
        path_text: Absolute path or path relative to the project root.

    Returns:
        Absolute path.

    Raises:
        ValueError: If a relative path contains ``..``.
    """
    candidate_path = Path(path_text)
    if not candidate_path.is_absolute() and ".." in candidate_path.parts:
        raise ValueError(f"Settings path must not contain '..': {path_text!r}")
    return candidate_path if candidate_path.is_absolute() else PROJECT_ROOT / candidate_path


def read_settings_file(settings_path: Path) -> dict:
    """Read one JSON settings file.

    Args:
        settings_path: File to read.

    Returns:
        Decoded mapping.

    Raises:
        ValueError: If the file is too large or not a JSON object.
    """
    if settings_path.stat().st_size > MAX_SETTINGS_BYTES:
        raise ValueError(f"{settings_path} is too large.")
    settings_mapping = json.loads(settings_path.read_text(encoding="utf-8"))
    if not isinstance(settings_mapping, dict):
        raise ValueError(f"{settings_path} must hold a JSON object.")
    return settings_mapping


def merged_sections(override_path: Path | None) -> dict:
    """Default settings with the override file applied section by section.

    Args:
        override_path: Optional JSON file with some of the keys.

    Returns:
        Merged mapping.
    """
    settings_mapping = read_settings_file(DEFAULT_SETTINGS_FILE)
    if override_path is not None:
        for section_name, section_values in read_settings_file(override_path).items():
            settings_mapping.setdefault(section_name, {}).update(section_values)
    return settings_mapping


def load_settings(override_path: Path | None = None) -> VerifierSettings:
    """Load and type the settings.

    Usage:
        Called once by every subcommand.

    Args:
        override_path: Optional ``--config`` file.

    Returns:
        The settings.

    Raises:
        ValueError: If a key is missing, unknown or has a wrong type.
        TypeError: If a section holds unknown keys.
    """
    settings_mapping = merged_sections(override_path)
    path_values = settings_mapping["paths"]
    training_values = dict(settings_mapping["training"])
    training_values["class_weights"] = tuple(float(class_weight) for class_weight in training_values["class_weights"])
    if len(training_values["class_weights"]) != 3:
        raise ValueError("training.class_weights needs three values (correct, misread, inserted).")
    return VerifierSettings(
        path_settings=PathSettings(**{path_name: project_path(path_text)
                                      for path_name, path_text in path_values.items()}),
        synthesis_settings=SynthesisSettings(**settings_mapping["synthesis"]),
        training_settings=TrainingSettings(**training_values))
