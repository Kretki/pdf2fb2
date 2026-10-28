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
        language_model: Masked-LM checkpoint for plausibility features (``models/xlm-roberta-base``), or None.
        visual_model: Vision checkpoint for word-crop embeddings (``models/dinov2-small``), or None.
    """

    base_model: Path
    output_model: Path
    data_directory: Path
    reports_directory: Path
    language_model: Path | None = None
    visual_model: Path | None = None


@dataclass(frozen=True)
class SynthesisSettings:
    """Corruption of reference text into labelled examples.

    Attributes:
        examples_per_chunk: Corrupted copies made of every reference chunk.
        misread_rate: Probability that a unit is misread.
        insert_rate: Probability that a segment receives an inserted span.
        max_insert_units: Longest repeated or remembered inserted run.
        random_seed: Seed of the generator.
        min_segment_units: Shortest synthetic segment (segment lengths are log-uniform).
        max_segment_units: Longest synthetic segment.
    """

    examples_per_chunk: int
    misread_rate: float
    insert_rate: float
    max_insert_units: int
    random_seed: int
    min_segment_units: int = 3
    max_segment_units: int = 150


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
        early_stop_patience: Epochs without a better validation score before training stops (0 never stops).
        backbone_kind: ``auto``, ``subword`` (XLM-R, mDeBERTa) or ``byte`` (ByT5).
        context_units: Unlabelled neighbouring units given to every window on each side.
        freeze_word_embeddings: Keep the pretrained (cross-lingual) word embeddings fixed.
        layer_decay: Layer-wise learning-rate decay per layer below the top (1 disables it).
        rdrop_weight: Weight of the R-Drop consistency loss between two dropout passes (0 disables it).
        ema_decay: Exponential moving average of the weights used for validation and saving (0 disables it).
        ensemble_members: Members trained with different seeds; their probabilities are averaged.
        evidence_dropout: Probability of hiding a training text's page evidence.
        real_stage_epochs: Epochs of the second stage on real (aligned and rendered) data (0 skips it).
        real_stage_learning_rate: Peak learning rate of the second stage.
        synthetic_replay_share: Share of synthetic windows mixed into the second stage.
        rendered_repeat: Repetitions of rendered-page examples (``rendered_train.jsonl``) per epoch.
        use_relabelled_real: Prefer ``aligned_train_relabelled.jsonl`` (written by ``relabel``) when present.
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
    language_balance_cap: int = 3
    early_stop_patience: int = 2
    backbone_kind: str = "auto"
    context_units: int = 32
    freeze_word_embeddings: bool = True
    layer_decay: float = 0.9
    rdrop_weight: float = 0.5
    ema_decay: float = 0.999
    ensemble_members: int = 1
    evidence_dropout: float = 0.3
    real_stage_epochs: int = 2
    real_stage_learning_rate: float = 1e-05
    synthetic_replay_share: float = 0.15
    rendered_repeat: int = 1
    use_relabelled_real: bool = True


@dataclass(frozen=True)
class RelabelSettings:
    """Confident-learning relabelling of the real training examples.

    Attributes:
        fold_count: Folds (by book) of cross-predictions.
        fold_epochs: Training epochs of every fold model.
        relabel_confidence: Probability the fold model needs to overrule a label.
    """

    fold_count: int = 4
    fold_epochs: int = 1
    relabel_confidence: float = 0.9


@dataclass(frozen=True)
class VerifierSettings:
    """All settings.

    Attributes:
        path_settings: Paths.
        synthesis_settings: Synthetic data.
        training_settings: Fine-tuning.
        relabel_settings: Confident-learning relabelling.
    """

    path_settings: PathSettings
    synthesis_settings: SynthesisSettings
    training_settings: TrainingSettings
    relabel_settings: RelabelSettings = RelabelSettings()


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
        path_settings=PathSettings(**{path_name: None if path_text is None else project_path(path_text)
                                      for path_name, path_text in path_values.items()}),
        synthesis_settings=SynthesisSettings(**settings_mapping["synthesis"]),
        training_settings=TrainingSettings(**training_values),
        relabel_settings=RelabelSettings(**settings_mapping.get("relabel", {})))
