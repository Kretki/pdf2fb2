"""Training data: synthetic, real (pipeline output aligned with references) and rendered-page examples as windows.

* ``synthetic_train.jsonl``: corrupted reference text (no page evidence);
* ``aligned_train.jsonl`` — or ``aligned_train_relabelled.jsonl`` written by ``relabel`` — real pipeline output,
  repeated ``real_repeat`` times and oversampled per language by at most ``language_balance_cap``;
* ``rendered_train.jsonl``: reference text rendered to page images and read by the real pipeline (exact labels
  with the pipeline's own error distribution), repeated ``rendered_repeat`` times.

Selection data: ``aligned_valid.jsonl`` when it exists, else ``synthetic_valid.jsonl``.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from token_verifier.datasets.example_files import TokenExample, read_examples
from token_verifier.datasets.token_encoding import EncodedWindow, text_windows
from token_verifier.datasets.verify_texts import VerifyText, text_from_example
from token_verifier.features.feature_assembly import FeatureAssembler
from token_verifier.networks.backbone_adapters import BackboneAdapter
from token_verifier.verifier_settings import TrainingSettings

RELABELLED_FILE = "aligned_train_relabelled.jsonl"


@dataclass
class TrainingCorpus:
    """Windows of both training stages and the selection texts.

    Attributes:
        synthetic_windows: Windows of synthetic examples.
        real_windows: Windows of real and rendered examples.
        validation_pairs: Texts used to pick epochs and thresholds, with their language names.
    """

    synthetic_windows: list[EncodedWindow]
    real_windows: list[EncodedWindow]
    validation_pairs: list[tuple[VerifyText, str]]


def language_balanced(token_examples: list[TokenExample], balance_cap: int) -> list[TokenExample]:
    """Oversample the languages with few examples towards the largest one.

    Args:
        token_examples: Examples of all languages.
        balance_cap: Largest repetition factor of one language.

    Returns:
        Examples with every language repeated ``clamp(round(largest / own count), 1, cap)`` times (rounded, so
        a language almost as large as the largest one is not doubled).
    """
    examples_by_language: dict[str, list[TokenExample]] = defaultdict(list)
    for token_example in token_examples:
        examples_by_language[token_example.language_name].append(token_example)
    largest_count = max((len(language_examples) for language_examples in examples_by_language.values()), default=0)
    return [token_example for language_examples in examples_by_language.values()
            for token_example in language_examples * max(1, min(balance_cap,
                                                                 round(largest_count / len(language_examples))))]


def aligned_training_file(data_directory: Path, training_settings: TrainingSettings) -> Path:
    """Real training file to use.

    Args:
        data_directory: Directory of the JSONL files.
        training_settings: ``use_relabelled_real``.

    Returns:
        The relabelled file when it exists and is enabled, else ``aligned_train.jsonl``.
    """
    relabelled_path = data_directory / RELABELLED_FILE
    if training_settings.use_relabelled_real and relabelled_path.exists():
        return relabelled_path
    return data_directory / "aligned_train.jsonl"


def real_examples(data_directory: Path, training_settings: TrainingSettings,
                  aligned_override: list[TokenExample] | None = None) -> list[TokenExample]:
    """Real and rendered training examples with their repetitions.

    Args:
        data_directory: Directory of the JSONL files.
        training_settings: Repetition and balancing settings.
        aligned_override: Aligned examples to use instead of the file (relabelling folds).

    Returns:
        Examples.
    """
    aligned_examples = aligned_override if aligned_override is not None else read_examples(
        aligned_training_file(data_directory, training_settings))
    balanced_examples = language_balanced(aligned_examples, training_settings.language_balance_cap)
    rendered_examples = read_examples(data_directory / "rendered_train.jsonl")
    return balanced_examples * training_settings.real_repeat + rendered_examples * training_settings.rendered_repeat


def selection_examples(data_directory: Path) -> list[TokenExample]:
    """Examples used to pick the best epoch and its thresholds.

    Args:
        data_directory: Directory of the JSONL files.

    Returns:
        Real validation output when available, else synthetic validation examples.
    """
    return read_examples(data_directory / "aligned_valid.jsonl") or \
        read_examples(data_directory / "synthetic_valid.jsonl")


def example_windows(token_examples: list[TokenExample], data_directory: Path, backbone_adapter: BackboneAdapter,
                    feature_assembler: FeatureAssembler, training_settings: TrainingSettings) -> list[EncodedWindow]:
    """Disjoint training windows of examples (evidence assembled once per example).

    Args:
        token_examples: Labelled examples.
        data_directory: Directory crop paths are relative to.
        backbone_adapter: Input encoding.
        feature_assembler: Evidence providers.
        training_settings: Window settings.

    Returns:
        Windows.
    """
    window_settings = (training_settings.window_units, training_settings.max_tokens, training_settings.context_units)
    encoded_windows: list[EncodedWindow] = []
    for token_example in token_examples:
        verify_text = text_from_example(token_example, data_directory)
        unit_evidence = feature_assembler.assemble(verify_text, True)
        encoded_windows.extend(text_windows(backbone_adapter, verify_text, unit_evidence, window_settings, False))
    return encoded_windows


def training_corpus(data_directory: Path, backbone_adapter: BackboneAdapter, feature_assembler: FeatureAssembler,
                    training_settings: TrainingSettings,
                    aligned_override: list[TokenExample] | None = None) -> TrainingCorpus:
    """All training windows and selection texts.

    Usage:
        Called once per trained member (and per relabelling fold).

    Args:
        data_directory: Directory of the JSONL files.
        backbone_adapter: Input encoding.
        feature_assembler: Evidence providers.
        training_settings: Settings.
        aligned_override: Aligned examples replacing the file (relabelling folds).

    Returns:
        The corpus.

    Raises:
        FileNotFoundError: If there is no training data at all.
    """
    synthetic_windows = example_windows(read_examples(data_directory / "synthetic_train.jsonl"), data_directory,
                                        backbone_adapter, feature_assembler, training_settings)
    real_windows = example_windows(real_examples(data_directory, training_settings, aligned_override), data_directory,
                                   backbone_adapter, feature_assembler, training_settings)
    if not synthetic_windows and not real_windows:
        raise FileNotFoundError(f"No synthetic, aligned or rendered training data in {data_directory}.")
    validation_pairs = [(text_from_example(token_example, data_directory), token_example.language_name)
                        for token_example in selection_examples(data_directory)]
    return TrainingCorpus(synthetic_windows, real_windows, validation_pairs)


def replay_sample(synthetic_windows: list[EncodedWindow], real_count: int, replay_share: float,
                  sample_generator: object) -> list[EncodedWindow]:
    """Synthetic windows mixed into a real-data epoch.

    Args:
        synthetic_windows: All synthetic windows.
        real_count: Real windows of the epoch.
        replay_share: Wanted share of synthetic windows in the epoch (0..0.9).
        sample_generator: ``random.Random`` used for sampling.

    Returns:
        A random sample.
    """
    bounded_share = min(max(replay_share, 0.0), 0.9)
    wanted_count = min(len(synthetic_windows), math.ceil(real_count * bounded_share / (1 - bounded_share)))
    return sample_generator.sample(synthetic_windows, wanted_count) if wanted_count else []
