"""One text to label with everything the verifier may use about it, built from a training example or a job item.

Crop paths are made absolute here: relative to the data directory for training examples and checked to stay
inside the work directory for pipeline jobs.
"""

from dataclasses import dataclass, field
from pathlib import Path

from token_verifier.datasets.example_files import TokenExample
from worker_protocol.path_safety import resolve_inside_directory
from worker_protocol.verify_job_schema import VerifyItem


@dataclass(frozen=True)
class VerifyText:
    """Units with optional labels and evidence.

    Attributes:
        unit_texts: Units to label.
        unit_labels: Labels (training / evaluation) or empty (prediction).
        context_before: Unlabelled units before the text.
        context_after: Unlabelled units after the text.
        unit_features: Page evidence columns (``worker_protocol.unit_features``).
        crop_paths: Absolute crop image per unit, or None (empty when no crops).
    """

    unit_texts: tuple[str, ...]
    unit_labels: tuple[str, ...] = ()
    context_before: tuple[str, ...] = ()
    context_after: tuple[str, ...] = ()
    unit_features: dict[str, tuple[float | None, ...]] = field(default_factory=dict)
    crop_paths: tuple[Path | None, ...] = ()


def text_from_example(token_example: TokenExample, data_directory: Path) -> VerifyText:
    """Training or evaluation text of an example.

    Args:
        token_example: Labelled example.
        data_directory: Directory crop paths are relative to.

    Returns:
        The text with labels.

    Raises:
        ValueError: If a crop path leaves the data directory.
    """
    crop_paths = tuple(None if crop_path is None else resolve_inside_directory(data_directory, crop_path)
                       for crop_path in token_example.unit_crops)
    return VerifyText(token_example.unit_texts, token_example.unit_labels, token_example.context_before,
                      token_example.context_after, token_example.unit_features, crop_paths)


def text_from_item(verify_item: VerifyItem, work_directory: Path) -> VerifyText:
    """Prediction text of a pipeline job item.

    Args:
        verify_item: Validated item.
        work_directory: Job work directory (crop paths are relative to it).

    Returns:
        The text without labels.

    Raises:
        ValueError: If a crop path leaves the work directory.
    """
    crop_paths = tuple(None if crop_path is None else resolve_inside_directory(work_directory, crop_path)
                       for crop_path in verify_item.unit_crops)
    return VerifyText(verify_item.item_units, (), verify_item.context_before, verify_item.context_after,
                      verify_item.unit_features, crop_paths)
