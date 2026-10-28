"""Token verification job (orchestrator -> verifier worker): label every unit of OCR text segments.

The worker runs a fine-tuned multilingual encoder (token classification) over the units of every item and
returns, per unit, the probabilities of ``correct`` (read as printed), ``misread`` (wrongly parsed) and
``inserted`` (text that is not on the page: hallucinated by the OCR model or a reader's note), together with
the decision thresholds tuned on the validation split.

Optional evidence per item (all language-independent): neighbouring units of the page as unlabelled context,
per-unit page features (``unit_features``: OCR confidence, word boxes, second read) and per-unit crop images
(paths relative to the work directory) for the image-and-text verifier.

Result: JSON document ``{"thresholds": {"misread": float, "inserted": float}, "items": [{"key": str,
"probabilities": [[correct, misread, inserted], ...]}]}`` with one probability triple per unit.
"""

from dataclasses import dataclass, field
from pathlib import Path

from worker_protocol.field_validation import (
    require_absolute_directory,
    require_list,
    require_string,
    validate_device_name,
)
from worker_protocol.job_schema import checked_relative_path
from worker_protocol.path_safety import resolve_inside_directory
from worker_protocol.unit_features import validate_feature_columns

LABEL_NAMES = ("correct", "misread", "inserted")
MAX_VERIFY_ITEMS = 200_000
MAX_ITEM_UNITS = 20_000
MAX_UNIT_LENGTH = 512
MAX_KEY_LENGTH = 128
MAX_CONTEXT_UNITS = 256
MAX_CROP_PATH_LENGTH = 512


@dataclass(frozen=True)
class VerifyItem:
    """One text to verify (normally one segment).

    Attributes:
        item_key: Caller's identifier, echoed in the result.
        item_units: Units of the text in order (``pdf2fb2.text_verification.verify_units``).
        context_before: Units of the page just before the item (not labelled).
        context_after: Units of the page just after the item (not labelled).
        unit_features: ``{feature name: value or None per unit}`` (``worker_protocol.unit_features``).
        unit_crops: Per unit, the crop image path relative to the work directory, or None.
    """

    item_key: str
    item_units: tuple[str, ...]
    context_before: tuple[str, ...] = ()
    context_after: tuple[str, ...] = ()
    unit_features: dict[str, tuple[float | None, ...]] = field(default_factory=dict)
    unit_crops: tuple[str | None, ...] = ()

    def to_payload(self) -> dict:
        """Serialise the item (optional fields only when set).

        Returns:
            JSON-compatible dictionary.
        """
        item_payload: dict = {"key": self.item_key, "units": list(self.item_units)}
        if self.context_before:
            item_payload["context_before"] = list(self.context_before)
        if self.context_after:
            item_payload["context_after"] = list(self.context_after)
        if self.unit_features:
            item_payload["features"] = {feature_name: list(column_values)
                                        for feature_name, column_values in self.unit_features.items()}
        if self.unit_crops:
            item_payload["crops"] = list(self.unit_crops)
        return item_payload


def validated_units(item_entry: dict, field_name: str, max_count: int, required: bool) -> tuple[str, ...]:
    """Decode a list of units.

    Args:
        item_entry: Item object.
        field_name: ``units``, ``context_before`` or ``context_after``.
        max_count: Largest number of units.
        required: Whether the field must be present.

    Returns:
        The units (empty when an optional field is absent).

    Raises:
        ValueError: If a unit is not a non-blank string of bounded length.
    """
    if not required and item_entry.get(field_name) is None:
        return ()
    unit_entries = require_list(item_entry, field_name, max_count)
    if not all(isinstance(unit_text, str) and 0 < len(unit_text) <= MAX_UNIT_LENGTH and not unit_text.isspace()
               for unit_text in unit_entries):
        raise ValueError(f"Units must be non-blank strings of at most {MAX_UNIT_LENGTH} characters.")
    return tuple(unit_entries)


def validated_crops(item_entry: dict, unit_count: int) -> tuple[str | None, ...]:
    """Decode the per-unit crop paths (relative, no ``..``, checked again against the work directory on use).

    Args:
        item_entry: Item object.
        unit_count: Units of the item.

    Returns:
        One path or None per unit (empty when absent).

    Raises:
        ValueError: If the list length differs from the units or a path is unsafe.
    """
    crop_entries = item_entry.get("crops")
    if crop_entries is None:
        return ()
    if not isinstance(crop_entries, list) or len(crop_entries) != unit_count:
        raise ValueError("Crops need one path or null per unit.")
    for crop_path in crop_entries:
        if crop_path is None:
            continue
        if not isinstance(crop_path, str) or not crop_path or len(crop_path) > MAX_CROP_PATH_LENGTH \
                or crop_path.startswith("/") or ".." in crop_path.replace("\\", "/").split("/"):
            raise ValueError("Crop paths must be relative paths inside the work directory.")
    return tuple(crop_entries)


def validate_verify_item(item_entry: object) -> VerifyItem:
    """Decode one item.

    Args:
        item_entry: Decoded JSON value.

    Returns:
        The validated item.

    Raises:
        ValueError: If the key or the units are missing, too long or not strings, or an optional field is
            malformed.
    """
    if not isinstance(item_entry, dict):
        raise ValueError("Each verify item must be an object.")
    item_key = require_string(item_entry, "key", MAX_KEY_LENGTH)
    item_units = validated_units(item_entry, "units", MAX_ITEM_UNITS, True)
    return VerifyItem(item_key, item_units,
                      validated_units(item_entry, "context_before", MAX_CONTEXT_UNITS, False),
                      validated_units(item_entry, "context_after", MAX_CONTEXT_UNITS, False),
                      validate_feature_columns(item_entry.get("features"), len(item_units)),
                      validated_crops(item_entry, len(item_units)))


@dataclass(frozen=True)
class VerifyJob:
    """Request to label the units of OCR texts.

    Usage:
        Built by the orchestrator, serialised with ``to_payload`` and decoded by the verifier worker with
        ``from_payload``.

    Attributes:
        work_directory: Absolute directory the result path refers to.
        model_directory: Absolute directory of the fine-tuned verifier (``models/token_verifier``).
        device_name: Torch device, e.g. ``gpu:0`` / ``cpu``.
        verify_items: Texts to label.
        result_file: Relative path of the JSON result document.
    """

    work_directory: Path
    model_directory: Path
    device_name: str
    verify_items: tuple[VerifyItem, ...]
    result_file: str

    def to_payload(self) -> dict:
        """Serialise the job.

        Returns:
            Dictionary ready for ``json.dumps``.
        """
        return {"work_directory": str(self.work_directory), "model_directory": str(self.model_directory),
                "device_name": self.device_name, "items": [verify_item.to_payload() for verify_item
                                                           in self.verify_items],
                "result_file": self.result_file}

    @classmethod
    def from_payload(cls, job_payload: dict) -> "VerifyJob":
        """Decode and validate a verification job.

        Args:
            job_payload: Decoded JSON object produced by ``to_payload``.

        Returns:
            The validated job.

        Raises:
            ValueError: If any field is missing, out of range or escapes the working directory.
        """
        work_directory = require_absolute_directory(require_string(job_payload, "work_directory"), "work_directory")
        return cls(
            work_directory=work_directory,
            model_directory=require_absolute_directory(require_string(job_payload, "model_directory"),
                                                       "model_directory"),
            device_name=validate_device_name(require_string(job_payload, "device_name", 16)),
            verify_items=tuple(validate_verify_item(item_entry)
                               for item_entry in require_list(job_payload, "items", MAX_VERIFY_ITEMS)),
            result_file=checked_relative_path(work_directory, require_string(job_payload, "result_file")),
        )

    def resolve_inside(self, relative_path: str) -> Path:
        """Resolve a relative job path against the working directory.

        Args:
            relative_path: Path relative to ``work_directory``.

        Returns:
            Absolute path inside ``work_directory``.

        Raises:
            ValueError: If the path escapes ``work_directory``.
        """
        return resolve_inside_directory(self.work_directory, relative_path)
