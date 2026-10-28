"""Labelled examples on disk (JSON lines) and the label scheme.

A line is ``{"book_key", "language", "words": [...], "labels": [...]}`` with labels ``correct``, ``misread``,
``inserted`` or ``ignore`` (markup tokens: no loss, no metric). Reference chunk files have no labels.

Optional keys carry the page evidence the pipeline also sends at prediction time: ``context_before`` /
``context_after`` (neighbouring units, unlabelled), ``features`` (``{name: [value | null per unit]}``, see
``worker_protocol.unit_features``) and ``crops`` (per unit an image path relative to the data directory, or null).
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from worker_protocol.unit_features import validate_feature_columns
from worker_protocol.verify_job_schema import LABEL_NAMES

IGNORE_LABEL = "ignore"
IGNORE_INDEX = -100
LABEL_INDEXES = {label_name: label_index for label_index, label_name in enumerate(LABEL_NAMES)}
MAX_FILE_BYTES = 4_000_000_000
MAX_LINE_UNITS = 50_000


@dataclass(frozen=True)
class TokenExample:
    """One labelled sequence of units.

    Attributes:
        book_key: Source book.
        language_name: Corpus language.
        unit_texts: Units in order.
        unit_labels: One label per unit.
        context_before: Units of the page before the example (no labels).
        context_after: Units of the page after the example (no labels).
        unit_features: Page evidence per unit (``worker_protocol.unit_features``).
        unit_crops: Per unit a crop image path (relative to the data directory) or None.
    """

    book_key: str
    language_name: str
    unit_texts: tuple[str, ...]
    unit_labels: tuple[str, ...]
    context_before: tuple[str, ...] = ()
    context_after: tuple[str, ...] = ()
    unit_features: dict[str, tuple[float | None, ...]] = field(default_factory=dict)
    unit_crops: tuple[str | None, ...] = ()

    def to_json_line(self) -> str:
        """Serialise as one JSON line (optional keys only when set).

        Returns:
            Line without the newline.
        """
        line_object: dict = {"book_key": self.book_key, "language": self.language_name,
                             "words": list(self.unit_texts), "labels": list(self.unit_labels)}
        if self.context_before:
            line_object["context_before"] = list(self.context_before)
        if self.context_after:
            line_object["context_after"] = list(self.context_after)
        if self.unit_features:
            line_object["features"] = {feature_name: list(column_values)
                                       for feature_name, column_values in self.unit_features.items()}
        if self.unit_crops:
            line_object["crops"] = list(self.unit_crops)
        return json.dumps(line_object, ensure_ascii=False)


def label_index(label_name: str) -> int:
    """Class index of a label (``IGNORE_INDEX`` for ``ignore``).

    Args:
        label_name: Label text.

    Returns:
        0, 1, 2 or ``IGNORE_INDEX``.

    Raises:
        KeyError: For an unknown label.
    """
    return IGNORE_INDEX if label_name == IGNORE_LABEL else LABEL_INDEXES[label_name]


def read_json_lines(file_path: Path) -> Iterator[dict]:
    """Decoded lines of a bounded JSONL file.

    Args:
        file_path: File to read.

    Yields:
        One object per non-empty line.

    Raises:
        ValueError: If the file is too large or a line is not an object with a bounded word list.
    """
    if file_path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"{file_path} is too large.")
    with file_path.open(encoding="utf-8") as input_file:
        for line_text in input_file:
            if not line_text.strip():
                continue
            line_object = json.loads(line_text)
            if not isinstance(line_object, dict) or not isinstance(line_object.get("words"), list) \
                    or len(line_object["words"]) > MAX_LINE_UNITS:
                raise ValueError(f"Malformed line in {file_path}.")
            yield line_object


def read_examples(file_path: Path) -> list[TokenExample]:
    """Labelled examples of a file (missing file: none).

    Args:
        file_path: ``synthetic_*.jsonl`` or ``aligned_*.jsonl``.

    Returns:
        Examples whose labels match their units.
    """
    if not file_path.exists():
        return []
    return [example_from_line(line_object) for line_object in read_json_lines(file_path)
            if isinstance(line_object.get("labels"), list) and len(line_object["labels"]) == len(line_object["words"])]


def string_tuple(line_object: dict, key_name: str) -> tuple[str, ...]:
    """Optional list of strings of a line.

    Args:
        line_object: Decoded line.
        key_name: Key to read.

    Returns:
        The strings (empty when absent).
    """
    return tuple(str(unit_text) for unit_text in line_object.get(key_name) or ())


def example_crops(line_object: dict, unit_count: int) -> tuple[str | None, ...]:
    """Per-unit crop paths of a line (rejected when unsafe).

    Args:
        line_object: Decoded line.
        unit_count: Units of the example.

    Returns:
        Paths or None per unit (empty when absent or malformed).
    """
    crop_entries = line_object.get("crops")
    if not isinstance(crop_entries, list) or len(crop_entries) != unit_count:
        return ()
    if any(crop_path is not None and (not isinstance(crop_path, str) or crop_path.startswith("/")
                                      or ".." in crop_path.split("/")) for crop_path in crop_entries):
        return ()
    return tuple(crop_entries)


def example_from_line(line_object: dict) -> TokenExample:
    """Decode one labelled line.

    Args:
        line_object: Decoded line with ``words`` and ``labels``.

    Returns:
        The example.

    Raises:
        ValueError: If the features are malformed.
    """
    unit_texts = string_tuple(line_object, "words")
    return TokenExample(str(line_object.get("book_key", "")), str(line_object.get("language", "")), unit_texts,
                        string_tuple(line_object, "labels"), string_tuple(line_object, "context_before"),
                        string_tuple(line_object, "context_after"),
                        validate_feature_columns(line_object.get("features"), len(unit_texts)),
                        example_crops(line_object, len(unit_texts)))


def write_examples(file_path: Path, token_examples: list[TokenExample]) -> None:
    """Write examples as JSON lines.

    Args:
        file_path: Output file (overwritten).
        token_examples: Examples.

    Returns:
        None.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as output_file:
        for token_example in token_examples:
            output_file.write(token_example.to_json_line() + "\n")
