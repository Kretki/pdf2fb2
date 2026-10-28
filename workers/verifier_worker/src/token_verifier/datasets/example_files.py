"""Labelled examples on disk (JSON lines) and the label scheme.

A line is ``{"book_key", "language", "words": [...], "labels": [...]}`` with labels ``correct``, ``misread``,
``inserted`` or ``ignore`` (markup tokens: no loss, no metric). Reference chunk files have no labels.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

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
    """

    book_key: str
    language_name: str
    unit_texts: tuple[str, ...]
    unit_labels: tuple[str, ...]

    def to_json_line(self) -> str:
        """Serialise as one JSON line.

        Returns:
            Line without the newline.
        """
        return json.dumps({"book_key": self.book_key, "language": self.language_name, "words": list(self.unit_texts),
                           "labels": list(self.unit_labels)}, ensure_ascii=False)


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
    return [TokenExample(str(line_object.get("book_key", "")), str(line_object.get("language", "")),
                         tuple(str(unit_text) for unit_text in line_object["words"]),
                         tuple(str(label_name) for label_name in line_object["labels"]))
            for line_object in read_json_lines(file_path)
            if isinstance(line_object.get("labels"), list) and len(line_object["labels"]) == len(line_object["words"])]


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
