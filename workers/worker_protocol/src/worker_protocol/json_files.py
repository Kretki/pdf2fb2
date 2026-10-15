"""Bounded JSON / JSON-lines file I/O used for job and result files."""

import json
import os
from pathlib import Path

MAX_DOCUMENT_BYTES = 512 * 1024 * 1024


def read_json_document(file_path: Path, max_bytes: int = MAX_DOCUMENT_BYTES) -> dict:
    """Read a JSON object from a file, refusing oversized files.

    Usage:
        Call to load a job file, a layout result or the OCR cache.

    Args:
        file_path: Path of the UTF-8 JSON file.
        max_bytes: Upper bound on the file size in bytes (default 512 MiB).

    Returns:
        The decoded top-level JSON object.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the file is too large, is not valid JSON or is not an object.
    """
    if file_path.stat().st_size > max_bytes:
        raise ValueError(f"{file_path} is larger than {max_bytes} bytes.")
    decoded_document = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(decoded_document, dict):
        raise ValueError(f"{file_path} must contain a JSON object.")
    return decoded_document


def write_json_document(file_path: Path, json_document: dict) -> None:
    """Write a JSON object atomically (temporary file + rename).

    Usage:
        Call to write job files, result files and caches so that a crash never
        leaves a half-written file behind.

    Args:
        file_path: Destination path; its parent directory must exist.
        json_document: JSON-serialisable object to write.

    Returns:
        None. Creates or replaces ``file_path``.

    Raises:
        OSError: If the file cannot be written.
        TypeError: If ``json_document`` is not JSON-serialisable.
    """
    temporary_path = file_path.with_name(file_path.name + ".tmp")
    temporary_path.write_text(json.dumps(json_document, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temporary_path, file_path)


def append_json_line(file_path: Path, json_record: dict) -> None:
    """Append one JSON object as a line to a JSON-lines file.

    Usage:
        Call from a worker after every finished item so partial progress
        survives a crash.

    Args:
        file_path: JSON-lines file; created when missing.
        json_record: JSON-serialisable object written on one line.

    Returns:
        None. Appends to ``file_path`` and flushes it.

    Raises:
        OSError: If the file cannot be written.
    """
    with file_path.open("a", encoding="utf-8") as line_stream:
        line_stream.write(json.dumps(json_record, ensure_ascii=False) + "\n")
        line_stream.flush()


def read_json_lines(file_path: Path, max_bytes: int = MAX_DOCUMENT_BYTES) -> list[dict]:
    """Read all well-formed JSON objects from a JSON-lines file.

    Usage:
        Call to collect worker results; a truncated last line (worker crash) is skipped.

    Args:
        file_path: JSON-lines file to read; a missing file yields an empty list.
        max_bytes: Upper bound on the file size in bytes.

    Returns:
        List of decoded JSON objects in file order.

    Raises:
        ValueError: If the file is larger than ``max_bytes``.
    """
    if not file_path.exists():
        return []
    if file_path.stat().st_size > max_bytes:
        raise ValueError(f"{file_path} is larger than {max_bytes} bytes.")
    decoded_records = []
    for line_text in file_path.read_text(encoding="utf-8").splitlines():
        try:
            decoded_line = json.loads(line_text)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded_line, dict):
            decoded_records.append(decoded_line)
    return decoded_records
