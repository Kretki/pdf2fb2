"""argparse ``type=`` validators that bound every command-line input."""

import argparse
import re
from collections.abc import Callable
from pathlib import Path, PurePath

from worker_protocol.field_validation import validate_device_name, validate_http_url

LANGUAGE_CODE_PATTERN = re.compile(r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})?$")
MAX_TEXT_LENGTH = 512


def parse_safe_path(raw_text: str) -> Path:
    """Accept a filesystem path without ``..`` components or NUL characters.

    Args:
        raw_text: Path from the command line.

    Returns:
        The path.

    Raises:
        argparse.ArgumentTypeError: If the path is empty, contains NUL or ``..``.
    """
    if not raw_text or "\x00" in raw_text or ".." in PurePath(raw_text).parts:
        raise argparse.ArgumentTypeError(f"invalid path (empty, NUL or '..'): {raw_text!r}")
    return Path(raw_text)


def parse_existing_pdf(raw_text: str) -> Path:
    """Accept an existing ``.pdf`` file.

    Args:
        raw_text: Path from the command line.

    Returns:
        The PDF path.

    Raises:
        argparse.ArgumentTypeError: If the path is unsafe, missing or not a .pdf file.
    """
    pdf_path = parse_safe_path(raw_text)
    if pdf_path.suffix.lower() != ".pdf" or not pdf_path.is_file():
        raise argparse.ArgumentTypeError(f"not an existing .pdf file: {raw_text!r}")
    return pdf_path


def parse_existing_directory(raw_text: str) -> Path:
    """Accept an existing directory (used for model weights).

    Args:
        raw_text: Path from the command line.

    Returns:
        The directory path.

    Raises:
        argparse.ArgumentTypeError: If the path is unsafe or not a directory.
    """
    directory_path = parse_safe_path(raw_text)
    if not directory_path.is_dir():
        raise argparse.ArgumentTypeError(f"directory not found: {raw_text!r}")
    return directory_path


def bounded_integer(lower_bound: int, upper_bound: int) -> Callable[[str], int]:
    """Build an argparse type for integers in an inclusive range.

    Args:
        lower_bound: Smallest allowed value.
        upper_bound: Largest allowed value.

    Returns:
        Converter raising ``argparse.ArgumentTypeError`` for invalid input.
    """
    def parse_integer(raw_text: str) -> int:
        """Convert and range-check one integer.

        Args:
            raw_text: Value from the command line.

        Returns:
            The integer.

        Raises:
            argparse.ArgumentTypeError: If not an integer within range.
        """
        try:
            parsed_value = int(raw_text)
        except ValueError as conversion_error:
            raise argparse.ArgumentTypeError(f"not an integer: {raw_text!r}") from conversion_error
        if not lower_bound <= parsed_value <= upper_bound:
            raise argparse.ArgumentTypeError(f"must be within {lower_bound}..{upper_bound}")
        return parsed_value

    return parse_integer


def parse_probability(raw_text: str) -> float:
    """Accept a number in 0..1 (score threshold).

    Args:
        raw_text: Value from the command line.

    Returns:
        The number.

    Raises:
        argparse.ArgumentTypeError: If not a number within 0..1.
    """
    try:
        parsed_value = float(raw_text)
    except ValueError as conversion_error:
        raise argparse.ArgumentTypeError(f"not a number: {raw_text!r}") from conversion_error
    if not 0.0 <= parsed_value <= 1.0:
        raise argparse.ArgumentTypeError("must be within 0..1")
    return parsed_value


def parse_device(raw_text: str) -> str:
    """Accept ``cpu``, ``gpu[:N]`` or ``cuda[:N]``.

    Args:
        raw_text: Value from the command line.

    Returns:
        The device name.

    Raises:
        argparse.ArgumentTypeError: For any other value.
    """
    try:
        return validate_device_name(raw_text)
    except ValueError as device_error:
        raise argparse.ArgumentTypeError(str(device_error)) from device_error


def parse_server_url(raw_text: str) -> str:
    """Accept an http(s) URL for the OCR server.

    Args:
        raw_text: Value from the command line.

    Returns:
        The URL.

    Raises:
        argparse.ArgumentTypeError: For non-http(s) URLs.
    """
    try:
        return validate_http_url(raw_text)
    except ValueError as url_error:
        raise argparse.ArgumentTypeError(str(url_error)) from url_error


def parse_language_code(raw_text: str) -> str:
    """Accept a BCP-47-like language code such as ``en``, ``ru`` or ``pt-BR``.

    Args:
        raw_text: Value from the command line.

    Returns:
        The language code.

    Raises:
        argparse.ArgumentTypeError: For malformed codes.
    """
    if not LANGUAGE_CODE_PATTERN.fullmatch(raw_text):
        raise argparse.ArgumentTypeError(f"invalid language code: {raw_text!r}")
    return raw_text


def parse_short_text(raw_text: str) -> str:
    """Accept a non-empty text of at most 512 characters (title, author, genre, model name).

    Args:
        raw_text: Value from the command line.

    Returns:
        The stripped text.

    Raises:
        argparse.ArgumentTypeError: If empty or too long.
    """
    stripped_text = raw_text.strip()
    if not stripped_text or len(stripped_text) > MAX_TEXT_LENGTH:
        raise argparse.ArgumentTypeError(f"text must have 1..{MAX_TEXT_LENGTH} characters")
    return stripped_text
