"""Validators for fields read from job files.

Every validator raises ``ValueError`` with a descriptive message, so a malformed
job file is rejected before any model is loaded.
"""

import math
import re
from pathlib import Path
from urllib.parse import urlparse

DEVICE_NAME_PATTERN = re.compile(r"^(cpu|gpu|cuda)(:\d{1,2})?$")
ALLOWED_URL_SCHEMES = frozenset({"http", "https"})
MAX_STRING_LENGTH = 4096


def require_string(job_payload: dict, field_name: str, max_length: int = MAX_STRING_LENGTH) -> str:
    """Return a mandatory non-empty string field.

    Usage:
        Call while decoding a job file.

    Args:
        job_payload: Decoded JSON object.
        field_name: Key to read.
        max_length: Maximum allowed length in characters.

    Returns:
        The string value.

    Raises:
        ValueError: If the field is missing, not a string, empty or too long.
    """
    field_value = job_payload.get(field_name)
    if not isinstance(field_value, str) or not field_value or len(field_value) > max_length:
        raise ValueError(f"Field {field_name!r} must be a non-empty string of at most {max_length} characters.")
    return field_value


def optional_string(job_payload: dict, field_name: str, max_length: int = MAX_STRING_LENGTH) -> str | None:
    """Return an optional string field, or None when it is absent or null.

    Usage:
        Call while decoding a job file for fields that are backend-specific.

    Args:
        job_payload: Decoded JSON object.
        field_name: Key to read.
        max_length: Maximum allowed length in characters.

    Returns:
        The string value, or None.

    Raises:
        ValueError: If the field is present but not a valid string.
    """
    if job_payload.get(field_name) is None:
        return None
    return require_string(job_payload, field_name, max_length)


def require_integer(job_payload: dict, field_name: str, lower_bound: int, upper_bound: int) -> int:
    """Return an integer field constrained to an inclusive range.

    Usage:
        Call while decoding a job file.

    Args:
        job_payload: Decoded JSON object.
        field_name: Key to read.
        lower_bound: Smallest allowed value.
        upper_bound: Largest allowed value.

    Returns:
        The integer value.

    Raises:
        ValueError: If the field is missing, not an integer or out of range.
    """
    field_value = job_payload.get(field_name)
    if isinstance(field_value, bool) or not isinstance(field_value, int):
        raise ValueError(f"Field {field_name!r} must be an integer.")
    if not lower_bound <= field_value <= upper_bound:
        raise ValueError(f"Field {field_name!r} must be within {lower_bound}..{upper_bound}.")
    return field_value


def require_number(job_payload: dict, field_name: str, lower_bound: float, upper_bound: float) -> float:
    """Return a finite numeric field constrained to an inclusive range.

    Usage:
        Call while decoding a job file.

    Args:
        job_payload: Decoded JSON object.
        field_name: Key to read.
        lower_bound: Smallest allowed value.
        upper_bound: Largest allowed value.

    Returns:
        The value as float.

    Raises:
        ValueError: If the field is missing, not a finite number or out of range.
    """
    field_value = job_payload.get(field_name)
    if isinstance(field_value, bool) or not isinstance(field_value, (int, float)) or not math.isfinite(field_value):
        raise ValueError(f"Field {field_name!r} must be a finite number.")
    if not lower_bound <= field_value <= upper_bound:
        raise ValueError(f"Field {field_name!r} must be within {lower_bound}..{upper_bound}.")
    return float(field_value)


def require_list(job_payload: dict, field_name: str, max_items: int) -> list:
    """Return a list field with a bounded number of entries.

    Usage:
        Call while decoding a job file.

    Args:
        job_payload: Decoded JSON object.
        field_name: Key to read.
        max_items: Maximum number of list entries.

    Returns:
        The list value (may be empty).

    Raises:
        ValueError: If the field is not a list or is too long.
    """
    field_value = job_payload.get(field_name)
    if not isinstance(field_value, list) or len(field_value) > max_items:
        raise ValueError(f"Field {field_name!r} must be a list of at most {max_items} entries.")
    return field_value


def require_absolute_directory(directory_text: str, field_name: str) -> Path:
    """Convert a string into an existing absolute directory path.

    Usage:
        Call for the working directory and model directories in a job file.

    Args:
        directory_text: Path string from the job file.
        field_name: Field name, used in the error message.

    Returns:
        The directory as a Path.

    Raises:
        ValueError: If the path is relative or is not an existing directory.
    """
    directory_path = Path(directory_text)
    if not directory_path.is_absolute() or not directory_path.is_dir():
        raise ValueError(f"Field {field_name!r} must be an existing absolute directory: {directory_text!r}")
    return directory_path


def validate_device_name(device_name: str) -> str:
    """Check a compute device name such as ``cpu``, ``gpu:0`` or ``cuda:1``.

    Usage:
        Call for device names coming from the command line or a job file.

    Args:
        device_name: Device string.

    Returns:
        The unchanged device string.

    Raises:
        ValueError: If the string does not match ``(cpu|gpu|cuda)(:N)?``.
    """
    if not DEVICE_NAME_PATTERN.fullmatch(device_name):
        raise ValueError(f"Unsupported device name {device_name!r}; use cpu, gpu:N or cuda:N.")
    return device_name


def validate_http_url(url_text: str) -> str:
    """Check that a URL uses http/https and names a host.

    Usage:
        Call before sending any request to an OCR server URL.

    Args:
        url_text: Base URL such as ``http://localhost:8000/v1``.

    Returns:
        The unchanged URL string.

    Raises:
        ValueError: If the scheme is not http/https or the host is missing.
    """
    parsed_url = urlparse(url_text)
    if parsed_url.scheme not in ALLOWED_URL_SCHEMES or not parsed_url.netloc:
        raise ValueError(f"URL must be http(s)://host[:port]/...: {url_text!r}")
    return url_text
