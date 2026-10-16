"""Progress events that workers print on stdout and the orchestrator reads back.

An event is one stdout line ``@@pdf2fb2-progress {"event": "...", ...}``. Every
other worker output line is ordinary log text.
"""

import json
import sys

PROGRESS_PREFIX = "@@pdf2fb2-progress "
MAX_EVENT_LINE_LENGTH = 4096
MODEL_LOADING_EVENT = "model_loading"
LAYOUT_PAGE_EVENT = "layout_page"
OCR_PROGRESS_EVENT = "ocr_progress"
MARKS_PAGE_EVENT = "marks_page"
STYLE_PAGE_EVENT = "style_page"


def emit_progress_event(event_name: str, **event_fields: int | str) -> None:
    """Print one progress event line and flush it immediately.

    Usage:
        Called by workers at model loading, before each page and after each crop.

    Args:
        event_name: One of the ``*_EVENT`` constants.
        **event_fields: Small integer or string fields (e.g. ``page=3, total=40``).

    Returns:
        None. Writes one line to ``sys.stdout``.
    """
    event_payload = {"event": event_name, **event_fields}
    sys.stdout.write(PROGRESS_PREFIX + json.dumps(event_payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def parse_progress_line(output_line: str) -> dict | None:
    """Decode a worker output line if it is a progress event.

    Usage:
        Called by the orchestrator for every line a worker prints.

    Args:
        output_line: One line of worker output, with or without the trailing newline.

    Returns:
        Event dict with a string ``event`` field, or None for ordinary log lines
        and for malformed or oversized event lines.
    """
    if not output_line.startswith(PROGRESS_PREFIX) or len(output_line) > MAX_EVENT_LINE_LENGTH:
        return None
    try:
        event_payload = json.loads(output_line[len(PROGRESS_PREFIX):])
    except ValueError:
        return None
    if not isinstance(event_payload, dict) or not isinstance(event_payload.get("event"), str):
        return None
    return event_payload


def read_counter_field(event_payload: dict, field_name: str) -> int | None:
    """Read a non-negative integer field of an event.

    Args:
        event_payload: Event from ``parse_progress_line``.
        field_name: Field to read, e.g. ``page`` or ``total``.

    Returns:
        The value, or None if it is missing, not an int (bools excluded) or negative.
    """
    field_value = event_payload.get(field_name)
    if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value < 0:
        return None
    return field_value
