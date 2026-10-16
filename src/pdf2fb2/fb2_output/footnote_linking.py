"""Footnote parsing and linking of footnote markers in the text flow."""

import re

from pdf2fb2.fb2_output.text_markup import make_note_token

FOOTNOTE_PATTERN = re.compile(
    r"^\s*(?:\[(\d+)\]|<sup>(\w+)</sup>|\$?\^\{?(\w+)\}?\$?|([⁰¹²³⁴⁵⁶⁷⁸⁹]+)|(\d{1,3})(?=[\s.)])|([*†‡]+))"
    r"[.)]?\s*(.+)$", re.S)
FOOTNOTE_BODY_GROUP = 7
SUPERSCRIPT_TO_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
DIGITS_TO_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def parse_footnote_paragraph(paragraph_text: str) -> tuple[str, str] | None:
    """Split a footnote paragraph into marker and body.

    Args:
        paragraph_text: One paragraph of a footnote segment.

    Returns:
        ``(marker, body)`` with superscript digits normalised, or None when the
        paragraph does not start with a marker (a continuation line).
    """
    footnote_match = FOOTNOTE_PATTERN.match(paragraph_text)
    if footnote_match is None:
        return None
    marker_text = next(group_text for group_text in footnote_match.groups()[:-1] if group_text)
    return marker_text.translate(SUPERSCRIPT_TO_DIGITS), footnote_match.group(FOOTNOTE_BODY_GROUP)


def _marker_patterns(marker_text: str) -> list[str]:
    """Regular expressions that find a numeric footnote marker in body text.

    Args:
        marker_text: Marker consisting of digits only (safe to embed in a regex).

    Returns:
        Patterns for ``$^{n}$``, ``<sup>n</sup>``, ``^{n}``, ``^n``, superscript digits and ``[n]``.
    """
    return [rf"\$\^\{{?{marker_text}\}}?\$", rf"<sup>{marker_text}</sup>", rf"\^\{{{marker_text}\}}",
            rf"\^{marker_text}(?!\d)",
            re.escape(marker_text.translate(DIGITS_TO_SUPERSCRIPT)) + r"(?![⁰¹²³⁴⁵⁶⁷⁸⁹])",
            rf"\[{marker_text}\]"]


def link_note_marker(flow_items: list[dict], page_number: int, marker_text: str, note_identifier: str) -> bool:
    """Replace the last matching marker on the page with a note-link token.

    Usage:
        Call when a footnote is read, so the marker in the body text links to it.

    Args:
        flow_items: Book flow built so far (paragraph items are mutated).
        page_number: Page the footnote belongs to.
        marker_text: Footnote marker; only numeric markers are linked.
        note_identifier: FB2 id of the note section.

    Returns:
        True if a marker was found and replaced.
    """
    if not marker_text.isdigit():
        return False
    note_token = make_note_token(note_identifier, marker_text)
    for flow_item in reversed(flow_items):
        if flow_item["kind"] != "p" or page_number not in flow_item["pages"]:
            continue
        for marker_pattern in _marker_patterns(marker_text):
            replaced_text, replacement_count = re.subn(marker_pattern, note_token, flow_item["text"], count=1)
            if replacement_count:
                flow_item["text"] = replaced_text
                return True
    return False
