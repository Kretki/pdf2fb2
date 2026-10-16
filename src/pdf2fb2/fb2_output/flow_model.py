"""Data model of the linear book flow built from OCR'd page records."""

from dataclasses import dataclass, field


@dataclass
class BookFlow:
    """Result of flow building.

    Attributes:
        flow_items: Flow items in reading order (see module docstring).
        footnotes: Notes ``{"id", "marker", "text", "linked"}``.
        book_title: Text of the first ``doc_title`` segment, if any.
    """

    flow_items: list[dict] = field(default_factory=list)
    footnotes: list[dict] = field(default_factory=list)
    book_title: str | None = None


def segment_text(page_segment: dict) -> str:
    """Return the stripped OCR text of a segment.

    Args:
        page_segment: Segment, possibly without ``text``.

    Returns:
        Stripped text (empty if none).
    """
    return (page_segment.get("text") or "").strip()
