"""Running-head check before OCR: a ``header`` far from where the book prints its running heads is a heading.

The layout model labels a chapter title set lower on an opening page (e.g. "LIFE OF JOAQUIN MURIETA" a fifth of
the way down) as ``header``. Running heads are never OCR'd and never reach the FB2, so such a title would be lost.
The book's running-head band is the median position of its ``header`` segments per page group (same page
parity, see ``residual_text.collect_margin_bands``); a ``header`` whose centre lies more than
``MISPLACED_TOLERANCE_RATIO`` of the page height outside that band becomes ``paragraph_title`` (the original
label is kept in ``original_label``). Groups with too few running heads to know their band are left alone.
"""

from pdf2fb2.segments.residual_text import HEADER_LABEL, collect_margin_bands

HEADING_LABEL = "paragraph_title"
MISPLACED_TOLERANCE_RATIO = 0.03  # fraction of the page height around the running-head band


def is_misplaced_running_head(page_segment: dict, page_height: float,
                              header_band: tuple[float, float] | None) -> bool:
    """Tell whether a ``header`` segment lies away from the running-head band.

    Args:
        page_segment: Segment of the page.
        page_height: Page image height in pixels.
        header_band: (top, bottom) of the band as fractions of the page height, or None when unknown.

    Returns:
        True for ``header`` segments whose vertical centre is more than ``MISPLACED_TOLERANCE_RATIO`` outside
        a known band.
    """
    if page_segment["label"] != HEADER_LABEL or header_band is None or page_height <= 0:
        return False
    center_ratio = (page_segment["bbox"][1] + page_segment["bbox"][3]) / 2 / page_height
    return not header_band[0] - MISPLACED_TOLERANCE_RATIO <= center_ratio <= header_band[1] + MISPLACED_TOLERANCE_RATIO


def relabel_misplaced_running_heads(page_records: list[dict]) -> int:
    """Turn the ``header`` segments that are not running heads into headings.

    Usage:
        Call on the final segmented page records, before OCR (running heads are not OCR'd).

    Args:
        page_records: Segmented page records (mutated: labels rewritten, ``original_label`` kept).

    Returns:
        Number of relabelled segments.
    """
    bands_by_parity = collect_margin_bands(page_records)
    relabelled_count = 0
    for page_record in page_records:
        margin_bands = bands_by_parity.get(page_record["page"] % 2)
        header_band = margin_bands.header_band if margin_bands is not None else None
        for page_segment in page_record["segments"]:
            if is_misplaced_running_head(page_segment, float(page_record["size"][1]), header_band):
                page_segment["original_label"] = page_segment["label"]
                page_segment["label"] = HEADING_LABEL
                relabelled_count += 1
    return relabelled_count
