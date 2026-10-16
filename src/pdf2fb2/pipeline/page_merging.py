"""Merge a rescanned (cleaned) page with the picture segments cut out in the first pass."""

from pdf2fb2.segments.label_roles import PICTURE_LABELS


def picture_segments(page_record: dict) -> list[dict]:
    """Picture-like segments of a page record (embedded as images, never OCR'd).

    Args:
        page_record: Page record.

    Returns:
        The picture segments, unchanged.
    """
    return [page_segment for page_segment in page_record["segments"] if page_segment["label"] in PICTURE_LABELS]


def insert_by_position(text_segments: list[dict], pictures: list[dict]) -> list[dict]:
    """Insert pictures into an ordered text flow before the first text segment that starts below them.

    Orders of the two layout passes are not comparable, so the rescan's text order is kept and every
    picture is placed by its top edge.

    Args:
        text_segments: Rescan segments in reading order.
        pictures: First-pass picture segments.

    Returns:
        New list of segment copies with ``order`` renumbered 0, 1, 2, ...
    """
    merged_segments = [dict(page_segment) for page_segment in text_segments]
    for picture_segment in sorted(pictures, key=lambda candidate_picture: candidate_picture["bbox"][1]):
        insert_index = next((segment_index for segment_index, page_segment in enumerate(merged_segments)
                             if page_segment["label"] not in PICTURE_LABELS
                             and page_segment["bbox"][1] >= picture_segment["bbox"][1]), len(merged_segments))
        merged_segments.insert(insert_index, dict(picture_segment))
    for order_index, page_segment in enumerate(merged_segments):
        page_segment["order"] = order_index
    return merged_segments


def merge_rescanned_page(first_record: dict, rescanned_record: dict) -> dict:
    """Combine the text of the rescan with the pictures of the first pass.

    Usage:
        Called for every page whose reader marks were removed. The cleaned page has its pictures
        blanked, so pictures come from the first pass and everything else from the rescan.

    Args:
        first_record: First-pass page record (original page).
        rescanned_record: Record of the cleaned page (segmented again, not yet OCR'd).

    Returns:
        New page record with the original page number and size, the cleaned image and the merged,
        reading-ordered segments.
    """
    text_segments = [page_segment for page_segment in rescanned_record["segments"]
                     if page_segment["label"] not in PICTURE_LABELS]
    merged_segments = insert_by_position(text_segments, picture_segments(first_record))
    return {**first_record, "image": rescanned_record["image"], "segments": merged_segments, "rescanned": True}


def merge_page_records(first_records: list[dict], rescanned_records: dict[int, dict]) -> list[dict]:
    """Final page list: rescanned pages merged, all other pages kept from the first pass.

    Args:
        first_records: First-pass records in page order.
        rescanned_records: Mapping page number -> rescanned record.

    Returns:
        Page records in page order.
    """
    return [merge_rescanned_page(page_record, rescanned_records[page_record["page"]])
            if page_record["page"] in rescanned_records else page_record
            for page_record in first_records]
