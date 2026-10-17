"""Merge a rescanned (cleaned) page with the picture and handwritten segments of the first pass."""

from pdf2fb2.segments.label_roles import HANDWRITTEN_LABEL, PICTURE_LABELS
from pdf2fb2.segments.segment_geometry import box_area, box_intersection

# Segments taken from the first pass: pictures are blanked on the cleaned page, handwritten notes erased.
PRESERVED_LABELS = PICTURE_LABELS | {HANDWRITTEN_LABEL}
# A rescan segment mostly inside a preserved one is the same thing found again (e.g. a hand-lettered note the
# cleaner did not erase): it is dropped, so the first pass decision stands.
DUPLICATE_OVERLAP_RATIO = 0.5


def preserved_segments(page_record: dict) -> list[dict]:
    """Picture-like and handwritten segments of a page record (never OCR'd).

    Args:
        page_record: Page record.

    Returns:
        The preserved segments, unchanged.
    """
    return [page_segment for page_segment in page_record["segments"] if page_segment["label"] in PRESERVED_LABELS]


def insert_by_position(text_segments: list[dict], kept_segments: list[dict]) -> list[dict]:
    """Insert preserved segments into an ordered text flow before the first text segment that starts below them.

    Orders of the two layout passes are not comparable, so the rescan's text order is kept and every
    preserved segment is placed by its top edge.

    Args:
        text_segments: Rescan segments in reading order.
        kept_segments: First-pass picture and handwritten segments.

    Returns:
        New list of segment copies with ``order`` renumbered 0, 1, 2, ...
    """
    merged_segments = [dict(page_segment) for page_segment in text_segments]
    for inserted_segment in sorted(kept_segments, key=lambda kept_segment: kept_segment["bbox"][1]):
        insert_index = next((segment_index for segment_index, page_segment in enumerate(merged_segments)
                             if page_segment["label"] not in PRESERVED_LABELS
                             and page_segment["bbox"][1] >= inserted_segment["bbox"][1]), len(merged_segments))
        merged_segments.insert(insert_index, dict(inserted_segment))
    for order_index, page_segment in enumerate(merged_segments):
        page_segment["order"] = order_index
    return merged_segments


def is_duplicate_of_preserved(page_segment: dict, kept_segments: list[dict]) -> bool:
    """Tell whether a rescan segment lies mostly inside a preserved first-pass segment.

    Args:
        page_segment: Segment of the rescanned page.
        kept_segments: Preserved (picture and handwritten) segments of the first pass.

    Returns:
        True if at least ``DUPLICATE_OVERLAP_RATIO`` of the segment's area is inside one preserved segment.
    """
    segment_area = box_area(page_segment["bbox"]) or 1.0
    return any(box_intersection(page_segment["bbox"], kept_segment["bbox"]) >= DUPLICATE_OVERLAP_RATIO * segment_area
               for kept_segment in kept_segments)


def merge_rescanned_page(first_record: dict, rescanned_record: dict) -> dict:
    """Combine the text of the rescan with the pictures and handwritten segments of the first pass.

    Usage:
        Called for every page whose reader marks were removed. The cleaned page has its pictures
        blanked and its handwriting erased, so those come from the first pass and everything else
        from the rescan; rescan segments found again inside them are dropped.

    Args:
        first_record: First-pass page record (original page).
        rescanned_record: Record of the cleaned page (segmented again, not yet OCR'd).

    Returns:
        New page record with the original page number and size, the cleaned image and the merged,
        reading-ordered segments.
    """
    kept_segments = preserved_segments(first_record)
    text_segments = [page_segment for page_segment in rescanned_record["segments"]
                     if page_segment["label"] not in PRESERVED_LABELS
                     and not is_duplicate_of_preserved(page_segment, kept_segments)]
    merged_segments = insert_by_position(text_segments, kept_segments)
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
