"""Turns residual-text blocks into page segments, preferring what the layout model proposes for them.

For a missed paragraph the page is segmented again; the most probable text-like proposal covering at least half
of the block, and lying mostly on free page area (at most 30% of it over existing segments), gives the segment
its box and label. Without such a proposal the block itself becomes a ``text`` segment, so text the layout model
keeps missing (e.g. a short centred quotation) is still read. A heading label is kept only for one-line blocks:
several lines of the usual font are a paragraph or a verse. Missed running heads and page numbers are added as
found.
"""

from pdf2fb2.segments.label_roles import (
    DOCUMENT_TITLE_LABEL,
    HEADING_LABELS,
    PAGE_NUMBER_LABELS,
    requires_ocr,
)
from pdf2fb2.segments.residual_text import TEXT_LABEL, ResidualBlock
from pdf2fb2.segments.segment_geometry import box_area, box_intersection, union_box

BLOCK_COVERAGE_RATIO = 0.5
MAXIMUM_OCCUPIED_RATIO = 0.3
BLOCK_PADDING = 4  # pixels around the glyphs of a block used as a segment
RESIDUAL_FIELD = "residual"  # segment field: added by the residual-text rescan
HEADING_LABEL = "paragraph_title"


def occupied_area(proposal_box: list[float], page_segments: list[dict]) -> float:
    """Area of a proposal that lies over existing segments.

    Args:
        proposal_box: [x1, y1, x2, y2] of the proposal.
        page_segments: Existing segments of the page.

    Returns:
        Sum of the intersections in square pixels.
    """
    return sum(box_intersection(proposal_box, page_segment["bbox"]) for page_segment in page_segments)


def is_fitting_proposal(proposed_box: dict, residual_block: ResidualBlock, page_segments: list[dict]) -> bool:
    """Tell whether a layout proposal describes a residual block.

    Args:
        proposed_box: Proposal ``{"label", "bbox", "score"}``.
        residual_block: Missed paragraph.
        page_segments: Existing segments of the page.

    Returns:
        True for OCR'd, non-page-number labels covering half of the block and mostly over free page area.
    """
    proposal_box = proposed_box["bbox"]
    return (requires_ocr(proposed_box["label"]) and proposed_box["label"] not in PAGE_NUMBER_LABELS
            and box_intersection(proposal_box, residual_block.block_box)
            >= BLOCK_COVERAGE_RATIO * (box_area(residual_block.block_box) or 1.0)
            and occupied_area(proposal_box, page_segments) <= MAXIMUM_OCCUPIED_RATIO * (box_area(proposal_box) or 1.0))


def proposal_label(proposed_label: str, residual_block: ResidualBlock) -> str:
    """Label of a segment built from a proposal.

    Args:
        proposed_label: Label of the layout proposal.
        residual_block: The block it covers.

    Returns:
        The proposal's label; headings and titles become ``paragraph_title`` for one-line blocks and ``text``
        for longer ones.
    """
    if proposed_label not in HEADING_LABELS and proposed_label != DOCUMENT_TITLE_LABEL:
        return proposed_label
    return HEADING_LABEL if residual_block.line_count == 1 else TEXT_LABEL


def padded_box(block_box: list[float], page_size: tuple[float, float]) -> list[float]:
    """Block box padded by ``BLOCK_PADDING`` and clamped to the page.

    Args:
        block_box: [x1, y1, x2, y2] of the glyphs.
        page_size: (width, height) of the page in pixels.

    Returns:
        The padded box.
    """
    return [max(0.0, block_box[0] - BLOCK_PADDING), max(0.0, block_box[1] - BLOCK_PADDING),
            min(page_size[0], block_box[2] + BLOCK_PADDING), min(page_size[1], block_box[3] + BLOCK_PADDING)]


def residual_segment(residual_block: ResidualBlock, proposed_boxes: list[dict], page_record: dict) -> dict:
    """Segment for one residual block.

    Usage:
        Call for every block of a page, with the layout proposals of that page (empty for band elements).

    Args:
        residual_block: Missed paragraph, running head or page number.
        proposed_boxes: Layout proposals of the page.
        page_record: Page record (its segments are not modified).

    Returns:
        New segment ``{"label", "bbox", "score", "order": None, "residual": True}``.
    """
    page_size = (float(page_record["size"][0]), float(page_record["size"][1]))
    block_box = padded_box(residual_block.block_box, page_size)
    fitting_proposals = [proposed_box for proposed_box in proposed_boxes if residual_block.block_label == TEXT_LABEL
                         and is_fitting_proposal(proposed_box, residual_block, page_record["segments"])]
    best_proposal = max(fitting_proposals, key=lambda proposed_box: proposed_box["score"], default=None)
    if best_proposal is None:
        return {"label": residual_block.block_label, "bbox": block_box, "score": 0.0, "order": None,
                RESIDUAL_FIELD: True}
    return {"label": proposal_label(best_proposal["label"], residual_block),
            "bbox": union_box([best_proposal["bbox"], block_box]), "score": best_proposal["score"], "order": None,
            RESIDUAL_FIELD: True}
