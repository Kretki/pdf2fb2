"""Unit tests of the position-outlier statistics and the handwriting check of outlying segments."""

from pathlib import Path

from pdf2fb2.pipeline.handwriting_check import HandwritingCheck
from pdf2fb2.pipeline.page_merging import merge_rescanned_page
from pdf2fb2.pipeline.stage_contracts import IsolatedRegion
from pdf2fb2.segments.label_roles import HANDWRITTEN_LABEL, requires_ocr
from pdf2fb2.segments.position_statistics import find_position_outliers
from pdf2fb2.segments.segment_geometry import surrounding_boxes

PAGE_WIDTH = 1000
PAGE_HEIGHT = 1500


class SilentProgress:
    """Stage progress that records nothing.

    Usage:
        Passed wherever a ``StageProgress`` is needed in tests.
    """

    def advance_stage(self, completed_units: int, total_units: int, status_text: str) -> None:
        """Ignore a progress update.

        Args:
            completed_units: Units done.
            total_units: Units in the stage.
            status_text: Step description.
        """

    def report_message(self, message_text: str) -> None:
        """Ignore a message.

        Args:
            message_text: Message.
        """


class FixedInspector:
    """Region inspector that reports marks in the regions given at construction.

    Usage:
        Injected into ``HandwritingCheck`` instead of the reader-mark worker.
    """

    def __init__(self, marked_indexes: set[int]) -> None:
        """Store the answer.

        Args:
            marked_indexes: Region indexes to report as marked.
        """
        self.marked_indexes = marked_indexes
        self.seen_regions: list[IsolatedRegion] = []

    def find_marked_regions(self, isolated_regions: list[IsolatedRegion], work_directory: Path,
                            stage_progress: SilentProgress) -> set[int]:
        """Record the regions and return the fixed answer.

        Args:
            isolated_regions: Regions to inspect.
            work_directory: Ignored.
            stage_progress: Ignored.

        Returns:
            The indexes given at construction.
        """
        self.seen_regions = list(isolated_regions)
        return self.marked_indexes


def make_segment(segment_label: str, bounding_box: list[float]) -> dict:
    """Segment dictionary as produced by the layout stage.

    Args:
        segment_label: Layout label.
        bounding_box: [x1, y1, x2, y2] in pixels.

    Returns:
        Segment.
    """
    return {"label": segment_label, "bbox": bounding_box, "score": 0.9, "order": 0}


def make_book(page_count: int, note_page: int | None = None) -> list[dict]:
    """Odd pages with regular text and a heading; optionally a margin note on one page.

    Args:
        page_count: Number of pages (odd page numbers 1, 3, 5, ...).
        note_page: Page number that also gets a narrow text segment in the right margin, or None.

    Returns:
        Page records.
    """
    page_records = []
    for page_index in range(page_count):
        page_number = 2 * page_index + 1
        jitter_pixels = page_index % 3
        page_segments = [make_segment("paragraph_title", [400 + jitter_pixels, 100, 600 + jitter_pixels, 140]),
                         make_segment("text", [100 + jitter_pixels, 200, 900 - jitter_pixels, 600]),
                         make_segment("text", [100 - jitter_pixels, 650, 900 + jitter_pixels, 1200])]
        if page_number == note_page:
            page_segments.append(make_segment("text", [920, 300, 990, 500]))
        page_records.append({"page": page_number, "image": f"page_{page_number:04d}.png",
                             "size": [PAGE_WIDTH, PAGE_HEIGHT], "segments": page_segments})
    return page_records


def test_regular_layout_has_no_outliers() -> None:
    """Jitter of a few pixels stays inside the interval."""
    assert find_position_outliers(make_book(10)) == []


def test_margin_note_is_an_outlier() -> None:
    """Only the margin segment is flagged."""
    outlier_segments = find_position_outliers(make_book(10, note_page=5))
    assert [outlier_segment.page_segment["bbox"] for outlier_segment in outlier_segments] == [[920, 300, 990, 500]]


def test_small_groups_are_not_compared() -> None:
    """Fewer than five segments of a role give no statistics."""
    assert find_position_outliers(make_book(1, note_page=1)) == []


def test_marked_outlier_is_relabelled() -> None:
    """An outlier with reader marks becomes ``handwritten`` and is no longer OCR'd."""
    page_records = make_book(10, note_page=5)
    region_inspector = FixedInspector({0})
    marked_count = HandwritingCheck(region_inspector).mark_handwritten_segments(
        page_records, Path("unused"), SilentProgress())
    note_segment = page_records[2]["segments"][-1]
    assert marked_count == 1
    assert note_segment["label"] == HANDWRITTEN_LABEL and note_segment["original_label"] == "text"
    assert not requires_ocr(note_segment["label"])
    assert region_inspector.seen_regions[0].region_box == [912, 292, 998, 508]


def test_unmarked_outlier_is_kept() -> None:
    """An outlier without reader marks keeps its label."""
    page_records = make_book(10, note_page=5)
    HandwritingCheck(FixedInspector(set())).mark_handwritten_segments(page_records, Path("unused"), SilentProgress())
    assert page_records[2]["segments"][-1]["label"] == "text"


def test_surrounding_boxes_cover_rest_of_page() -> None:
    """The four boxes plus the region tile the page."""
    surrounding = surrounding_boxes([100, 200, 300, 400], (1000, 1500))
    covered_area = sum((right_edge - left_edge) * (bottom_edge - top_edge)
                       for left_edge, top_edge, right_edge, bottom_edge in surrounding)
    assert covered_area + 200 * 200 == 1000 * 1500


def test_rescan_keeps_handwritten_segments() -> None:
    """Merging a rescanned page keeps the first-pass handwritten segment."""
    first_record = make_book(1)[0]
    first_record["segments"].append(make_segment(HANDWRITTEN_LABEL, [920, 300, 990, 500]))
    rescanned_record = {**first_record, "image": "clean.png", "segments": [make_segment("text", [100, 200, 900, 600])]}
    merged_labels = [page_segment["label"] for page_segment in merge_rescanned_page(first_record,
                                                                                    rescanned_record)["segments"]]
    assert merged_labels == ["text", HANDWRITTEN_LABEL]
