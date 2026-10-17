"""Unit tests of the frame check, the font statistics, the glyph measurement and the picture recovery."""

from pathlib import Path

import cv2
import numpy as np

from pdf2fb2.pipeline.handwriting_check import HandwritingCheck
from pdf2fb2.pipeline.picture_recovery import FONT_REPORT_FILE, PICTURE_RESCAN_DIRECTORY, PictureRecovery
from pdf2fb2.segments.font_statistics import (
    GLYPH_HEIGHT_FIELD,
    FontStatistics,
    OffSizeRegion,
    PageGlyphs,
    collect_font_statistics,
    find_off_size_regions,
)
from pdf2fb2.segments.frame_statistics import find_frame_outliers
from pdf2fb2.segments.glyph_measurement import read_ink_mask, segment_glyph_height, uncovered_glyph_boxes
from pdf2fb2.segments.label_roles import HANDWRITTEN_LABEL, IMAGE_LABEL
from pdf2fb2.segments.picture_selection import RECOVERED_FIELD, choose_picture

PAGE_WIDTH = 1000
PAGE_HEIGHT = 1500
TYPICAL_HEIGHT = 14.0
USUAL_FONT = FontStatistics(TYPICAL_HEIGHT, 12.0, 15.0, 20)


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


class SilentInspector:
    """Region inspector that never finds reader marks.

    Usage:
        Injected into ``HandwritingCheck`` instead of the reader-mark worker.
    """

    def find_marked_regions(self, isolated_regions: list, work_directory: Path,
                            stage_progress: SilentProgress) -> set[int]:
        """Report no marks.

        Args:
            isolated_regions: Ignored.
            work_directory: Ignored.
            stage_progress: Ignored.

        Returns:
            Empty set.
        """
        return set()


class FixedGlyphMeter:
    """Glyph meter returning prepared measurements by page number.

    Usage:
        Injected into ``PictureRecovery`` instead of the OpenCV meter.
    """

    def __init__(self, measurements_by_page: dict[int, PageGlyphs]) -> None:
        """Store the measurements.

        Args:
            measurements_by_page: Mapping page number -> measurements aligned with its segments.
        """
        self._measurements_by_page = measurements_by_page

    def measure_page(self, page_record: dict) -> PageGlyphs:
        """Return the prepared measurements.

        Args:
            page_record: Page record.

        Returns:
            Measurements of the page.
        """
        return self._measurements_by_page[page_record["page"]]


class FixedProposer:
    """Box proposer returning the same boxes for every page.

    Usage:
        Injected into ``PictureRecovery`` instead of the low-threshold layout stage.
    """

    def __init__(self, proposed_boxes: list[dict]) -> None:
        """Store the proposals.

        Args:
            proposed_boxes: Boxes returned for every page.
        """
        self._proposed_boxes = proposed_boxes
        self.proposed_pages: list[Path] = []

    def propose_boxes(self, page_images: list[Path], work_directory: Path,
                      stage_progress: SilentProgress) -> list[list[dict]]:
        """Record the pages and return the fixed proposals.

        Args:
            page_images: Staged page images.
            work_directory: Ignored.
            stage_progress: Ignored.

        Returns:
            The proposals, once per page.
        """
        self.proposed_pages = list(page_images)
        return [list(self._proposed_boxes) for _ in page_images]


def make_segment(segment_label: str, bounding_box: list[float], segment_score: float = 0.9) -> dict:
    """Segment dictionary as produced by the layout stage.

    Args:
        segment_label: Layout label.
        bounding_box: [x1, y1, x2, y2] in pixels.
        segment_score: Layout score.

    Returns:
        Segment.
    """
    return {"label": segment_label, "bbox": bounding_box, "score": segment_score, "order": 0}


def make_page(page_number: int, page_segments: list[dict], image_path: str = "unused.png") -> dict:
    """Page record of the test page size.

    Args:
        page_number: 1-based page number.
        page_segments: Segments of the page.
        image_path: Page image path.

    Returns:
        Page record.
    """
    return {"page": page_number, "image": image_path, "size": [PAGE_WIDTH, PAGE_HEIGHT], "segments": page_segments}


def make_framed_book(page_count: int) -> list[dict]:
    """Odd pages with a running head, a paragraph and a page number at the same heights.

    Args:
        page_count: Number of pages (page numbers 1, 3, 5, ...).

    Returns:
        Page records.
    """
    return [make_page(2 * page_index + 1, [make_segment("header", [500, 60, 900, 90]),
                                           make_segment("text", [100, 120, 900, 1200]),
                                           make_segment("number", [480, 1250, 520, 1280])])
            for page_index in range(page_count)]


def test_regular_frame_has_no_outliers() -> None:
    """Pages with the same frame give no frame outliers."""
    assert find_frame_outliers(make_framed_book(10)) == []


def test_note_above_running_head_is_relabelled() -> None:
    """Text written above the running head is handwritten even when the mark model finds nothing."""
    page_records = make_framed_book(10)
    note_segment = make_segment("doc_title", [300, 0, 600, 47])
    page_records[3]["segments"].insert(0, note_segment)
    marked_count = HandwritingCheck(SilentInspector()).mark_handwritten_segments(
        page_records, Path("unused"), SilentProgress())
    assert marked_count == 1
    assert note_segment["label"] == HANDWRITTEN_LABEL and note_segment["original_label"] == "doc_title"


def test_font_statistics_use_plain_paragraphs() -> None:
    """Headings do not shift the usual font size."""
    page_records = [make_page(1, [{**make_segment("text", [0, 0, 10, 10]), GLYPH_HEIGHT_FIELD: 14.0}
                                  for _ in range(6)] + [{**make_segment("paragraph_title", [0, 0, 10, 10]),
                                                         GLYPH_HEIGHT_FIELD: 40.0}])]
    font_statistics = collect_font_statistics(page_records)
    assert font_statistics is not None and font_statistics.typical_height == 14.0


def test_large_lettering_outside_frame_is_ignored() -> None:
    """Large lettering above the printed frame is a reader note, not a picture."""
    page_records = make_framed_book(10)
    lettering_glyphs = tuple((50 * glyph_index + 100, 2, 50 * glyph_index + 140, 52) for glyph_index in range(10))
    page_measurements = [PageGlyphs((None, None, None), ()) for _ in page_records]
    page_measurements[2] = PageGlyphs((None, None, None), lettering_glyphs)
    assert find_off_size_regions(page_records, page_measurements, USUAL_FONT) == []
    inner_glyphs = tuple((glyph_box[0], glyph_box[1] + 300, glyph_box[2], glyph_box[3] + 300)
                         for glyph_box in lettering_glyphs)
    page_measurements[2] = PageGlyphs((None, None, None), inner_glyphs)
    assert [off_size_region.region_reason for off_size_region in
            find_off_size_regions(page_records, page_measurements, USUAL_FONT)] == ["large lettering"]


def poster_page() -> dict:
    """Page with an undetected poster, its small print taken for text, and a caption.

    Returns:
        Page record with glyph heights stored.
    """
    return make_page(34, [{**make_segment("text", [97, 835, 803, 914]), GLYPH_HEIGHT_FIELD: 11.0},
                          {**make_segment("text", [97, 932, 804, 1036]), GLYPH_HEIGHT_FIELD: 14.0}])


def test_picture_takes_in_off_font_text_but_not_caption() -> None:
    """The poster picture grows over its small print and stops at the caption."""
    page_record = poster_page()
    off_size_region = OffSizeRegion(page_record, [102, 55, 800, 792], "large lettering", 0.0)
    picture_choice = choose_picture(off_size_region, [{"label": "header", "bbox": [93, 34, 803, 808], "score": 0.3},
                                                      {"label": "image", "bbox": [93, 34, 803, 808], "score": 0.16}],
                                    USUAL_FONT)
    assert picture_choice is not None
    assert picture_choice.picture_box == [93.0, 34.0, 803.0, 914.0]
    assert picture_choice.absorbed_indexes == frozenset({0})


def test_text_more_probable_than_picture_is_kept() -> None:
    """A large-type title stays text when its own score beats the picture proposal."""
    page_record = make_page(5, [{**make_segment("text", [33, 274, 746, 372], 0.6), GLYPH_HEIGHT_FIELD: 51.0}])
    off_size_region = OffSizeRegion(page_record, [33, 274, 746, 372], "off-size text", 0.6)
    assert choose_picture(off_size_region, [{"label": "image", "bbox": [33, 274, 746, 372], "score": 0.09}],
                          USUAL_FONT) is None


def test_recovery_replaces_poster_text(tmp_path: Path) -> None:
    """The recovery segments the poster page again, replaces its small print and writes the report."""
    page_records = [make_page(page_number, [make_segment("text", [100, 100, 900, 1200])],
                              str(tmp_path / f"page_{page_number:04d}.png")) for page_number in range(1, 31)]
    page_records.append(make_page(34, poster_page()["segments"], str(tmp_path / "page_0034.png")))
    for page_record in page_records:
        Path(page_record["image"]).write_bytes(b"png")
    measurements = {page_number: PageGlyphs((TYPICAL_HEIGHT,), ()) for page_number in range(1, 31)}
    poster_glyphs = tuple((60 * glyph_index + 100, 100, 60 * glyph_index + 150, 200) for glyph_index in range(10))
    measurements[34] = PageGlyphs((11.0, 14.0), poster_glyphs)
    box_proposer = FixedProposer([{"label": "image", "bbox": [93, 34, 803, 808], "score": 0.16}])
    recovered_count = PictureRecovery(FixedGlyphMeter(measurements), box_proposer).recover_pictures(
        page_records, tmp_path, SilentProgress())
    assert recovered_count == 1
    assert [page_image.name for page_image in box_proposer.proposed_pages] == ["page_0034.png"]
    assert [(page_segment["label"], page_segment["order"]) for page_segment in page_records[-1]["segments"]] == [
        (IMAGE_LABEL, 0), ("text", 1)]
    assert page_records[-1]["segments"][0][RECOVERED_FIELD] == ["text"]
    assert (tmp_path / PICTURE_RESCAN_DIRECTORY / FONT_REPORT_FILE).is_file()


def test_glyph_heights_follow_font_size(tmp_path: Path) -> None:
    """OpenCV measures larger glyphs for larger type and finds lettering outside the segments."""
    page_image = np.full((600, 800), 255, dtype=np.uint8)
    cv2.putText(page_image, "reading small words", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.putText(page_image, "LARGE TYPE", (40, 400), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 0, 6)
    image_path = tmp_path / "page.png"
    cv2.imwrite(str(image_path), page_image)
    ink_mask = read_ink_mask(image_path)
    small_height = segment_glyph_height(ink_mask, [30, 70, 400, 115])
    assert small_height is not None and small_height < 25
    large_glyphs = uncovered_glyph_boxes(ink_mask, [[30, 70, 400, 115]])
    assert len(large_glyphs) >= 6  # some capitals touch each other
    assert min(glyph_box[3] - glyph_box[1] for glyph_box in large_glyphs) > 2.5 * small_height
