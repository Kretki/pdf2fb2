"""Unit tests of the rescan merge, ellipsis normalisation, page numbering, sparse headings and residual text."""

from pathlib import Path

from pdf2fb2.fb2_output.text_markup import normalize_ellipsis
from pdf2fb2.pipeline.conversion_pipeline import OptionalStages, active_stage_names
from pdf2fb2.pipeline.layout_review import LayoutReview, LayoutRule
from pdf2fb2.pipeline.page_merging import merge_rescanned_page
from pdf2fb2.pipeline.residual_rescan import RESIDUAL_DIRECTORY, RESIDUAL_REPORT_FILE, ResidualTextRescan
from pdf2fb2.segments.font_statistics import GLYPH_HEIGHT_FIELD, FontStatistics, PageGlyphs
from pdf2fb2.segments.heading_statistics import mark_sparse_headings
from pdf2fb2.segments.label_roles import HANDWRITTEN_LABEL
from pdf2fb2.segments.number_parsing import read_page_number
from pdf2fb2.segments.page_numbering import NUMBER_VALUE_FIELD, verify_page_numbers
from pdf2fb2.segments.residual_segments import residual_segment
from pdf2fb2.segments.residual_text import MarginBands, ResidualBlock, find_residual_blocks

PAGE_WIDTH = 1000
PAGE_HEIGHT = 1500
TYPICAL_HEIGHT = 14.0
USUAL_FONT = FontStatistics(TYPICAL_HEIGHT, 12.0, 15.0, 40)
LONG_PARAGRAPH = "word " * 120


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


class FixedGlyphMeter:
    """Glyph meter returning prepared uncovered glyphs by page number, every segment at the usual height.

    Usage:
        Injected into ``ResidualTextRescan`` instead of the OpenCV meter.
    """

    def __init__(self, glyphs_by_page: dict[int, tuple]) -> None:
        """Store the glyphs.

        Args:
            glyphs_by_page: Mapping page number -> uncovered glyph boxes.
        """
        self._glyphs_by_page = glyphs_by_page

    def measure_page(self, page_record: dict) -> PageGlyphs:
        """Return the prepared measurements.

        Args:
            page_record: Page record.

        Returns:
            Usual glyph height for every segment and the prepared uncovered glyphs.
        """
        return PageGlyphs(tuple(TYPICAL_HEIGHT for _ in page_record["segments"]),
                          self._glyphs_by_page.get(page_record["page"], ()))


class FixedProposer:
    """Box proposer returning no boxes and recording the pages it was asked about.

    Usage:
        Injected into ``ResidualTextRescan`` instead of the layout stage.
    """

    def __init__(self) -> None:
        """Start with no recorded pages."""
        self.proposed_pages: list[str] = []

    def propose_boxes(self, page_images: list[Path], work_directory: Path,
                      stage_progress: SilentProgress) -> list[list[dict]]:
        """Record the pages and propose nothing.

        Args:
            page_images: Staged page images.
            work_directory: Ignored.
            stage_progress: Ignored.

        Returns:
            One empty list per page.
        """
        self.proposed_pages = [page_image.name for page_image in page_images]
        return [[] for _ in page_images]


def make_segment(segment_label: str, bounding_box: list[float], segment_text: str = "") -> dict:
    """Segment dictionary as produced by the layout and OCR stages.

    Args:
        segment_label: Layout label.
        bounding_box: [x1, y1, x2, y2] in pixels.
        segment_text: OCR text.

    Returns:
        Segment.
    """
    return {"label": segment_label, "bbox": bounding_box, "score": 0.9, "order": 0, "text": segment_text}


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


def body_page(page_number: int, number_segment: dict | None = None) -> dict:
    """Page with a running head, two full paragraphs and optionally a page-number segment.

    Args:
        page_number: 1-based page number.
        number_segment: Segment printed at the foot, or None.

    Returns:
        Page record.
    """
    page_segments = [make_segment("header", [550, 60, 900, 95]),
                     make_segment("text", [100, 120, 900, 600], LONG_PARAGRAPH),
                     make_segment("text", [100, 620, 900, 1200], LONG_PARAGRAPH)]
    return make_page(page_number, page_segments + ([number_segment] if number_segment else []))


def test_spaced_ellipsis_becomes_one_character() -> None:
    """Typeset and OCR'd ellipses become ``…``; a fourth dot stays a full stop."""
    assert normalize_ellipsis("more . . . I am") == "more … I am"
    assert normalize_ellipsis("climax.... Then") == "climax…. Then"
    assert normalize_ellipsis("e.g. a.b") == "e.g. a.b"


def test_rescan_drops_text_found_again_inside_handwriting() -> None:
    """A note the cleaner left on the page stays handwritten after the rescan."""
    first_record = make_page(17, [make_segment(HANDWRITTEN_LABEL, [292, 0, 574, 47]),
                                  make_segment("text", [50, 120, 750, 450])])
    rescanned_record = {**first_record, "segments": [make_segment("doc_title", [292, 0, 575, 47]),
                                                     make_segment("text", [50, 120, 750, 450])]}
    merged_labels = [page_segment["label"] for page_segment in merge_rescanned_page(first_record,
                                                                                    rescanned_record)["segments"]]
    assert merged_labels == [HANDWRITTEN_LABEL, "text"]


def test_page_number_readings_survive_ocr_noise() -> None:
    """Markup, spaces and letter/digit confusions still give the printed value."""
    assert [number_reading.number_value for number_reading in read_page_number("$5^{8}$")] == [58]
    assert [number_reading.number_value for number_reading in read_page_number("$ 8 1 $")] == [81]
    assert [number_reading.number_value for number_reading in read_page_number("IIO")] == [110]
    assert [number_reading.number_value for number_reading in read_page_number("x11")] == [12]
    prefixed_readings = read_page_number("A-12")
    assert [(number_reading.numbering_key, number_reading.number_value)
            for number_reading in prefixed_readings] == [(("arabic", "a-"), 12)]
    assert read_page_number("Introduction") == []


def test_numbering_runs_relabel_roman_and_arabic_numbers() -> None:
    """Roman front matter and arabic body numbers are verified; a chapter number at the top is not."""
    roman_numerals = ["xi", "xii", "xiii", "xiv", "xv"]
    page_records = [body_page(page_number, make_segment("text", [450, 1250, 520, 1280], roman_numerals[page_index]))
                    for page_index, page_number in enumerate(range(11, 16))]
    page_records += [body_page(page_number, make_segment("number", [480, 1250, 520, 1280], str(page_number - 50)))
                     for page_number in range(52, 56)]
    page_records.append(body_page(56, make_segment("paragraph_title", [450, 100, 520, 140], "III")))
    assert verify_page_numbers(page_records) == 5
    assert [page_record["segments"][-1]["label"] for page_record in page_records] == ["number"] * 9 + [
        "paragraph_title"]
    assert page_records[5]["segments"][-1][NUMBER_VALUE_FIELD] == 2


def test_title_page_lines_become_headings_and_lists_stay_text() -> None:
    """Short, centred, isolated lines of a near-empty page are headings; a tight list of entries is not."""
    page_records = [body_page(page_number) for page_number in range(1, 21)]
    page_records.append(make_page(51, [make_segment("text", [300, 260, 650, 400], "The Life and Adventures"),
                                       make_segment("text", [300, 1110, 650, 1150], "SAN FRANCISCO: 1854")]))
    page_records.append(make_page(53, [make_segment("text", [350, 300, 650, 340], "Murieta, by Nahl"),
                                       make_segment("text", [350, 360, 650, 400], "Poster of the exhibition"),
                                       make_segment("text", [350, 420, 650, 460], "Captain Harry Love")]))
    for page_record in page_records:
        for page_segment in page_record["segments"]:
            page_segment[GLYPH_HEIGHT_FIELD] = TYPICAL_HEIGHT
    assert mark_sparse_headings(page_records) == 2
    assert [page_segment["label"] for page_segment in page_records[-2]["segments"]] == ["paragraph_title"] * 2
    assert [page_segment["label"] for page_segment in page_records[-1]["segments"]] == ["text"] * 3


def text_line(top_edge: int, glyph_count: int, left_edge: int = 120) -> tuple:
    """Glyph boxes of one text line of the usual size.

    Args:
        top_edge: Top of the line in pixels.
        glyph_count: Number of glyphs.
        left_edge: Left edge of the first glyph.

    Returns:
        Glyph boxes 10 px wide with 4 px gaps.
    """
    return tuple((left_edge + 14 * glyph_index, top_edge, left_edge + 14 * glyph_index + 10,
                  top_edge + int(TYPICAL_HEIGHT)) for glyph_index in range(glyph_count))


def test_residual_glyphs_split_into_number_and_paragraph() -> None:
    """A short piece in the page-number band is a number, two full lines form a paragraph, dust is ignored."""
    margin_bands = MarginBands((0.03, 0.07), (0.82, 0.86), 0.5)
    glyph_boxes = list(text_line(1250, 3, 480) + text_line(700, 30) + text_line(736, 25) + text_line(900, 2))
    residual_blocks = find_residual_blocks(make_page(5, []), glyph_boxes, USUAL_FONT, margin_bands)
    assert [(residual_block.block_label, residual_block.line_count) for residual_block in residual_blocks] == [
        ("number", 1), ("text", 2)]


def test_heading_proposal_for_several_lines_is_text() -> None:
    """The layout model's title label is kept for one line only; without a proposal the block is used."""
    page_record = make_page(5, [make_segment("text", [100, 120, 900, 600])])
    verse_block = ResidualBlock([200, 830, 600, 900], "text", 50, 2)
    heading_proposal = {"label": "paragraph_title", "bbox": [190, 825, 610, 905], "score": 0.47}
    assert residual_segment(verse_block, [heading_proposal], page_record)["label"] == "text"
    single_line = ResidualBlock([200, 830, 600, 850], "text", 20, 1)
    assert residual_segment(single_line, [heading_proposal], page_record)["label"] == "paragraph_title"
    assert residual_segment(single_line, [], page_record)["bbox"] == [196, 826, 604, 854]


def test_rescan_adds_missed_paragraph_and_caption(tmp_path: Path) -> None:
    """A missed paragraph is segmented again; a line under a picture becomes its caption."""
    page_records = [body_page(page_number, make_segment("number", [480, 1250, 520, 1280], str(page_number)))
                    for page_number in range(1, 11)]
    page_records.append(make_page(11, [make_segment("image", [100, 100, 900, 1300])]))
    for page_record in page_records:
        page_record["image"] = str(tmp_path / f"page_{page_record['page']:04d}.png")
        Path(page_record["image"]).write_bytes(b"png")
    glyphs_by_page = {3: text_line(1205, 30) + text_line(1240, 30, 120), 11: text_line(1320, 20, 300)}
    box_proposer = FixedProposer()
    added_count = ResidualTextRescan(FixedGlyphMeter(glyphs_by_page), box_proposer).add_residual_text(
        page_records, tmp_path, SilentProgress())
    assert added_count == 2
    assert box_proposer.proposed_pages == ["page_0003.png"]
    assert [page_segment["label"] for page_segment in page_records[2]["segments"]] == [
        "header", "text", "text", "text", "number"]
    assert page_records[-1]["segments"][-1]["label"] == "figure_title"
    assert (tmp_path / RESIDUAL_DIRECTORY / RESIDUAL_REPORT_FILE).is_file()


def test_review_runs_rules_in_order() -> None:
    """The review sums the relabelled segments of its rules."""
    called_rules: list[str] = []
    review_rules = tuple(LayoutRule(rule_name, lambda page_records, rule_name=rule_name: called_rules.append(
        rule_name) or 2) for rule_name in ("first", "second"))
    assert LayoutReview(review_rules).review_layout([], SilentProgress()) == 4
    assert called_rules == ["first", "second"]


def test_stage_names_follow_execution_order() -> None:
    """Residual text runs before OCR, the layout review right after it."""
    stage_names = active_stage_names(False, OptionalStages(residual_finder=object(), layout_reviewer=object()))
    assert stage_names == ["rendering", "segmentation", "residual", "recognition", "review", "assembling"]
