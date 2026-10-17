"""Page-number check: short texts at the top or bottom of the page that continue a numbering sequence.

A printed page number grows by one from page to page, so ``page index - printed value`` stays constant along
one numbering. A book may have several numberings (roman front matter, arabic body, prefixed appendix pages,
restarts), each a run of its own: (kind, prefix, offset). A run seen on at least ``MINIMUM_RUN_PAGES`` pages is
a numbering of the book, and every candidate whose reading falls on such a run is a page number, whatever label
the layout model gave it. Chapter numbers, dates and stray digits do not continue any run and are left alone.
"""

from collections import defaultdict
from dataclasses import dataclass

from pdf2fb2.segments.label_roles import PAGE_NUMBER_LABELS, requires_ocr
from pdf2fb2.segments.number_parsing import NumberReading, read_page_number

PAGE_NUMBER_LABEL = "number"
MARGIN_BAND_RATIO = 0.2  # top and bottom share of the page height searched for page numbers
MINIMUM_RUN_PAGES = 3
NUMBER_VALUE_FIELD = "page_number_value"  # segment field: verified printed value
NUMBERING_FIELD = "numbering"  # segment field: kind (and prefix) of the numbering
NumberingRun = tuple[str, str, int]  # (kind, prefix, page index - value)


@dataclass(frozen=True)
class NumberCandidate:
    """A segment that may be a page number, with its readings.

    Attributes:
        page_record: Page record of the segment.
        page_segment: The segment.
        number_readings: Plausible values of its text.
    """

    page_record: dict
    page_segment: dict
    number_readings: tuple[NumberReading, ...]


def in_margin_band(page_segment: dict, page_height: float) -> bool:
    """Tell whether a segment lies in the top or bottom band of the page.

    Args:
        page_segment: Segment with ``bbox``.
        page_height: Page image height in pixels (> 0).

    Returns:
        True if the segment ends in the top ``MARGIN_BAND_RATIO`` or starts in the bottom one.
    """
    return (page_segment["bbox"][3] <= MARGIN_BAND_RATIO * page_height
            or page_segment["bbox"][1] >= (1 - MARGIN_BAND_RATIO) * page_height)


def is_number_candidate(page_segment: dict, page_height: float) -> bool:
    """Tell whether a segment may be a printed page number.

    Args:
        page_segment: OCR'd segment.
        page_height: Page image height in pixels.

    Returns:
        True for ``number`` segments and for OCR'd segments in the top or bottom band of the page.
    """
    if page_segment["label"] in PAGE_NUMBER_LABELS:
        return True
    return requires_ocr(page_segment["label"]) and page_height > 0 and in_margin_band(page_segment, page_height)


def collect_candidates(page_records: list[dict]) -> list[NumberCandidate]:
    """Page-number candidates of the book with at least one reading.

    Args:
        page_records: OCR'd page records.

    Returns:
        Candidates in page order.
    """
    number_candidates = []
    for page_record in page_records:
        page_height = float(page_record["size"][1])
        for page_segment in page_record["segments"]:
            number_readings = tuple(read_page_number(page_segment.get("text") or ""))
            if number_readings and is_number_candidate(page_segment, page_height):
                number_candidates.append(NumberCandidate(page_record, page_segment, number_readings))
    return number_candidates


def reading_run(page_record: dict, number_reading: NumberReading) -> NumberingRun:
    """Numbering run a reading belongs to.

    Args:
        page_record: Page record of the reading.
        number_reading: One reading of a candidate.

    Returns:
        (kind, prefix, page index - value).
    """
    return (*number_reading.numbering_key, page_record["page"] - number_reading.number_value)


def find_numbering_runs(number_candidates: list[NumberCandidate]) -> set[NumberingRun]:
    """Runs continued on at least ``MINIMUM_RUN_PAGES`` different pages.

    Args:
        number_candidates: Candidates of the whole book.

    Returns:
        The book's numberings.
    """
    run_pages: dict[NumberingRun, set[int]] = defaultdict(set)
    for number_candidate in number_candidates:
        for number_reading in number_candidate.number_readings:
            run_pages[reading_run(number_candidate.page_record, number_reading)].add(
                number_candidate.page_record["page"])
    return {numbering_run for numbering_run, page_numbers in run_pages.items()
            if len(page_numbers) >= MINIMUM_RUN_PAGES}


def mark_page_number(page_segment: dict, number_reading: NumberReading) -> None:
    """Label a segment as a verified page number.

    Args:
        page_segment: Segment (mutated: label ``number`` with ``original_label`` kept when it changes, value and
            numbering stored).
        number_reading: The reading that continues a numbering run.

    Returns:
        None.
    """
    if page_segment["label"] != PAGE_NUMBER_LABEL:
        page_segment["original_label"] = page_segment["label"]
        page_segment["label"] = PAGE_NUMBER_LABEL
    page_segment[NUMBER_VALUE_FIELD] = number_reading.number_value
    numbering_kind, numbering_prefix = number_reading.numbering_key
    page_segment[NUMBERING_FIELD] = numbering_prefix + numbering_kind


def verify_page_numbers(page_records: list[dict]) -> int:
    """Label as page numbers the candidates that continue a numbering of the book.

    Usage:
        Call on the OCR'd page records, before the book flow is built.

    Args:
        page_records: OCR'd page records (mutated: matching segments relabelled and annotated).

    Returns:
        Number of segments relabelled from another label to ``number``.
    """
    number_candidates = collect_candidates(page_records)
    numbering_runs = find_numbering_runs(number_candidates)
    relabelled_count = 0
    for number_candidate in number_candidates:
        fitting_reading = next((number_reading for number_reading in number_candidate.number_readings
                                if reading_run(number_candidate.page_record, number_reading) in numbering_runs), None)
        if fitting_reading is None:
            continue
        relabelled_count += number_candidate.page_segment["label"] != PAGE_NUMBER_LABEL
        mark_page_number(number_candidate.page_segment, fitting_reading)
    return relabelled_count
