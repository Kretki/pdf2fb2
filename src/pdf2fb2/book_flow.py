"""Builds the linear book flow (pages, headings, paragraphs, pictures, tables, notes).

Flow item kinds:
    page    {"kind": "page", "page": int, "printed": str | None}
    heading {"kind": "heading", "level": int, "text": str}
    p       {"kind": "p", "text": str, "style": "p" | "cite", "list": bool, "pages": set[int]}
    caption {"kind": "caption", "text": str}
    image   {"kind": "image", "id": str}
    table   {"kind": "table", "rows": list}
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import pymupdf

from pdf2fb2.binary_store import BinaryStore
from pdf2fb2.footnote_linking import link_note_marker, parse_footnote_paragraph
from pdf2fb2.label_roles import (
    ABSTRACT_LABEL,
    CAPTION_LABELS,
    DOCUMENT_TITLE_LABEL,
    FOOTNOTE_LABELS,
    HEADING_LABELS,
    PAGE_NUMBER_LABELS,
    PICTURE_LABELS,
    SKIPPED_LABELS,
    TABLE_LABEL,
)
from pdf2fb2.paragraph_flow import find_printed_page_number, merge_or_append_paragraph
from pdf2fb2.pdf_rendering import crop_pdf_region
from pdf2fb2.segment_geometry import box_area
from pdf2fb2.table_markup import parse_table_text
from pdf2fb2.text_markup import (
    collapse_to_single_line,
    heading_depth,
    join_text_fragments,
    split_markdown_paragraphs,
)

MIN_PICTURE_AREA = 400
LOSSLESS_PICTURE_LABELS = frozenset({"display_formula", TABLE_LABEL})

SegmentHandler = Callable[[dict, dict], None]


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


def _segment_text(page_segment: dict) -> str:
    """Return the stripped OCR text of a segment.

    Args:
        page_segment: Segment, possibly without ``text``.

    Returns:
        Stripped text (empty if none).
    """
    return (page_segment.get("text") or "").strip()


class FlowBuilder:
    """Turns OCR'd page records into a ``BookFlow``; one handler per segment label.

    Usage:
        Create, call ``add_page`` for every page in order, then read ``book_flow``.
        New labels are supported by registering a handler in ``_build_label_handlers``.
    """

    def __init__(self, pdf_document: pymupdf.Document, binary_store: BinaryStore,
                 segmentation_dpi: int, image_dpi: int) -> None:
        """Prepare an empty flow.

        Args:
            pdf_document: Open source PDF (pictures are re-rendered from it).
            binary_store: Receives embedded pictures.
            segmentation_dpi: DPI of the page images the boxes refer to.
            image_dpi: DPI used for embedded pictures.
        """
        self._pdf_document = pdf_document
        self._binary_store = binary_store
        self._segmentation_dpi = segmentation_dpi
        self._image_dpi = image_dpi
        self._picture_counter = 0
        self.book_flow = BookFlow()
        self._label_handlers = self._build_label_handlers()

    def _build_label_handlers(self) -> dict[str, SegmentHandler]:
        """Map segment labels to handler methods; unknown labels become body text.

        Returns:
            Label -> handler mapping.
        """
        label_handlers: dict[str, SegmentHandler] = {
            DOCUMENT_TITLE_LABEL: self._add_document_title,
            TABLE_LABEL: self._add_table,
            ABSTRACT_LABEL: self._add_abstract,
        }
        label_groups = ((HEADING_LABELS, self._add_heading), (CAPTION_LABELS, self._add_caption),
                        (FOOTNOTE_LABELS, self._add_footnotes), (PICTURE_LABELS, self._add_picture))
        for label_group, group_handler in label_groups:
            label_handlers.update(dict.fromkeys(label_group, group_handler))
        return label_handlers

    def add_page(self, page_record: dict) -> None:
        """Add one page: a page marker followed by its segments in reading order.

        Args:
            page_record: Page record with OCR'd segments.

        Returns:
            None.
        """
        self.book_flow.flow_items.append({"kind": "page", "page": page_record["page"],
                                          "printed": find_printed_page_number(page_record)})
        self._picture_counter = 0
        for page_segment in page_record["segments"]:
            segment_label = page_segment["label"]
            if segment_label in SKIPPED_LABELS or segment_label in PAGE_NUMBER_LABELS:
                continue
            self._label_handlers.get(segment_label, self._add_body_text)(page_record, page_segment)

    def _add_picture(self, page_record: dict, page_segment: dict) -> None:
        """Embed a picture-like segment, re-rendered from the PDF.

        Args:
            page_record: Page the segment belongs to.
            page_segment: Picture, chart, formula or unparsed table segment.

        Returns:
            None. Tiny or empty regions are skipped.
        """
        if box_area(page_segment["bbox"]) < MIN_PICTURE_AREA:
            return
        picture_image = crop_pdf_region(self._pdf_document, page_record, page_segment["bbox"],
                                  self._segmentation_dpi, self._image_dpi)
        if picture_image is None:
            return
        self._picture_counter += 1
        lossless = page_segment["label"] in LOSSLESS_PICTURE_LABELS
        binary_id = f"p{page_record['page']:04d}_{self._picture_counter}.{'png' if lossless else 'jpg'}"
        self._binary_store.add_image(binary_id, picture_image, "PNG" if lossless else "JPEG")
        self.book_flow.flow_items.append({"kind": "image", "id": binary_id})

    def _add_table(self, page_record: dict, page_segment: dict) -> None:
        """Add a parsed table, or embed the table as a picture when parsing fails.

        Args:
            page_record: Page the segment belongs to.
            page_segment: Table segment.

        Returns:
            None.
        """
        table_rows = parse_table_text(_segment_text(page_segment))
        if table_rows is None:
            self._add_picture(page_record, page_segment)
            return
        self.book_flow.flow_items.append({"kind": "table", "rows": table_rows})

    def _add_document_title(self, page_record: dict, page_segment: dict) -> None:
        """Use the first document title as the book title; later ones become headings.

        Args:
            page_record: Page the segment belongs to.
            page_segment: ``doc_title`` segment.

        Returns:
            None.
        """
        if self.book_flow.book_title is not None:
            self._add_heading(page_record, page_segment)
            return
        title_text = _segment_text(page_segment)
        if title_text:
            self.book_flow.book_title = collapse_to_single_line(title_text)

    def _add_heading(self, page_record: dict, page_segment: dict) -> None:
        """Add a heading; its depth comes from "1.2.3"-style numbering.

        Args:
            page_record: Page the segment belongs to (unused).
            page_segment: Heading segment.

        Returns:
            None.
        """
        heading_text = collapse_to_single_line(_segment_text(page_segment))
        if heading_text:
            self.book_flow.flow_items.append({"kind": "heading", "level": heading_depth(heading_text),
                                              "text": heading_text})

    def _add_caption(self, page_record: dict, page_segment: dict) -> None:
        """Add a figure/table/chart caption.

        Args:
            page_record: Page the segment belongs to (unused).
            page_segment: Caption segment.

        Returns:
            None.
        """
        caption_text = collapse_to_single_line(_segment_text(page_segment))
        if caption_text:
            self.book_flow.flow_items.append({"kind": "caption", "text": caption_text})

    def _add_footnotes(self, page_record: dict, page_segment: dict) -> None:
        """Add every note of a footnote segment and link its marker in the text.

        Args:
            page_record: Page the segment belongs to.
            page_segment: Footnote segment.

        Returns:
            None.
        """
        for paragraph_text, _ in split_markdown_paragraphs(_segment_text(page_segment)):
            self._add_single_footnote(page_record["page"], paragraph_text)

    def _add_single_footnote(self, page_number: int, paragraph_text: str) -> None:
        """Add one footnote paragraph, or append it to the previous note if it has no marker.

        Args:
            page_number: Page of the footnote.
            paragraph_text: One footnote paragraph.

        Returns:
            None.
        """
        footnotes = self.book_flow.footnotes
        parsed_note = parse_footnote_paragraph(paragraph_text)
        if parsed_note is None and footnotes:
            footnotes[-1]["text"] = join_text_fragments(footnotes[-1]["text"], paragraph_text)
            return
        marker_text, note_body = parsed_note or ("*", paragraph_text)
        note_identifier = f"n{len(footnotes) + 1}"
        footnotes.append({"id": note_identifier, "marker": marker_text, "text": note_body,
                          "linked": link_note_marker(self.book_flow.flow_items, page_number, marker_text,
                                                     note_identifier)})

    def _add_abstract(self, page_record: dict, page_segment: dict) -> None:
        """Add an abstract as cited paragraphs.

        Args:
            page_record: Page the segment belongs to.
            page_segment: Abstract segment.

        Returns:
            None.
        """
        self._add_paragraphs(page_record["page"], _segment_text(page_segment), "cite")

    def _add_body_text(self, page_record: dict, page_segment: dict) -> None:
        """Add body text paragraphs (default for unknown labels).

        Args:
            page_record: Page the segment belongs to.
            page_segment: Text-like segment.

        Returns:
            None.
        """
        self._add_paragraphs(page_record["page"], _segment_text(page_segment), "p")

    def _add_paragraphs(self, page_number: int, segment_text: str, paragraph_style: str) -> None:
        """Split text into paragraphs and add them, re-joining broken paragraphs.

        Args:
            page_number: Page of the text.
            segment_text: OCR Markdown.
            paragraph_style: ``p`` or ``cite``.

        Returns:
            None.
        """
        for paragraph_text, is_list_item in split_markdown_paragraphs(segment_text):
            merge_or_append_paragraph(self.book_flow.flow_items, paragraph_text, page_number,
                                      paragraph_style, is_list_item)


def build_book_flow(page_records: list[dict], pdf_document: pymupdf.Document, binary_store: BinaryStore,
                    segmentation_dpi: int, image_dpi: int) -> BookFlow:
    """Build the book flow from all OCR'd pages.

    Usage:
        Called by the FB2 document builder.

    Args:
        page_records: OCR'd page records in page order.
        pdf_document: Open source PDF.
        binary_store: Receives embedded pictures.
        segmentation_dpi: DPI of the page images.
        image_dpi: DPI for embedded pictures.

    Returns:
        The complete BookFlow.
    """
    flow_builder = FlowBuilder(pdf_document, binary_store, segmentation_dpi, image_dpi)
    for page_record in page_records:
        flow_builder.add_page(page_record)
    return flow_builder.book_flow
