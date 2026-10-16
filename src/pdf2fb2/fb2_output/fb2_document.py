"""Assembles the complete FictionBook 2 document."""

import base64
import datetime
import uuid
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from pdf2fb2.fb2_output.binary_store import BinaryStore
from pdf2fb2.fb2_output.book_flow import build_book_flow
from pdf2fb2.fb2_output.fb2_namespaces import FB2_NAMESPACE, XLINK_NAMESPACE
from pdf2fb2.fb2_output.flow_model import BookFlow
from pdf2fb2.fb2_output.section_tree import render_body_sections
from pdf2fb2.fb2_output.text_markup import escape_text, markdown_to_fb2_inline
from pdf2fb2.rendering.pdf_rendering import render_cover_image

COVER_BINARY_ID = "cover.jpg"
XML_DECLARATION = '<?xml version="1.0" encoding="UTF-8"?>\n'
PROGRAM_NAME = "pdf2fb2 (PP-DocLayoutV3 + Qianfan-OCR)"


@dataclass(frozen=True)
class BookOptions:
    """User-controlled FB2 settings.

    Attributes:
        book_title: Title override (None: detected title, PDF metadata, file name).
        author_name: Author override (None: PDF metadata).
        genre_name: FB2 genre code, e.g. ``nonfiction``.
        language_code: Book language code, e.g. ``en``, ``ru``.
        page_marker_mode: ``anchor``, ``visible`` or ``none``.
        segmentation_dpi: DPI the page images were rendered at.
        image_dpi: DPI for embedded pictures.
    """

    book_title: str | None
    author_name: str | None
    genre_name: str
    language_code: str
    page_marker_mode: str
    segmentation_dpi: int
    image_dpi: int


def _choose_title(book_options: BookOptions, book_flow: BookFlow, pdf_metadata: dict, pdf_path: Path) -> str:
    """Pick the book title by priority: option, detected title, PDF metadata, file name.

    Args:
        book_options: FB2 settings (``book_title`` override).
        book_flow: Built flow (detected ``doc_title``).
        pdf_metadata: PyMuPDF metadata dictionary.
        pdf_path: Source PDF (file stem is the last resort).

    Returns:
        Non-empty title string.
    """
    return (book_options.book_title or book_flow.book_title or (pdf_metadata.get("title") or "").strip()
            or pdf_path.stem)


def _author_xml(author_text: str) -> str:
    """Render the ``<author>`` content from a free-form name.

    Args:
        author_text: Author name; may be empty.

    Returns:
        First/last name elements for two or more words, else a nickname (``Unknown`` if empty).
    """
    name_parts = author_text.split()
    if len(name_parts) >= 2:
        return (f"<first-name>{escape_text(name_parts[0])}</first-name>"
                f"<last-name>{escape_text(' '.join(name_parts[1:]))}</last-name>")
    return f"<nickname>{escape_text(name_parts[0] if name_parts else 'Unknown')}</nickname>"


def _description_xml(book_options: BookOptions, book_title: str, author_text: str) -> str:
    """Render the ``<description>`` element (title-info and document-info).

    Args:
        book_options: FB2 settings.
        book_title: Chosen title.
        author_text: Chosen author name.

    Returns:
        Description markup.
    """
    today_text = datetime.date.today().isoformat()
    return ("<description><title-info>"
            f"<genre>{escape_text(book_options.genre_name)}</genre><author>{_author_xml(author_text)}</author>"
            f"<book-title>{escape_text(book_title)}</book-title>"
            f'<coverpage><image l:href="#{COVER_BINARY_ID}"/></coverpage>'
            f"<lang>{escape_text(book_options.language_code)}</lang>"
            "</title-info><document-info>"
            "<author><nickname>pdf2fb2</nickname></author>"
            f"<program-used>{PROGRAM_NAME}</program-used>"
            f'<date value="{today_text}">{today_text}</date><id>{uuid.uuid4()}</id><version>1.0</version>'
            "<src-ocr>Qianfan-OCR</src-ocr>"
            "</document-info></description>")


def _notes_body_xml(footnotes: list[dict]) -> str:
    """Render the notes body.

    Args:
        footnotes: Notes from the book flow.

    Returns:
        ``<body name="notes">`` markup, or an empty string when there are no notes.
    """
    if not footnotes:
        return ""
    note_sections = "".join(
        f'<section id="{footnote["id"]}"><title><p>{escape_text(footnote["marker"])}</p></title>'
        f'<p>{markdown_to_fb2_inline(footnote["text"])}</p></section>' for footnote in footnotes)
    return f'<body name="notes"><title><p>Notes</p></title>{note_sections}</body>'


def _binaries_xml(binary_store: BinaryStore) -> str:
    """Render all ``<binary>`` elements.

    Args:
        binary_store: Stored pictures.

    Returns:
        Concatenated binary elements with base64 payloads.
    """
    return "".join(f'<binary id="{binary_id}" content-type="{content_type}">'
                   f"{base64.b64encode(encoded_bytes).decode()}</binary>"
                   for binary_id, (content_type, encoded_bytes) in binary_store.stored_items.items())


def collect_statistics(book_flow: BookFlow, binary_store: BinaryStore) -> dict:
    """Count the main elements of the book.

    Args:
        book_flow: Built flow.
        binary_store: Stored pictures (including the cover).

    Returns:
        Counts of paragraphs, images (without cover), tables, headings, notes and linked notes.
    """
    def count_kind(item_kind: str) -> int:
        """Count flow items of one kind.

        Args:
            item_kind: Flow item kind.

        Returns:
            Number of matching items.
        """
        return sum(1 for flow_item in book_flow.flow_items if flow_item["kind"] == item_kind)

    return {"paragraphs": count_kind("p"), "images": len(binary_store.stored_items) - 1,
            "tables": count_kind("table"), "headings": count_kind("heading"),
            "notes": len(book_flow.footnotes),
            "notes_linked": sum(1 for footnote in book_flow.footnotes if footnote["linked"])}


@dataclass
class PreparedBook:
    """Book content before rendering: the flow (editable, e.g. by proofreading) plus its pictures.

    Attributes:
        book_flow: Flow items, notes and detected title.
        binary_store: Pictures, including the cover.
        pdf_metadata: PyMuPDF metadata of the source PDF.
    """

    book_flow: BookFlow
    binary_store: BinaryStore
    pdf_metadata: dict


def prepare_book(pdf_path: Path, page_records: list[dict], book_options: BookOptions) -> PreparedBook:
    """Build the flow, the pictures and the cover of a book.

    Usage:
        First assembly step; edit ``book_flow`` if needed, then call ``render_prepared_book``.

    Args:
        pdf_path: Source PDF (pictures, cover and metadata come from it).
        page_records: OCR'd page records in page order.
        book_options: FB2 settings.

    Returns:
        The prepared book.
    """
    binary_store = BinaryStore()
    with pymupdf.open(pdf_path) as pdf_document:
        book_flow = build_book_flow(page_records, pdf_document, binary_store,
                                    book_options.segmentation_dpi, book_options.image_dpi)
        pdf_metadata = pdf_document.metadata or {}
        binary_store.add_image(COVER_BINARY_ID, render_cover_image(pdf_document))
    return PreparedBook(book_flow=book_flow, binary_store=binary_store, pdf_metadata=pdf_metadata)


def render_prepared_book(prepared_book: PreparedBook, pdf_path: Path, book_options: BookOptions) -> tuple[str, dict]:
    """Render the FB2 XML of a prepared book.

    Args:
        prepared_book: Result of ``prepare_book`` (possibly edited).
        pdf_path: Source PDF (its stem is the last-resort title).
        book_options: FB2 settings.

    Returns:
        ``(fb2_markup, statistics)``.
    """
    book_flow, binary_store = prepared_book.book_flow, prepared_book.binary_store
    book_title = _choose_title(book_options, book_flow, prepared_book.pdf_metadata, pdf_path)
    author_text = book_options.author_name or (prepared_book.pdf_metadata.get("author") or "").strip()
    body_xml = render_body_sections(book_flow.flow_items, book_options.page_marker_mode)
    fb2_markup = "".join([
        XML_DECLARATION, f'<FictionBook xmlns="{FB2_NAMESPACE}" xmlns:l="{XLINK_NAMESPACE}">',
        _description_xml(book_options, book_title, author_text),
        f"<body><title><p>{markdown_to_fb2_inline(book_title)}</p></title>{body_xml}</body>",
        _notes_body_xml(book_flow.footnotes), _binaries_xml(binary_store), "</FictionBook>"])
    return fb2_markup, collect_statistics(book_flow, binary_store)


def build_fb2_document(pdf_path: Path, page_records: list[dict], book_options: BookOptions) -> tuple[str, dict]:
    """Build the FB2 XML for OCR'd pages without any editing step.

    Usage:
        Final assembly step when no proofreading is wanted.

    Args:
        pdf_path: Source PDF (pictures, cover and metadata come from it).
        page_records: OCR'd page records in page order.
        book_options: FB2 settings.

    Returns:
        ``(fb2_markup, statistics)``.
    """
    return render_prepared_book(prepare_book(pdf_path, page_records, book_options), pdf_path, book_options)
