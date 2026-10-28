"""Reference text of a corpus book, as the search region for one scanned page.

For paged references the region of a page is that page followed by the next one: a paragraph that starts on
the page may end on the next page, and the output attributes it to the page where it starts.

* ``djvu_xml``: Internet Archive hidden text; one ``<OBJECT>`` per scan page, words in ``<WORD>`` elements.
  Parsed with bounded regular expressions (no XML parser runs on the downloaded file).
* ``ndl_zip``: National Diet Library OCR; one ``<id>_<page>.txt`` member per page.
* ``transcript``: one text of the whole book (Wikisource proofread transcription, plain OCR text); the
  whole book is the search region and ``reference_locator`` finds the page in it.
"""

import html
import re
import zipfile
from pathlib import Path
from typing import Protocol

from pdf2fb2.benchmark.corpus_catalog import DJVU_XML_REFERENCE, NDL_ZIP_REFERENCE, CatalogBook
from pdf2fb2.benchmark.text_normalization import comparison_words

MAX_REFERENCE_BYTES = 400_000_000
MAX_MEMBER_BYTES = 5_000_000
MAX_ARCHIVE_MEMBERS = 20_000
PAGE_OBJECT_PATTERN = re.compile(r"<OBJECT\b.*?</OBJECT>", re.DOTALL)
WORD_ELEMENT_PATTERN = re.compile(r"<WORD\b[^>]{0,400}>([^<]{0,400})</WORD>")
LINE_END_PATTERN = re.compile(r"</LINE>")
PAGE_MEMBER_PATTERN = re.compile(r"^[\w.-]{1,120}_(\d{1,8})\.txt$")


class ReferenceSource(Protocol):
    """Normalised reference words in which one scan page is searched."""

    def region_words(self, page_index: int) -> list[str]:
        """Words that contain the reference of a page.

        Args:
            page_index: 0-based page of the scan.

        Returns:
            Normalised words (the page itself, or the whole book for a transcript).
        """
        ...


def read_bounded_text(text_path: Path) -> str:
    """Read a reference file with a size limit.

    Args:
        text_path: File to read.

    Returns:
        Its text (undecodable bytes replaced).

    Raises:
        ValueError: If the file is larger than ``MAX_REFERENCE_BYTES``.
    """
    if text_path.stat().st_size > MAX_REFERENCE_BYTES:
        raise ValueError(f"{text_path} is larger than {MAX_REFERENCE_BYTES} bytes.")
    return text_path.read_text(encoding="utf-8", errors="replace")


class DjvuXmlPages:
    """Pages of an Internet Archive ``_djvu.xml`` file.

    Usage:
        Built by ``reference_source``; pages are parsed on first use.
    """

    def __init__(self, xml_path: Path) -> None:
        """Remember the file.

        Args:
            xml_path: The ``_djvu.xml`` file.
        """
        self._xml_path = xml_path
        self._page_words: list[list[str]] | None = None

    def region_words(self, page_index: int) -> list[str]:
        """Normalised words of one page and the next (empty for pages without text or out of range).

        Args:
            page_index: 0-based page of the scan.

        Returns:
            Words of the page followed by the words of the next page.
        """
        if self._page_words is None:
            page_objects = PAGE_OBJECT_PATTERN.findall(read_bounded_text(self._xml_path))
            self._page_words = [comparison_words(" ".join(
                html.unescape(word_text) for word_text in WORD_ELEMENT_PATTERN.findall(page_object)))
                for page_object in page_objects]
        return [page_word for region_page in (page_index, page_index + 1)
                if 0 <= region_page < len(self._page_words) for page_word in self._page_words[region_page]]


class NdlZipPages:
    """Pages of a National Diet Library OCR archive.

    Usage:
        Built by ``reference_source``; the member list is read on first use.
    """

    def __init__(self, zip_path: Path) -> None:
        """Remember the archive.

        Args:
            zip_path: The ``*_ndl_ocr.zip`` file.
        """
        self._zip_path = zip_path
        self._page_members: dict[int, str] | None = None

    def _members(self) -> dict[int, str]:
        """Page number -> member name, with archive limits checked.

        Returns:
            Mapping of 1-based page numbers to member names.

        Raises:
            ValueError: If the archive has too many members.
        """
        if self._page_members is None:
            with zipfile.ZipFile(self._zip_path) as page_archive:
                member_infos = page_archive.infolist()
                if len(member_infos) > MAX_ARCHIVE_MEMBERS:
                    raise ValueError(f"{self._zip_path} has too many members.")
                self._page_members = {int(name_match.group(1)): member_info.filename for member_info in member_infos
                                      if (name_match := PAGE_MEMBER_PATTERN.match(member_info.filename))
                                      and member_info.file_size <= MAX_MEMBER_BYTES}
        return self._page_members

    def region_words(self, page_index: int) -> list[str]:
        """Normalised words of one page and the next.

        Args:
            page_index: 0-based page of the scan (member numbers are 1-based).

        Returns:
            Words of the page followed by the words of the next page (missing members contribute nothing).
        """
        page_members = self._members()
        member_names = [page_members[member_number] for member_number in (page_index + 1, page_index + 2)
                        if member_number in page_members]
        with zipfile.ZipFile(self._zip_path) as page_archive:
            return [page_word for member_name in member_names for page_word in comparison_words(
                page_archive.read(member_name).decode("utf-8", errors="replace"))]


class BookTranscript:
    """One text of the whole book.

    Usage:
        Built by ``reference_source``; read once on first use.
    """

    def __init__(self, text_path: Path) -> None:
        """Remember the file.

        Args:
            text_path: Transcript file.
        """
        self._text_path = text_path
        self._book_words: list[str] | None = None

    def region_words(self, page_index: int) -> list[str]:
        """All words of the book (the page is located later).

        Args:
            page_index: Ignored: a transcript has no page boundaries.

        Returns:
            Normalised words of the whole book.
        """
        if self._book_words is None:
            self._book_words = comparison_words(read_bounded_text(self._text_path))
        return self._book_words


def reference_source(catalog_book: CatalogBook) -> ReferenceSource:
    """Reference reader matching a book's reference kind.

    Args:
        catalog_book: Benchmark book.

    Returns:
        The reader.
    """
    source_builders = {DJVU_XML_REFERENCE: DjvuXmlPages, NDL_ZIP_REFERENCE: NdlZipPages}
    return source_builders.get(catalog_book.reference_kind, BookTranscript)(catalog_book.reference_path)
