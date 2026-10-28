"""Books of the test corpus: ``catalog.csv`` rows whose reference text is usable (``ocr_state == correct``).

Every path of the catalog is relative to the corpus root; paths that are absolute, contain ``..`` or resolve
outside the root are rejected. A book's reference is either paged (Internet Archive ``_djvu.xml`` hidden text,
NDL per-page OCR zip) or one transcript of the whole book (Wikisource proofread text, plain OCR text).
"""

import csv
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

CATALOG_FILE = "catalog.csv"
CORRECT_STATE = "correct"
STATE_COLUMN = "ocr_state"
DJVU_XML_REFERENCE = "djvu_xml"
NDL_ZIP_REFERENCE = "ndl_zip"
TRANSCRIPT_REFERENCE = "transcript"
MAX_CATALOG_BYTES = 50_000_000


@dataclass(frozen=True)
class CatalogBook:
    """One benchmark book.

    Attributes:
        book_key: Unique key of the catalog row.
        language_name: Corpus folder language, e.g. ``old_russian``.
        language_code: Code passed to ``pdf2fb2 --lang``, e.g. ``ru``.
        scan_path: Absolute path of the scan (PDF or DjVu).
        reference_kind: ``djvu_xml``, ``ndl_zip`` or ``transcript``.
        reference_path: Absolute path of the reference text.
        page_count: Verified page count of the scan.
        text_quality: Catalog ``text_quality`` (``hand_proofread_same_edition`` marks proofread references).
    """

    book_key: str
    language_name: str
    language_code: str
    scan_path: Path
    reference_kind: str
    reference_path: Path
    page_count: int
    text_quality: str = ""

    @property
    def is_proofread(self) -> bool:
        """Tell whether the reference was proofread by hand (its labels are trusted as they are).

        Returns:
            True for hand-proofread references.
        """
        return "hand_proofread" in self.text_quality


def corpus_path(corpus_root: Path, relative_text: str) -> Path:
    """Resolve a catalog path safely inside the corpus.

    Args:
        corpus_root: Absolute corpus directory.
        relative_text: Path as written in the catalog.

    Returns:
        Absolute resolved path.

    Raises:
        ValueError: If the path is empty, absolute, contains ``..`` or leaves the corpus.
    """
    relative_path = PurePosixPath(relative_text)
    if not relative_text or relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe catalog path: {relative_text!r}")
    resolved_path = (corpus_root / relative_path).resolve()
    if not resolved_path.is_relative_to(corpus_root.resolve()):
        raise ValueError(f"Catalog path leaves the corpus: {relative_text!r}")
    return resolved_path


def reference_of(catalog_row: dict) -> tuple[str, str]:
    """Kind and relative path of a row's reference text.

    Args:
        catalog_row: Catalog row.

    Returns:
        ``(reference kind, relative path)``; structured paged text is preferred over a whole-book text.
    """
    structured_text = catalog_row.get("structured_text_path") or ""
    if structured_text.endswith(".xml"):
        return DJVU_XML_REFERENCE, structured_text
    if structured_text.endswith(".zip"):
        return NDL_ZIP_REFERENCE, structured_text
    return TRANSCRIPT_REFERENCE, catalog_row["parsed_text_path"]


def catalog_book(corpus_root: Path, catalog_row: dict) -> CatalogBook:
    """Build a book from one catalog row.

    Args:
        corpus_root: Absolute corpus directory.
        catalog_row: Row with ``ocr_state == correct``.

    Returns:
        The book.

    Raises:
        ValueError: If a path is unsafe or the page count is not a positive integer.
    """
    reference_kind, reference_text = reference_of(catalog_row)
    page_count = int(catalog_row["page_count_verified"])
    if page_count <= 0:
        raise ValueError(f"Book {catalog_row['book_key']!r} has no pages.")
    return CatalogBook(book_key=catalog_row["book_key"], language_name=catalog_row["language"],
                       language_code=catalog_row["language_code"],
                       scan_path=corpus_path(corpus_root, catalog_row["image_pdf_path"]),
                       reference_kind=reference_kind, reference_path=corpus_path(corpus_root, reference_text),
                       page_count=page_count, text_quality=catalog_row.get("text_quality") or "")


def read_correct_books(corpus_root: Path) -> list[CatalogBook]:
    """All books of the catalog whose reference text is marked correct.

    Usage:
        Entry point of every benchmark and export command; mixed and wrong books are never used.

    Args:
        corpus_root: Corpus directory holding ``catalog.csv``.

    Returns:
        Books in catalog order.

    Raises:
        ValueError: If the catalog is too large, lacks the ``ocr_state`` column or holds unsafe paths.
    """
    catalog_path = corpus_root / CATALOG_FILE
    if catalog_path.stat().st_size > MAX_CATALOG_BYTES:
        raise ValueError(f"{catalog_path} is larger than {MAX_CATALOG_BYTES} bytes.")
    with catalog_path.open(encoding="utf-8", newline="") as catalog_file:
        catalog_reader = csv.DictReader(catalog_file)
        if STATE_COLUMN not in (catalog_reader.fieldnames or []):
            raise ValueError(f"{catalog_path} has no {STATE_COLUMN!r} column.")
        return [catalog_book(corpus_root.resolve(), catalog_row) for catalog_row in catalog_reader
                if catalog_row[STATE_COLUMN] == CORRECT_STATE]
