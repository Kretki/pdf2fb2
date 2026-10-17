"""Registry of the supported input formats, dispatching on the file suffix.

Usage:
    ``SourceFormatRegistry({".pdf": PdfPassThrough(), ".djvu": DjvuToPdfConverter()})`` is built by the wiring
    code; the pipeline calls ``prepare_document`` once per book. A new format is one more preparer.
"""

from pathlib import Path
from typing import Protocol


class DocumentConversionError(RuntimeError):
    """Raised when a book cannot be turned into a PDF (unsupported format, missing tool, failed converter)."""


class DocumentPreparer(Protocol):
    """Turns one input book into a PDF."""

    def prepare_document(self, book_path: Path, work_directory: Path) -> Path:
        """Return a PDF with the book's pages.

        Args:
            book_path: Input book.
            work_directory: Conversion output directory (converted files may be cached there).

        Returns:
            Path of a PDF named after the book (same stem).

        Raises:
            DocumentConversionError: If the book cannot be converted.
        """
        ...


class SourceFormatRegistry:
    """Chooses the preparer of a book by its suffix.

    Usage:
        Inject into ``ConversionPipeline``; ``supported_suffixes`` also drives the input-directory scan.
    """

    def __init__(self, preparers_by_suffix: dict[str, DocumentPreparer]) -> None:
        """Store the preparers.

        Args:
            preparers_by_suffix: Lowercase suffix with dot (``.pdf``) -> preparer.
        """
        self._preparers_by_suffix = dict(preparers_by_suffix)

    @property
    def supported_suffixes(self) -> frozenset[str]:
        """Suffixes that can be converted.

        Returns:
            Lowercase suffixes with the dot.
        """
        return frozenset(self._preparers_by_suffix)

    def prepare_document(self, book_path: Path, work_directory: Path) -> Path:
        """Turn a book into a PDF with the preparer of its suffix.

        Args:
            book_path: Input book.
            work_directory: Conversion output directory.

        Returns:
            Path of the PDF to render.

        Raises:
            DocumentConversionError: For unsupported suffixes or failed conversions.
        """
        document_preparer = self._preparers_by_suffix.get(book_path.suffix.lower())
        if document_preparer is None:
            supported_text = ", ".join(sorted(self._preparers_by_suffix))
            raise DocumentConversionError(f"unsupported book format {book_path.suffix!r} (supported: {supported_text})")
        return document_preparer.prepare_document(book_path, work_directory)
