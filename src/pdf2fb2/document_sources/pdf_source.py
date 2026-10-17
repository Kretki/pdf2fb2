"""PDF input: already in the format the pipeline renders."""

from pathlib import Path


class PdfPassThrough:
    """``DocumentPreparer`` for PDF books.

    Usage:
        Registered for ``.pdf`` in the source-format registry.
    """

    def prepare_document(self, book_path: Path, work_directory: Path) -> Path:
        """Return the PDF itself.

        Args:
            book_path: Input PDF.
            work_directory: Conversion output directory (unused).

        Returns:
            ``book_path``.
        """
        return book_path
