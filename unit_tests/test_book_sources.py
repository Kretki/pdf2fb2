"""Unit tests of the input formats (DjVu conversion), the book discovery and the batch conversion."""

import io
import subprocess  # nosec B404  # noqa: S404
from pathlib import Path

import pytest

from pdf2fb2.document_sources.djvu_source import SOURCE_DIRECTORY, DjvuToPdfConverter
from pdf2fb2.document_sources.pdf_source import PdfPassThrough
from pdf2fb2.document_sources.source_formats import DocumentConversionError, SourceFormatRegistry
from pdf2fb2.interface.batch_conversion import convert_books
from pdf2fb2.interface.book_jobs import BookConversionJob, find_input_books, single_book_job

BOOK_SUFFIXES = frozenset({".pdf", ".djvu"})


class FakeConverterRunner:
    """Stands in for ``ddjvu``: writes a small file to the output argument and records the calls.

    Usage:
        Injected into ``DjvuToPdfConverter``.
    """

    def __init__(self, exit_code: int = 0) -> None:
        """Store the exit code to report.

        Args:
            exit_code: Return code of every run.
        """
        self.exit_code = exit_code
        self.recorded_calls: list[list[str]] = []

    def __call__(self, command_arguments: list[str], timeout_seconds: int) -> subprocess.CompletedProcess:
        """Pretend to convert.

        Args:
            command_arguments: ``[ddjvu, -format=pdf, input, output]``.
            timeout_seconds: Time limit (unused).

        Returns:
            Finished process with the configured exit code.
        """
        self.recorded_calls.append(command_arguments)
        if self.exit_code == 0:
            Path(command_arguments[-1]).write_bytes(b"%PDF-1.4 fake")
        return subprocess.CompletedProcess(command_arguments, self.exit_code, "", "bad file")


def write_books(input_directory: Path, relative_names: list[str]) -> None:
    """Create empty book files.

    Args:
        input_directory: Root directory.
        relative_names: Paths relative to it.
    """
    for relative_name in relative_names:
        book_path = input_directory / relative_name
        book_path.parent.mkdir(parents=True, exist_ok=True)
        book_path.write_bytes(b"book")


def test_djvu_is_converted_once_and_cached(tmp_path: Path) -> None:
    """The converted PDF lands in ``source/`` named after the book and is reused while fresh."""
    write_books(tmp_path, ["book.djvu"])
    converter_runner = FakeConverterRunner()
    djvu_converter = DjvuToPdfConverter(converter_runner, lambda executable_name: "/usr/bin/ddjvu")
    converted_path = djvu_converter.prepare_document(tmp_path / "book.djvu", tmp_path / "out")
    assert converted_path == tmp_path / "out" / SOURCE_DIRECTORY / "book.pdf"
    assert converted_path.read_bytes().startswith(b"%PDF")
    assert converter_runner.recorded_calls[0][:2] == ["/usr/bin/ddjvu", "-format=pdf"]
    djvu_converter.prepare_document(tmp_path / "book.djvu", tmp_path / "out")
    assert len(converter_runner.recorded_calls) == 1


def test_djvu_errors_are_reported(tmp_path: Path) -> None:
    """A missing ``ddjvu`` and a failed run raise ``DocumentConversionError`` and leave no partial file."""
    write_books(tmp_path, ["book.djvu"])
    with pytest.raises(DocumentConversionError, match="djvulibre"):
        DjvuToPdfConverter(FakeConverterRunner(), lambda executable_name: None).prepare_document(
            tmp_path / "book.djvu", tmp_path / "out")
    with pytest.raises(DocumentConversionError, match="bad file"):
        DjvuToPdfConverter(FakeConverterRunner(1), lambda executable_name: "/usr/bin/ddjvu").prepare_document(
            tmp_path / "book.djvu", tmp_path / "out")
    assert not list((tmp_path / "out" / SOURCE_DIRECTORY).iterdir())


def test_registry_dispatches_on_suffix(tmp_path: Path) -> None:
    """PDFs pass through; unknown formats are rejected."""
    source_registry = SourceFormatRegistry({".pdf": PdfPassThrough()})
    assert source_registry.prepare_document(tmp_path / "Book.PDF", tmp_path) == tmp_path / "Book.PDF"
    with pytest.raises(DocumentConversionError, match="unsupported"):
        source_registry.prepare_document(tmp_path / "book.epub", tmp_path)


def test_input_books_mirror_folders_and_split_same_names(tmp_path: Path) -> None:
    """Sub-folders are mirrored, same-stem books get their format appended, hidden and other files are skipped."""
    input_directory = tmp_path / "input"
    write_books(input_directory, ["alpha.pdf", "alpha.djvu", "series/beta.djvu", ".hidden/gamma.pdf", "notes.txt"])
    book_jobs = find_input_books(input_directory, tmp_path / "output", BOOK_SUFFIXES)
    output_names = {book_job.book_path.name: book_job.output_directory.relative_to(tmp_path / "output").as_posix()
                    for book_job in book_jobs}
    assert output_names == {"alpha.djvu": "alpha_djvu", "alpha.pdf": "alpha_pdf", "beta.djvu": "series/beta"}
    with pytest.raises(FileNotFoundError):
        find_input_books(tmp_path / "missing", tmp_path / "output", BOOK_SUFFIXES)


def test_single_book_goes_to_output_named_after_it(tmp_path: Path) -> None:
    """Without ``-o`` the results go to ``<output root>/<stem>``."""
    assert single_book_job(Path("input/joaquim.pdf"), Path("output"), None).output_directory == Path(
        "output/joaquim")
    assert single_book_job(Path("a.pdf"), Path("output"), Path("custom")).output_directory == Path("custom")


def test_batch_continues_after_a_failed_book() -> None:
    """A failing book is summarised and the next one still converts; the exit code reports the failure."""
    book_jobs = [BookConversionJob(Path("bad.djvu"), Path("output/bad")),
                 BookConversionJob(Path("good.pdf"), Path("output/good"))]
    converted_books = []

    def convert_book(book_job: BookConversionJob) -> int:
        """Fail the DjVu book, convert the PDF.

        Args:
            book_job: Book to convert.

        Returns:
            0 for the PDF.
        """
        if book_job.book_path.suffix == ".djvu":
            raise DocumentConversionError("ddjvu not found")
        converted_books.append(book_job.book_path.name)
        return 0

    summary_stream = io.StringIO()
    assert convert_books(book_jobs, convert_book, summary_stream) == 1
    assert converted_books == ["good.pdf"]
    assert "ok: 1, checks failed: 0, error: 1" in summary_stream.getvalue()
    assert convert_books([book_jobs[1]], convert_book, io.StringIO()) == 0
