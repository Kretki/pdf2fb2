"""DjVu input: converted once to a lossless PDF with DjVuLibre's ``ddjvu`` and cached in the output directory.

The PDF keeps the scan resolution of every page (bitonal pages as CCITT G4, colour pages losslessly), so page
rendering, picture crops and the cover work exactly as for a scanned PDF. The converted file is
``<output>/source/<book stem>.pdf``; it is reused while it is newer than the DjVu file.

Requires ``ddjvu`` on PATH (Debian/Ubuntu: ``sudo apt install djvulibre-bin``).
"""

# B404/S404: subprocess is used only with a fixed argv list and no shell.
import shutil
import subprocess  # nosec B404  # noqa: S404
from collections.abc import Callable
from pathlib import Path

from pdf2fb2.document_sources.source_formats import DocumentConversionError

SOURCE_DIRECTORY = "source"
CONVERTER_NAME = "ddjvu"
CONVERSION_TIMEOUT_SECONDS = 3600
PARTIAL_SUFFIX = ".partial"
MAX_ERROR_CHARACTERS = 400
CommandRunner = Callable[[list[str], int], subprocess.CompletedProcess]


def run_converter(command_arguments: list[str], timeout_seconds: int) -> subprocess.CompletedProcess:
    """Run the converter without a shell and capture its output.

    Args:
        command_arguments: Executable path followed by its arguments.
        timeout_seconds: Time limit of the run.

    Returns:
        The finished process (output captured as text).

    Raises:
        subprocess.TimeoutExpired: If the converter runs too long.
    """
    return subprocess.run(command_arguments, capture_output=True, text=True,  # nosec B603  # noqa: S603
                          timeout=timeout_seconds, check=False)


def is_cached_conversion(converted_path: Path, book_path: Path) -> bool:
    """Tell whether an earlier conversion can be reused.

    Args:
        converted_path: Converted PDF.
        book_path: Source DjVu file.

    Returns:
        True when the PDF exists, is not empty and is newer than the DjVu file.
    """
    return (converted_path.is_file() and converted_path.stat().st_size > 0
            and converted_path.stat().st_mtime >= book_path.stat().st_mtime)


class DjvuToPdfConverter:
    """``DocumentPreparer`` for DjVu books.

    Usage:
        Registered for ``.djvu`` and ``.djv`` in the source-format registry.
    """

    def __init__(self, command_runner: CommandRunner = run_converter,
                 executable_finder: Callable[[str], str | None] = shutil.which) -> None:
        """Store how the converter is found and run (injectable for tests).

        Args:
            command_runner: Runs ``(arguments, timeout)`` and returns the finished process.
            executable_finder: Returns the full path of an executable on PATH, or None.
        """
        self._command_runner = command_runner
        self._executable_finder = executable_finder

    def prepare_document(self, book_path: Path, work_directory: Path) -> Path:
        """Convert the DjVu book to ``<work_directory>/source/<stem>.pdf`` unless a fresh copy exists.

        Args:
            book_path: Input DjVu file.
            work_directory: Conversion output directory.

        Returns:
            Path of the converted PDF.

        Raises:
            DocumentConversionError: If ``ddjvu`` is missing, fails, times out or writes no PDF.
        """
        converted_path = work_directory / SOURCE_DIRECTORY / f"{book_path.stem}.pdf"
        if is_cached_conversion(converted_path, book_path):
            return converted_path
        converter_path = self._executable_finder(CONVERTER_NAME)
        if converter_path is None:
            raise DocumentConversionError(f"{CONVERTER_NAME} not found: install DjVuLibre "
                                          "(e.g. 'sudo apt install djvulibre-bin') to convert DjVu books")
        converted_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = converted_path.with_name(converted_path.name + PARTIAL_SUFFIX)
        self._run_conversion(converter_path, book_path, partial_path)
        partial_path.replace(converted_path)
        return converted_path

    def _run_conversion(self, converter_path: str, book_path: Path, partial_path: Path) -> None:
        """Run ``ddjvu -format=pdf`` into a partial file.

        Args:
            converter_path: Full path of ``ddjvu``.
            book_path: Input DjVu file (passed as an absolute path, so it can never look like an option).
            partial_path: Output file, renamed by the caller once complete.

        Raises:
            DocumentConversionError: If the converter fails, times out or writes nothing.
        """
        command_arguments = [converter_path, "-format=pdf", str(book_path.resolve()), str(partial_path.resolve())]
        try:
            finished_process = self._command_runner(command_arguments, CONVERSION_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as timeout_error:
            partial_path.unlink(missing_ok=True)
            raise DocumentConversionError(f"{CONVERTER_NAME} timed out on {book_path.name}") from timeout_error
        if finished_process.returncode != 0 or not partial_path.is_file() or partial_path.stat().st_size == 0:
            partial_path.unlink(missing_ok=True)
            error_text = (finished_process.stderr or "").strip()[-MAX_ERROR_CHARACTERS:]
            raise DocumentConversionError(f"{CONVERTER_NAME} failed on {book_path.name} "
                                          f"(exit {finished_process.returncode}): {error_text}")
