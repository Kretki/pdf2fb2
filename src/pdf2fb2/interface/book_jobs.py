"""Which books to convert and where their results go.

Two modes:

* one book: results in ``<output root>/<book stem>`` (or the directory given with ``-o``);
* input directory: every supported book under it, sub-folders included, each into
  ``<output root>/<sub-folders>/<book stem>``. Books of one folder sharing a stem (``a.pdf`` and ``a.djvu``) get
  the format appended (``a_pdf``, ``a_djvu``). Hidden files and links leading outside the input directory are
  skipped.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BookConversionJob:
    """One book to convert.

    Attributes:
        book_path: Input book file.
        output_directory: Directory for its pages, caches and FB2.
    """

    book_path: Path
    output_directory: Path


def single_book_job(book_path: Path, output_root: Path, explicit_output: Path | None) -> BookConversionJob:
    """Job of a book given on the command line.

    Args:
        book_path: Input book file.
        output_root: Root of the result folders (e.g. ``output``).
        explicit_output: Directory given with ``-o``, or None.

    Returns:
        The job; its output is ``explicit_output`` or ``<output_root>/<book stem>``.
    """
    return BookConversionJob(book_path, explicit_output or output_root / book_path.stem)


def is_visible_book(candidate_path: Path, input_root: Path, supported_suffixes: frozenset[str]) -> bool:
    """Tell whether a file under the input directory is a book to convert.

    Args:
        candidate_path: File found under the input directory.
        input_root: Resolved input directory.
        supported_suffixes: Lowercase suffixes with the dot.

    Returns:
        True for regular files of a supported format, not hidden (no part starting with ``.``) and resolving
        inside the input directory.
    """
    relative_parts = candidate_path.relative_to(input_root).parts
    return (candidate_path.suffix.lower() in supported_suffixes and candidate_path.is_file()
            and not any(path_part.startswith(".") for path_part in relative_parts)
            and candidate_path.resolve().is_relative_to(input_root))


def job_folder(relative_stem: Path, book_suffix: str, shares_stem: bool) -> Path:
    """Result folder of a book, relative to the output root.

    Args:
        relative_stem: Book path relative to the input directory, without its suffix.
        book_suffix: Book suffix with the dot, e.g. ``.djvu``.
        shares_stem: Whether another book of the same folder has the same stem.

    Returns:
        ``relative_stem``, or ``<relative_stem>_<format>`` when the stem is shared.
    """
    if not shares_stem:
        return relative_stem
    return relative_stem.with_name(f"{relative_stem.name}_{book_suffix.lower().lstrip('.')}")


def find_input_books(input_directory: Path, output_root: Path,
                     supported_suffixes: frozenset[str]) -> list[BookConversionJob]:
    """Jobs of every book under an input directory.

    Args:
        input_directory: Directory with the books (sub-folders are searched too).
        output_root: Root of the result folders; the input sub-folders are mirrored under it.
        supported_suffixes: Lowercase suffixes with the dot.

    Returns:
        Jobs sorted by book path.

    Raises:
        FileNotFoundError: If the input directory does not exist.
    """
    input_root = input_directory.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_directory}")
    book_paths = sorted(candidate_path for candidate_path in input_root.rglob("*")
                        if is_visible_book(candidate_path, input_root, supported_suffixes))
    relative_stems = [book_path.relative_to(input_root).with_suffix("") for book_path in book_paths]
    stem_counts = Counter(relative_stems)
    return [BookConversionJob(book_path, output_root / job_folder(relative_stem, book_path.suffix,
                                                                  stem_counts[relative_stem] > 1))
            for book_path, relative_stem in zip(book_paths, relative_stems, strict=True)]
