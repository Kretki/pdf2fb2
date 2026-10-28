"""Benchmark run: convert sampled pages of every book of a split with the real pipeline.

Each book gets ``<run>/<book>/subset.pdf`` (the sampled pages) and ``<run>/<book>/conversion/`` (a normal
pdf2fb2 output folder). ``run_manifest.json`` records which original page every subset page is, so scoring
(``benchmark_scoring``) can run again later without converting.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pymupdf

from pdf2fb2.benchmark.book_splits import locked_splits
from pdf2fb2.benchmark.corpus_catalog import CatalogBook, read_correct_books
from pdf2fb2.benchmark.page_sampling import sample_pages, write_page_subset
from pdf2fb2.document_sources.source_formats import DocumentPreparer

RUN_MANIFEST_FILE = "run_manifest.json"
SUBSET_FILE = "subset.pdf"
CONVERSION_DIRECTORY = "conversion"
SOURCE_DIRECTORY = "source"
SEGMENTS_FILE = "segments.json"
UNSAFE_NAME_PATTERN = re.compile(r"[^\w.-]")


class BookConverter(Protocol):
    """Converts one PDF with the pdf2fb2 pipeline."""

    def convert_book(self, pdf_path: Path, language_code: str, output_directory: Path) -> int:
        """Run the conversion.

        Args:
            pdf_path: PDF to convert.
            language_code: Value of ``--lang``.
            output_directory: Conversion output folder.

        Returns:
            Exit code of the conversion (0 = all checks passed).
        """
        ...


@dataclass(frozen=True)
class RunSettings:
    """What to run.

    Attributes:
        corpus_root: Corpus directory with ``catalog.csv``.
        data_directory: Directory of ``splits.json`` and exported datasets.
        run_directory: Output directory of this run.
        split_name: ``train``, ``valid`` or ``test``.
        pages_per_book: Sampled pages per book.
        language_names: Corpus languages to include (empty: all).
        reuse_outputs: Keep existing conversions instead of converting again.
    """

    corpus_root: Path
    data_directory: Path
    run_directory: Path
    split_name: str
    pages_per_book: int
    language_names: tuple[str, ...] = ()
    reuse_outputs: bool = False


def book_folder(run_directory: Path, book_key: str) -> Path:
    """Folder of one book inside a run.

    Args:
        run_directory: Run output directory.
        book_key: Catalog key (may contain any letters).

    Returns:
        ``<run>/<key with unsafe characters replaced>``.
    """
    return run_directory / UNSAFE_NAME_PATTERN.sub("_", book_key)


def selected_books(run_settings: RunSettings) -> list[CatalogBook]:
    """Correct books of the requested split and languages.

    Args:
        run_settings: Run settings.

    Returns:
        Books in catalog order.
    """
    catalog_books = read_correct_books(run_settings.corpus_root)
    book_splits = locked_splits(catalog_books, run_settings.data_directory)
    return [catalog_book for catalog_book in catalog_books
            if book_splits[catalog_book.book_key] == run_settings.split_name
            and (not run_settings.language_names or catalog_book.language_name in run_settings.language_names)]


def is_reused(run_settings: RunSettings, catalog_book: CatalogBook) -> bool:
    """Tell whether a book keeps an existing conversion of the run.

    Args:
        run_settings: Run settings (``reuse_outputs``).
        catalog_book: Book of the run.

    Returns:
        True when reuse is on and the book's conversion already has its segments.
    """
    conversion_directory = book_folder(run_settings.run_directory, catalog_book.book_key) / CONVERSION_DIRECTORY
    return run_settings.reuse_outputs and (conversion_directory / SEGMENTS_FILE).exists()


def books_to_convert(run_settings: RunSettings) -> list[tuple[str, Path]]:
    """Books a run will convert, for its start-of-run estimate.

    Args:
        run_settings: Run settings.

    Returns:
        ``(book key, scan path)`` of every selected book without a reused conversion.
    """
    return [(catalog_book.book_key, catalog_book.scan_path) for catalog_book in selected_books(run_settings)
            if not is_reused(run_settings, catalog_book)]


def prepare_subset(catalog_book: CatalogBook, run_settings: RunSettings, document_preparer: DocumentPreparer) -> dict:
    """Write the sampled-page PDF of a book.

    Args:
        catalog_book: Book to sample.
        run_settings: Run settings.
        document_preparer: Turns DjVu scans into PDF (PDF passes through).

    Returns:
        Manifest entry ``{"book_key", "folder", "language", "pages": [original 0-based index per subset page]}``.
    """
    book_directory = book_folder(run_settings.run_directory, catalog_book.book_key)
    full_pdf = document_preparer.prepare_document(catalog_book.scan_path, book_directory / SOURCE_DIRECTORY)
    with pymupdf.open(full_pdf) as scan_document:
        page_count = scan_document.page_count
    page_indexes = sample_pages(catalog_book.book_key, page_count, run_settings.pages_per_book)
    write_page_subset(full_pdf, page_indexes, book_directory / SUBSET_FILE)
    return {"book_key": catalog_book.book_key, "folder": book_directory.name,
            "language": catalog_book.language_name, "pages": page_indexes}


def run_book(catalog_book: CatalogBook, run_settings: RunSettings, document_preparer: DocumentPreparer,
             book_converter: BookConverter) -> dict:
    """Sample and convert one book.

    Args:
        catalog_book: Book.
        run_settings: Run settings.
        document_preparer: Source format conversion.
        book_converter: Pipeline runner.

    Returns:
        Manifest entry with the conversion's ``exit_code`` (None when an existing output was reused).
    """
    manifest_entry = prepare_subset(catalog_book, run_settings, document_preparer)
    book_directory = run_settings.run_directory / manifest_entry["folder"]
    if is_reused(run_settings, catalog_book):
        return {**manifest_entry, "exit_code": None}
    exit_code = book_converter.convert_book(book_directory / SUBSET_FILE, catalog_book.language_code,
                                            book_directory / CONVERSION_DIRECTORY)
    return {**manifest_entry, "exit_code": exit_code}


def run_benchmark(run_settings: RunSettings, document_preparer: DocumentPreparer,
                  book_converter: BookConverter) -> Path:
    """Convert the sampled pages of every selected book and write the run manifest.

    Usage:
        Called by ``pdf2fb2-bench run``; score the run afterwards with ``benchmark_scoring.score_run``.

    Args:
        run_settings: Run settings.
        document_preparer: Source format conversion.
        book_converter: Pipeline runner.

    Returns:
        Path of ``run_manifest.json``.
    """
    run_settings.run_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = run_settings.run_directory / RUN_MANIFEST_FILE
    book_entries = []
    for catalog_book in selected_books(run_settings):
        book_entries.append(run_book(catalog_book, run_settings, document_preparer, book_converter))
        manifest_path.write_text(json.dumps({"split": run_settings.split_name, "books": book_entries},
                                            ensure_ascii=False, indent=1), encoding="utf-8")
    if not book_entries:
        manifest_path.write_text(json.dumps({"split": run_settings.split_name, "books": []}), encoding="utf-8")
    return manifest_path
