"""Unit tests of the benchmark: catalog filter, normalisation, alignment, locating, splits and labels."""

import csv
from pathlib import Path

import pytest

from pdf2fb2.benchmark.book_splits import TEST_SPLIT, VALID_SPLIT, locked_splits
from pdf2fb2.benchmark.corpus_catalog import CatalogBook, corpus_path, read_correct_books
from pdf2fb2.benchmark.edit_alignment import align_words, character_errors
from pdf2fb2.benchmark.page_sampling import sample_pages
from pdf2fb2.benchmark.page_scoring import BookScorer
from pdf2fb2.benchmark.text_normalization import comparison_words, normalize_text
from pdf2fb2.benchmark.verifier_export import token_labels
from pdf2fb2.text_verification.verify_units import verify_units

CATALOG_FIELDS = ["book_key", "language", "language_code", "image_pdf_path", "structured_text_path",
                  "parsed_text_path", "page_count_verified", "ocr_state"]


class ListReference:
    """Reference source over fixed pages (test double).

    Usage:
        Pass to ``BookScorer``.
    """

    def __init__(self, reference_pages: list[str]) -> None:
        """Store the pages.

        Args:
            reference_pages: Text per page.
        """
        self._page_words = [comparison_words(page_text) for page_text in reference_pages]

    def region_words(self, page_index: int) -> list[str]:
        """Words of a page.

        Args:
            page_index: Page.

        Returns:
            Normalised words.
        """
        return self._page_words[page_index]


def write_catalog(corpus_root: Path, catalog_rows: list[dict]) -> None:
    """Write a small catalog.

    Args:
        corpus_root: Corpus directory.
        catalog_rows: Rows.

    Returns:
        None.
    """
    with (corpus_root / "catalog.csv").open("w", encoding="utf-8", newline="") as catalog_file:
        catalog_writer = csv.DictWriter(catalog_file, fieldnames=CATALOG_FIELDS)
        catalog_writer.writeheader()
        catalog_writer.writerows(catalog_rows)


def catalog_row(book_key: str, ocr_state: str, image_path: str = "book.pdf") -> dict:
    """One catalog row.

    Args:
        book_key: Key.
        ocr_state: ``correct``, ``mixed`` or ``wrong``.
        image_path: Scan path.

    Returns:
        Row.
    """
    return {"book_key": book_key, "language": "english", "language_code": "en", "image_pdf_path": image_path,
            "structured_text_path": "book_djvu.xml", "parsed_text_path": "book.txt", "page_count_verified": "300",
            "ocr_state": ocr_state}


def test_only_correct_books_are_read(tmp_path) -> None:
    """Mixed and wrong rows are skipped; unsafe paths are rejected."""
    write_catalog(tmp_path, [catalog_row("good", "correct"), catalog_row("mixed", "mixed"),
                             catalog_row("wrong", "wrong")])
    assert [catalog_book.book_key for catalog_book in read_correct_books(tmp_path)] == ["good"]
    write_catalog(tmp_path, [catalog_row("escape", "correct", "../outside.pdf")])
    with pytest.raises(ValueError):
        read_correct_books(tmp_path)
    with pytest.raises(ValueError):
        corpus_path(tmp_path, "/etc/passwd")


def test_normalisation_removes_conventions_not_errors() -> None:
    """Hyphen breaks, markup, quotes and long s are unified; case and old letters are kept."""
    assert normalize_text("заклю¬ чается «Ѣ» '''жирный''' ## Глава") == 'заключается "Ѣ" жирный Глава'
    assert normalize_text("Maſter <i>said</i> — yes") == "Master said - yes"
    assert comparison_words("世界探偵 abc") == ["世", "界", "探", "偵", "abc"]


def test_alignment_labels_and_character_errors() -> None:
    """Substitutions, insertions and misses are separated; CER counts characters inside words."""
    output_words = ["the", "rnodern", "extra", "age"]
    reference_words = ["the", "modern", "age", "began"]
    aligned_pairs = align_words(output_words, reference_words)
    assert [aligned_pair.pair_label for aligned_pair in aligned_pairs] == [
        "correct", "misread", "inserted", "correct", "missed"]
    assert character_errors(aligned_pairs, output_words, reference_words) == 2 + 6 + 6


def test_page_located_without_running_head_and_notes_counted_as_inserted() -> None:
    """Running head and page number of the reference are skipped; a reader's note at the edge is inserted."""
    body_text = " ".join(f"word{word_number}" for word_number in range(60))
    book_scorer = BookScorer(ListReference([f"CHAPTER ONE 17 {body_text} catchword"]))
    clean_counts = book_scorer.score_page(0, body_text)
    assert clean_counts.error_rates()["wer"] == 0.0
    noisy_counts = book_scorer.score_page(0, f"scribbled margin note {body_text}")
    assert noisy_counts.inserted_words == 3 and noisy_counts.misread_words == 0


def test_splits_are_locked_and_keep_test_books(tmp_path) -> None:
    """Every language with three books gets a test and a valid book; a rerun returns the same split."""
    catalog_books = [CatalogBook(f"book-{book_number}", "english", "en", tmp_path, "djvu_xml", tmp_path, 100)
                     for book_number in range(6)]
    first_split = locked_splits(catalog_books, tmp_path)
    assert set(first_split.values()) >= {TEST_SPLIT, VALID_SPLIT}
    assert locked_splits(catalog_books, tmp_path) == first_split


def test_page_sample_is_reproducible_and_skips_covers() -> None:
    """The same key gives the same consecutive pages, all inside the book body."""
    sampled_pages = sample_pages("book-key", 300, 10)
    assert sampled_pages == sample_pages("book-key", 300, 10)
    assert sampled_pages == list(range(sampled_pages[0], sampled_pages[0] + 10))
    assert min(sampled_pages) >= 24 and max(sampled_pages) < 276


def test_token_labels_for_verifier_training() -> None:
    """Output units get correct / misread / inserted labels; pure markup is ignored."""
    page_units = [verify_unit.unit_text for verify_unit in verify_units("## The rnodern NOTE age")]
    assert token_labels(page_units, ["The", "modern", "age"]) == ["ignore", "correct", "misread", "inserted",
                                                                  "correct"]
