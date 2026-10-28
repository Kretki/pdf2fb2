"""Unit tests of the benchmark: catalog filter, normalisation, alignment, locating, splits and labels."""

import csv
import json
from pathlib import Path

import pytest

from pdf2fb2.benchmark.book_splits import TEST_SPLIT, VALID_SPLIT, locked_splits
from pdf2fb2.benchmark.corpus_catalog import CatalogBook, corpus_path, read_correct_books
from pdf2fb2.benchmark.edit_alignment import align_words, character_errors
from pdf2fb2.benchmark.label_reliability import page_is_reliable, segment_is_reliable
from pdf2fb2.benchmark.page_sampling import sample_pages
from pdf2fb2.benchmark.page_scoring import BookScorer
from pdf2fb2.benchmark.rendered_export import export_rendered
from pdf2fb2.benchmark.text_normalization import comparison_words, normalize_text
from pdf2fb2.benchmark.verifier_export import page_examples, token_labels
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


def test_convention_differences_are_ignored() -> None:
    """Misreads that differ only by case, punctuation or a footnote marker are not OCR errors."""
    assert token_labels(["Пролет", "ассоциациям¹)", "rnodern"], ["Пролет.", "ассоциациям", "modern"]) == \
        ["ignore", "ignore", "misread"]


def test_unreliable_pages_and_segments_are_detected() -> None:
    """Pages that barely match and segments that are mostly inserted are reference noise."""
    assert page_is_reliable(["correct"] * 4 + ["inserted"] * 6)
    assert not page_is_reliable(["correct"] * 3 + ["inserted"] * 7)
    assert not segment_is_reliable(["correct", "inserted", "inserted", "ignore", "misread", "inserted"])
    assert segment_is_reliable(["inserted", "inserted", "correct"])


def test_segments_are_aligned_against_their_own_passage() -> None:
    """Two blocks read in the opposite order of the reference are both labelled correct."""
    first_block = "the quick brown fox jumps over the lazy dog today"
    second_block = "a second block of plain text stands here alone now"
    book_scorer = BookScorer(ListReference([f"{first_block} {second_block}"]))
    page_record = {"segments": [{"label": "text", "text": second_block}, {"label": "text", "text": first_block}]}
    catalog_book = CatalogBook("english-book", "english", "en", Path("scan.pdf"), "transcript",
                               Path("reference.txt"), 1)
    page_rows = list(page_examples(catalog_book, book_scorer, page_record, 0))
    assert [set(page_row["labels"]) for page_row in page_rows] == [{"correct"}, {"correct"}]


def test_loops_and_proofread_pages_are_kept() -> None:
    """A repetition loop or a hand-proofread reference keeps pages and segments the noise filter would drop."""
    poor_labels = ["correct"] * 2 + ["inserted"] * 8
    assert not page_is_reliable(poor_labels)
    assert page_is_reliable(poor_labels, proofread_reference=True)
    assert page_is_reliable(poor_labels, page_units=["a", "b", "x", "y", "x", "y", "x", "y"])
    assert segment_is_reliable(["inserted"] * 6, segment_units=["敵"] * 6)
    assert not segment_is_reliable(["inserted"] * 6, segment_units=list("abcdef"))


def test_exported_examples_carry_page_context(tmp_path) -> None:
    """Exported examples carry the neighbouring segments as context (as pipeline jobs do)."""
    first_block = "the quick brown fox jumps over the lazy dog today"
    second_block = "a second block of plain text stands here alone now"
    book_scorer = BookScorer(ListReference([f"{first_block} {second_block}"]))
    page_record = {"page": 1, "segments": [{"label": "text", "text": first_block},
                                           {"label": "text", "text": second_block}]}
    catalog_book = CatalogBook("english-book", "english", "en", Path("scan.pdf"), "transcript",
                               Path("reference.txt"), 1)
    page_rows = list(page_examples(catalog_book, book_scorer, page_record, 0))
    assert page_rows[0]["context_after"][:2] == ["a", "second"] and "context_before" not in page_rows[0]
    assert page_rows[1]["context_before"][-1] == "today"


def test_rendered_pages_are_labelled_against_their_words(tmp_path) -> None:
    """The pipeline output of a rendered page is labelled against the exact words of the manifest."""
    language_directory = tmp_path / "rendered_train" / "english"
    (language_directory / "conversion").mkdir(parents=True)
    (language_directory / "manifest.json").write_text(json.dumps(
        {"language": "english", "pages": [{"book_key": "b", "words": "the modern age went fast".split()}]}))
    (language_directory / "conversion" / "segments.json").write_text(json.dumps(
        [{"page": 1, "segments": [{"label": "text", "text": "the rnodern age went fast"}]}]))
    export_rendered(tmp_path, "train")
    exported_line = json.loads((tmp_path / "rendered_train.jsonl").read_text().splitlines()[0])
    assert exported_line["labels"] == ["correct", "misread", "correct", "correct", "correct"]
    assert exported_line["book_key"] == "rendered-english"
