"""Training and evaluation data of the token verifier, written as JSON lines into the data directory.

* ``references_<split>.jsonl``: chunks of reference text of the split's correct books of every catalog language
  (``{"book_key", "language", "words"}``); the verifier worker corrupts them into synthetic examples. Every
  language gets the same chunk budget, shared by its books, so languages with many books do not dominate.
* ``aligned_<split>.jsonl``: real pipeline output of a benchmark run, one line per text segment
  (``{"book_key", "language", "page", "words", "labels"}``). Every verifier unit of the segment's OCR text
  (``text_verification.verify_units``: whitespace tokens, CJK characters) is labelled by aligning the page with
  its located reference: ``correct``, ``misread``, ``inserted`` (text
  that is not in the book), or ``ignore`` for pure markup tokens and convention-only differences. Every segment
  is aligned against its own passage of the page region; pages and segments whose labels are mostly reference
  noise are left out (``label_reliability``).
"""

import hashlib
import itertools
import json
import math
import random
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from pdf2fb2.benchmark.benchmark_runner import CONVERSION_DIRECTORY, RUN_MANIFEST_FILE
from pdf2fb2.benchmark.book_splits import locked_splits
from pdf2fb2.benchmark.corpus_catalog import TRANSCRIPT_REFERENCE, CatalogBook, read_correct_books
from pdf2fb2.benchmark.edit_alignment import MISREAD_LABEL, MISSED_LABEL, AlignedPair, align_words
from pdf2fb2.benchmark.export_evidence import CropOptions, page_evidence, page_segment_units
from pdf2fb2.benchmark.label_reliability import is_convention_difference, page_is_reliable, segment_is_reliable
from pdf2fb2.benchmark.page_outputs import SEGMENTS_FILE
from pdf2fb2.benchmark.page_sampling import sample_pages
from pdf2fb2.benchmark.page_scoring import BookScorer
from pdf2fb2.benchmark.reference_texts import reference_source
from pdf2fb2.benchmark.text_normalization import token_form
from pdf2fb2.segments.label_roles import is_book_text

IGNORE_LABEL = "ignore"
CHUNK_WORDS = 200
MIN_CHUNK_WORDS = 20
MIN_LOCATED_FORMS = 5    # shorter segments keep their page-level labels


def reference_chunks(catalog_book: CatalogBook, chunk_count: int) -> Iterator[list[str]]:
    """Reference word chunks of one book.

    Args:
        catalog_book: Book.
        chunk_count: Pages (paged references) or random chunks (transcripts) to take.

    Yields:
        Lists of normalised words (pages longer than ``CHUNK_WORDS`` are cut into pieces).
    """
    source_reader = reference_source(catalog_book)
    if catalog_book.reference_kind == TRANSCRIPT_REFERENCE:
        book_words = source_reader.region_words(0)
        seed_value = int(hashlib.sha256(catalog_book.book_key.encode("utf-8")).hexdigest()[:16], 16)
        chunk_generator = random.Random(seed_value)  # nosec B311  # noqa: S311 - reproducible sampling, not cryptography
        for _ in range(chunk_count if len(book_words) > CHUNK_WORDS else 0):
            start_index = chunk_generator.randrange(len(book_words) - CHUNK_WORDS)
            yield book_words[start_index:start_index + CHUNK_WORDS]
        return
    # A page region also holds the next page, so every second page of a run covers it without overlap.
    for page_index in sample_pages(catalog_book.book_key, catalog_book.page_count, 2 * chunk_count)[::2]:
        page_words = source_reader.region_words(page_index)
        for start_index in range(0, len(page_words), CHUNK_WORDS):
            if len(page_words) - start_index >= MIN_CHUNK_WORDS:
                yield page_words[start_index:start_index + CHUNK_WORDS]


def export_references(corpus_root: Path, data_directory: Path, split_name: str, chunks_per_language: int) -> Path:
    """Write ``references_<split>.jsonl`` with every catalog language of the split.

    Usage:
        Called by ``pdf2fb2-bench export-references``.

    Args:
        corpus_root: Corpus directory.
        data_directory: Output directory (also holds ``splits.json``).
        split_name: ``train``, ``valid`` or ``test``.
        chunks_per_language: Pages or chunks per language, divided among its books (at least one per book).

    Returns:
        Path of the written file.
    """
    catalog_books = read_correct_books(corpus_root)
    book_splits = locked_splits(catalog_books, data_directory)
    split_books = [catalog_book for catalog_book in catalog_books
                   if book_splits[catalog_book.book_key] == split_name]
    language_sizes = Counter(catalog_book.language_name for catalog_book in split_books)
    output_path = data_directory / f"references_{split_name}.jsonl"
    with output_path.open("w", encoding="utf-8") as output_file:
        for catalog_book in split_books:
            book_chunks = math.ceil(chunks_per_language / language_sizes[catalog_book.language_name])
            for chunk_words in itertools.islice(reference_chunks(catalog_book, book_chunks), book_chunks):
                output_file.write(json.dumps({"book_key": catalog_book.book_key,
                                              "language": catalog_book.language_name, "words": chunk_words},
                                             ensure_ascii=False) + "\n")
    return output_path


def pair_label(aligned_pair: AlignedPair, output_form: str, reference_words: list[str]) -> str:
    """Training label of one aligned output word.

    Args:
        aligned_pair: Alignment step with an output word.
        output_form: Normalised form of that output word.
        reference_words: Located reference words.

    Returns:
        The alignment label, or ``ignore`` for a misread that differs only by case, punctuation or a
        footnote marker (reference conventions, not OCR errors).
    """
    if aligned_pair.pair_label == MISREAD_LABEL and is_convention_difference(
            output_form, reference_words[aligned_pair.reference_index]):
        return IGNORE_LABEL
    return aligned_pair.pair_label


def token_labels(page_tokens: list[str], reference_words: list[str]) -> list[str]:
    """Label every raw output token of a page or segment.

    Args:
        page_tokens: Verifier units of the OCR text.
        reference_words: Located reference words.

    Returns:
        One label per token.
    """
    token_forms = [token_form(raw_token) for raw_token in page_tokens]
    compared_positions = [token_index for token_index, form_text in enumerate(token_forms) if form_text]
    page_labels = [IGNORE_LABEL] * len(page_tokens)
    for aligned_pair in align_words([token_forms[token_index] for token_index in compared_positions],
                                    reference_words, free_ends=True):
        if aligned_pair.pair_label != MISSED_LABEL:
            token_index = compared_positions[aligned_pair.output_index]
            page_labels[token_index] = pair_label(aligned_pair, token_forms[token_index], reference_words)
    return page_labels


def segment_labels(book_scorer: BookScorer, original_page: int, segment_tokens: list[str],
                   page_labels: list[str]) -> list[str]:
    """Labels of one segment, aligned against its own passage of the page's reference region.

    Usage:
        Blocks may be read in a different order than the reference lists them (columns, captions, vertical
        Japanese text), which a single page-level alignment turns into ``inserted`` runs.

    Args:
        book_scorer: Reference locator of the book.
        original_page: Original 0-based page index.
        segment_tokens: Verifier units of the segment.
        page_labels: The segment's slice of the page-level labels (used when the segment cannot be located).

    Returns:
        One label per unit.
    """
    segment_forms = [form_text for form_text in map(token_form, segment_tokens) if form_text]
    segment_reference = book_scorer.located_reference(original_page, segment_forms) \
        if len(segment_forms) >= MIN_LOCATED_FORMS else None
    return page_labels if segment_reference is None else token_labels(segment_tokens, segment_reference)


def page_examples(catalog_book: CatalogBook, book_scorer: BookScorer, page_record: dict, original_page: int,
                  crop_options: CropOptions | None = None) -> Iterator[dict]:
    """Labelled segments of one converted page, with their page evidence.

    Args:
        catalog_book: Book.
        book_scorer: Reference locator of the book (``BookScorer``, or ``ManifestReference`` for rendered pages).
        page_record: Page record of ``segments.json``.
        original_page: Original 0-based page index.
        crop_options: Word-crop settings (None: no crops).

    Yields:
        One example per reliable text segment (nothing when the page cannot be located or its reference
        matches too poorly, see ``label_reliability``).
    """
    segment_entries = page_segment_units([page_segment for page_segment in page_record["segments"]
                                          if is_book_text(page_segment)])
    segment_tokens = [[verify_unit.unit_text for verify_unit in segment_entry.segment_units]
                      for segment_entry in segment_entries]
    page_tokens = [raw_token for token_list in segment_tokens for raw_token in token_list]
    reference_words = book_scorer.located_reference(original_page, [form_text for form_text in map(
        token_form, page_tokens) if form_text])
    if reference_words is None:
        return
    page_labels = token_labels(page_tokens, reference_words)
    if not page_is_reliable(page_labels, catalog_book.is_proofread, page_tokens):
        return
    evidence_list = page_evidence(page_record, segment_entries,
                                  crop_options or CropOptions(None, catalog_book.book_key))
    read_offset = 0
    for token_list, segment_evidence in zip(segment_tokens, evidence_list, strict=True):
        unit_labels = segment_labels(book_scorer, original_page, token_list,
                                     page_labels[read_offset:read_offset + len(token_list)])
        read_offset += len(token_list)
        if token_list and segment_is_reliable(unit_labels, catalog_book.is_proofread, token_list):
            yield {"book_key": catalog_book.book_key, "language": catalog_book.language_name,
                   "page": original_page, "words": token_list, "labels": unit_labels,
                   **{evidence_key: evidence_value for evidence_key, evidence_value in segment_evidence.items()
                      if evidence_value}}


def export_aligned(run_directory: Path, corpus_root: Path, data_directory: Path, write_crops: bool = False) -> Path:
    """Write ``aligned_<split>.jsonl`` from a benchmark run.

    Usage:
        Called by ``pdf2fb2-bench export-aligned`` after ``pdf2fb2-bench run`` on that split.

    Args:
        run_directory: Run directory with ``run_manifest.json``.
        corpus_root: Corpus directory.
        data_directory: Output directory.
        write_crops: Write word crops (under ``data_directory/verify_crops``) for the image-and-text features.

    Returns:
        Path of the written file.
    """
    run_manifest = json.loads((run_directory / RUN_MANIFEST_FILE).read_text(encoding="utf-8"))
    books_by_key = {catalog_book.book_key: catalog_book for catalog_book in read_correct_books(corpus_root)}
    output_path = data_directory / f"aligned_{run_manifest['split']}.jsonl"
    data_directory.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for book_entry in run_manifest["books"]:
            catalog_book = books_by_key.get(book_entry["book_key"])
            segments_path = run_directory / book_entry["folder"] / CONVERSION_DIRECTORY / SEGMENTS_FILE
            if catalog_book is None or not segments_path.exists():
                continue
            book_scorer = BookScorer(reference_source(catalog_book))
            for page_record in json.loads(segments_path.read_text(encoding="utf-8")):
                original_page = book_entry["pages"][page_record["page"] - 1]
                crop_options = CropOptions(data_directory if write_crops else None,
                                           f"{run_manifest['split']}_{catalog_book.book_key}")
                for page_example in page_examples(catalog_book, book_scorer, page_record, original_page,
                                                  crop_options):
                    output_file.write(json.dumps(page_example, ensure_ascii=False) + "\n")
    return output_path
