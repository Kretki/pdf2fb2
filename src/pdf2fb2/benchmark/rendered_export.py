"""Run the pipeline on rendered pages and label its output against the exactly known words.

* ``run-rendered``: converts ``rendered_<split>/<language>/rendered.pdf`` with the real pipeline (the language
  code comes from the catalog books of that language);
* ``export-rendered``: aligns every converted page with the words in the manifest and writes
  ``rendered_<split>.jsonl`` (same format and page evidence as ``aligned_<split>.jsonl``). The reference is exact,
  so no page or segment is filtered.
"""

import json
from pathlib import Path

from pdf2fb2.benchmark.benchmark_runner import BookConverter
from pdf2fb2.benchmark.corpus_catalog import CatalogBook, read_correct_books
from pdf2fb2.benchmark.export_evidence import CropOptions
from pdf2fb2.benchmark.page_outputs import SEGMENTS_FILE
from pdf2fb2.benchmark.rendered_pages import MANIFEST_FILE, RENDERED_PDF
from pdf2fb2.benchmark.text_normalization import comparison_words
from pdf2fb2.benchmark.verifier_export import page_examples

CONVERSION_DIRECTORY = "conversion"
EXACT_REFERENCE = "hand_proofread_rendered"


class ManifestReference:
    """Exact words of rendered pages, used directly as each page's located reference.

    Usage:
        Passed to ``page_examples`` in place of a ``BookScorer``: a rendered page needs no locating, its region is
        exactly its own words (the alignment skips the words of other segments at its free ends).
    """

    def __init__(self, manifest_pages: list[dict]) -> None:
        """Normalise the words of every page.

        Args:
            manifest_pages: Manifest entries ``{"book_key", "words"}``.
        """
        self._page_words = [comparison_words(" ".join(manifest_page["words"])) for manifest_page in manifest_pages]

    def located_reference(self, page_index: int, output_words: list[str]) -> list[str] | None:
        """Words of a rendered page (same interface as ``BookScorer.located_reference``).

        Args:
            page_index: 0-based page.
            output_words: Unused: the page is known exactly.

        Returns:
            Normalised words, or None for a page out of range or without words.
        """
        page_words = self._page_words[page_index] if 0 <= page_index < len(self._page_words) else []
        return page_words or None


def language_codes(corpus_root: Path) -> dict[str, str]:
    """Pipeline language code of every corpus language.

    Args:
        corpus_root: Corpus directory.

    Returns:
        Language name -> ``--lang`` code.
    """
    return {catalog_book.language_name: catalog_book.language_code for catalog_book in read_correct_books(corpus_root)}


def run_rendered(data_directory: Path, split_name: str, corpus_root: Path, book_converter: BookConverter) -> int:
    """Convert every rendered language PDF of a split.

    Usage:
        Called by ``pdf2fb2-bench run-rendered`` after ``render-references``.

    Args:
        data_directory: ``benchmark_data``.
        split_name: Rendered split.
        corpus_root: Corpus directory (language codes).
        book_converter: Pipeline runner.

    Returns:
        Number of languages whose conversion failed.
    """
    codes_by_language = language_codes(corpus_root)
    failure_count = 0
    for language_directory in sorted((data_directory / f"rendered_{split_name}").iterdir()):
        pdf_path = language_directory / RENDERED_PDF
        if not pdf_path.exists() or language_directory.name not in codes_by_language:
            continue
        exit_code = book_converter.convert_book(pdf_path, codes_by_language[language_directory.name],
                                                language_directory / CONVERSION_DIRECTORY)
        failure_count += exit_code != 0
        print(f"converted rendered {language_directory.name}: exit code {exit_code}", flush=True)
    return failure_count


def language_examples(language_directory: Path, split_name: str, data_directory: Path,
                      write_crops: bool) -> list[dict]:
    """Labelled examples of one rendered language.

    Args:
        language_directory: ``rendered_<split>/<language>``.
        split_name: Split.
        data_directory: ``benchmark_data`` (crop root).
        write_crops: Write word crops.

    Returns:
        Example lines.
    """
    segments_path = language_directory / CONVERSION_DIRECTORY / SEGMENTS_FILE
    manifest = json.loads((language_directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    if not segments_path.exists() or not manifest["pages"]:
        return []
    pseudo_book = CatalogBook(f"rendered-{manifest['language']}", manifest["language"], "", language_directory /
                              RENDERED_PDF, "rendered", language_directory / MANIFEST_FILE, len(manifest["pages"]),
                              EXACT_REFERENCE)
    book_scorer = ManifestReference(manifest["pages"])
    crop_options = CropOptions(data_directory if write_crops else None, f"rendered_{split_name}_{manifest['language']}")
    return [page_example for page_record in json.loads(segments_path.read_text(encoding="utf-8"))
            for page_example in page_examples(pseudo_book, book_scorer, page_record, page_record["page"] - 1,
                                              crop_options)]


def export_rendered(data_directory: Path, split_name: str, write_crops: bool = False) -> Path:
    """Write ``rendered_<split>.jsonl``.

    Usage:
        Called by ``pdf2fb2-bench export-rendered`` after ``run-rendered``.

    Args:
        data_directory: ``benchmark_data``.
        split_name: Rendered split.
        write_crops: Write word crops for the image-and-text features.

    Returns:
        Path of the written file.
    """
    output_path = data_directory / f"rendered_{split_name}.jsonl"
    with output_path.open("w", encoding="utf-8") as output_file:
        for language_directory in sorted((data_directory / f"rendered_{split_name}").iterdir()):
            if (language_directory / MANIFEST_FILE).exists():
                for page_example in language_examples(language_directory, split_name, data_directory, write_crops):
                    output_file.write(json.dumps(page_example, ensure_ascii=False) + "\n")
    return output_path
