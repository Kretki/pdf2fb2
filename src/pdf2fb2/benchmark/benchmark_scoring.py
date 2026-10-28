"""Score a benchmark run: page counts per book, sums per language, at the ``ocr`` and ``final`` levels."""

import json
from collections import defaultdict
from pathlib import Path

from pdf2fb2.benchmark.benchmark_runner import CONVERSION_DIRECTORY, RUN_MANIFEST_FILE
from pdf2fb2.benchmark.corpus_catalog import CatalogBook, read_correct_books
from pdf2fb2.benchmark.page_outputs import final_page_texts, ocr_page_texts
from pdf2fb2.benchmark.page_scoring import BookScorer, PageCounts
from pdf2fb2.benchmark.reference_texts import reference_source

OCR_LEVEL = "ocr"
FINAL_LEVEL = "final"


def book_counts(catalog_book: CatalogBook, book_directory: Path, original_pages: list[int]) -> dict[str, PageCounts]:
    """Summed page counts of one converted book.

    Args:
        catalog_book: Book.
        book_directory: Book folder of the run.
        original_pages: Original 0-based page of every subset page.

    Returns:
        ``{"ocr": counts, "final": counts}``.
    """
    conversion_directory = book_directory / CONVERSION_DIRECTORY
    level_texts = {OCR_LEVEL: ocr_page_texts(conversion_directory),
                   FINAL_LEVEL: final_page_texts(conversion_directory / "subset.fb2")}
    book_scorer = BookScorer(reference_source(catalog_book))
    summed_counts = {}
    for level_name, page_texts in level_texts.items():
        summed_counts[level_name] = sum((book_scorer.score_page(original_page, page_texts.get(page_number, ""))
                                         for page_number, original_page in enumerate(original_pages, start=1)),
                                        PageCounts())
    return summed_counts


def score_run(run_directory: Path, corpus_root: Path) -> dict:
    """Score every book of a run and write ``report.json`` and ``report.md``.

    Usage:
        Called by ``pdf2fb2-bench score`` (and by ``run`` when it finishes).

    Args:
        run_directory: Run output directory with ``run_manifest.json``.
        corpus_root: Corpus directory.

    Returns:
        The report document.
    """
    run_manifest = json.loads((run_directory / RUN_MANIFEST_FILE).read_text(encoding="utf-8"))
    books_by_key = {catalog_book.book_key: catalog_book for catalog_book in read_correct_books(corpus_root)}
    language_totals: dict[str, dict[str, PageCounts]] = defaultdict(lambda: defaultdict(PageCounts))
    book_reports = []
    for book_entry in run_manifest["books"]:
        catalog_book = books_by_key.get(book_entry["book_key"])
        if catalog_book is None:
            continue
        counts_by_level = book_counts(catalog_book, run_directory / book_entry["folder"], book_entry["pages"])
        for level_name, level_counts in counts_by_level.items():
            language_totals[catalog_book.language_name][level_name] += level_counts
            language_totals["ALL"][level_name] += level_counts
        book_reports.append({"book_key": catalog_book.book_key, "language": catalog_book.language_name,
                             "exit_code": book_entry.get("exit_code"),
                             **{level_name: level_counts.error_rates()
                                for level_name, level_counts in counts_by_level.items()}})
    report_document = {"split": run_manifest["split"], "books": book_reports,
                       "languages": {language_name: {level_name: level_counts.error_rates()
                                                     for level_name, level_counts in level_totals.items()}
                                     for language_name, level_totals in sorted(language_totals.items())}}
    (run_directory / "report.json").write_text(json.dumps(report_document, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    (run_directory / "report.md").write_text(markdown_report(report_document), encoding="utf-8")
    return report_document


def markdown_report(report_document: dict) -> str:
    """Human-readable summary of a report.

    Args:
        report_document: Result of ``score_run``.

    Returns:
        Markdown with one table per level (rows: languages, then ALL).
    """
    report_lines = [f"# Benchmark report ({report_document['split']} split)", ""]
    for level_name in (OCR_LEVEL, FINAL_LEVEL):
        report_lines += [f"## {level_name} level", "",
                         "| language | pages | unlocated | CER | WER | inserted | missed |",
                         "|---|---|---|---|---|---|---|"]
        for language_name, level_rates in report_document["languages"].items():
            level_values = level_rates.get(level_name, {})
            report_lines.append(f"| {language_name} | {level_values.get('pages', 0)} | "
                                f"{level_values.get('unlocated_pages', 0)} | {level_values.get('cer', 0):.2%} | "
                                f"{level_values.get('wer', 0):.2%} | {level_values.get('inserted_rate', 0):.2%} | "
                                f"{level_values.get('missed_rate', 0):.2%} |")
        report_lines.append("")
    return "\n".join(report_lines)
