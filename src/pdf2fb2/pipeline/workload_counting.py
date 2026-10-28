"""Measures the workload sizes of a conversion from its page records, for the time estimates."""

from pdf2fb2.segments.label_roles import requires_ocr


def count_text_pages(page_records: list[dict]) -> int:
    """Pages with at least one segment to OCR.

    Args:
        page_records: Segmented page records.

    Returns:
        Number of text pages.
    """
    return sum(any(requires_ocr(page_segment["label"]) for page_segment in page_record["segments"])
               for page_record in page_records)


def count_text_tokens(page_records: list[dict]) -> int:
    """Words read by OCR.

    Args:
        page_records: OCR'd page records (segments carry ``text``).

    Returns:
        Number of whitespace-separated words in the OCR'd segments.
    """
    return sum(len(str(page_segment.get("text") or "").split())
               for page_record in page_records for page_segment in page_record["segments"]
               if requires_ocr(page_segment["label"]))


def count_inspected_tokens(book_statistics: dict, token_count: int) -> int:
    """Words the proofreading language model inspected.

    Args:
        book_statistics: FB2 statistics with the proofreading summary (``flagged_words`` in flagged mode).
        token_count: Words read by OCR (all of them are inspected in full mode).

    Returns:
        Number of inspected words.
    """
    flagged_words = book_statistics.get("flagged_words")
    return flagged_words if isinstance(flagged_words, int) and flagged_words >= 0 else token_count
