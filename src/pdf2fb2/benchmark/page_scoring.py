"""Scores of one output page against its located reference, and their sums over books and languages."""

from dataclasses import asdict, dataclass

from pdf2fb2.benchmark.edit_alignment import (
    INSERTED_LABEL,
    MISREAD_LABEL,
    MISSED_LABEL,
    align_words,
    character_errors,
)
from pdf2fb2.benchmark.reference_locator import RegionIndex
from pdf2fb2.benchmark.reference_texts import ReferenceSource
from pdf2fb2.benchmark.text_normalization import comparison_words


@dataclass(frozen=True)
class PageCounts:
    """Error counts of one page (summable).

    Attributes:
        reference_characters: Characters of the located reference (spaces included).
        character_errors: Character edits.
        reference_words: Words of the located reference.
        misread_words: Substituted output words.
        inserted_words: Output words without reference word (text not in the book).
        missed_words: Reference words without output word.
        located_pages: 1 when the reference was found, else 0.
        unlocated_pages: 1 when no reference passage matched the output, else 0.
    """

    reference_characters: int = 0
    character_errors: int = 0
    reference_words: int = 0
    misread_words: int = 0
    inserted_words: int = 0
    missed_words: int = 0
    located_pages: int = 0
    unlocated_pages: int = 0

    def __add__(self, other_counts: "PageCounts") -> "PageCounts":
        """Sum two count sets.

        Args:
            other_counts: Counts to add.

        Returns:
            Field-wise sum.
        """
        return PageCounts(*(own_value + other_value for own_value, other_value
                            in zip(asdict(self).values(), asdict(other_counts).values(), strict=True)))

    def error_rates(self) -> dict:
        """Error rates of the counts.

        Returns:
            ``{"cer", "wer", "inserted_rate", "missed_rate", "pages", "unlocated_pages"}``; rates are 0..1+.
        """
        word_errors = self.misread_words + self.inserted_words + self.missed_words
        return {"cer": round(self.character_errors / max(self.reference_characters, 1), 4),
                "wer": round(word_errors / max(self.reference_words, 1), 4),
                "inserted_rate": round(self.inserted_words / max(self.reference_words, 1), 4),
                "missed_rate": round(self.missed_words / max(self.reference_words, 1), 4),
                "pages": self.located_pages, "unlocated_pages": self.unlocated_pages}


class BookScorer:
    """Scores the output pages of one book (keeps the transcript index between pages).

    Usage:
        Build per book with its reference source, call ``score_page`` per converted page.
    """

    def __init__(self, reference_source: ReferenceSource) -> None:
        """Bind the scorer to a reference.

        Args:
            reference_source: Reference reader of the book.
        """
        self._reference_source = reference_source
        self._cached_index: RegionIndex | None = None
        self._cached_page: int | None = None

    def region_index(self, page_index: int) -> RegionIndex:
        """Trigram index of a page's search region, reused while the page (or the transcript) stays the same.

        Args:
            page_index: 0-based page of the original scan.

        Returns:
            The region index.
        """
        if self._cached_index is not None and self._cached_page == page_index:
            return self._cached_index
        region_words = self._reference_source.region_words(page_index)
        if self._cached_index is None or self._cached_index.region_words is not region_words:
            self._cached_index = RegionIndex(region_words)
        self._cached_page = page_index
        return self._cached_index

    def located_reference(self, page_index: int, output_words: list[str]) -> list[str] | None:
        """Reference words of a page (or of one segment of it).

        Args:
            page_index: 0-based page of the original scan.
            output_words: Normalised output words.

        Returns:
            Located words, or None when the words cannot be located.
        """
        return self.region_index(page_index).locate_passage(output_words)

    def score_page(self, page_index: int, output_text: str) -> PageCounts:
        """Counts of one page.

        Args:
            page_index: 0-based page of the original scan.
            output_text: Pipeline text of the page.

        Returns:
            Error counts (only ``unlocated_pages`` set when no reference passage matches).
        """
        output_words = comparison_words(output_text)
        reference_words = self.located_reference(page_index, output_words)
        if reference_words is None:
            return PageCounts(unlocated_pages=1)
        aligned_pairs = align_words(output_words, reference_words, free_ends=True)
        label_counts = {pair_label: sum(aligned_pair.pair_label == pair_label for aligned_pair in aligned_pairs)
                        for pair_label in (MISREAD_LABEL, INSERTED_LABEL, MISSED_LABEL)}
        used_reference = [reference_words[aligned_pair.reference_index] for aligned_pair in aligned_pairs
                          if aligned_pair.reference_index is not None]
        return PageCounts(reference_characters=len(" ".join(used_reference)),
                          character_errors=character_errors(aligned_pairs, output_words, reference_words),
                          reference_words=len(used_reference), misread_words=label_counts[MISREAD_LABEL],
                          inserted_words=label_counts[INSERTED_LABEL], missed_words=label_counts[MISSED_LABEL],
                          located_pages=1)
