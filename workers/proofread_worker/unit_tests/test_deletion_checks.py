"""Unit tests of the document text and the deletion-test variants (no model needed)."""

from proofread_worker.deletion_checks import block_check, break_check, word_check
from proofread_worker.document_text import build_document_text
from worker_protocol.proofread_job_schema import TextBlock


def sample_document():
    """Paragraph, wrong heading, continuation.

    Returns:
        DocumentText.
    """
    return build_document_text((TextBlock("p", ("He", "rode", "to", "the")), TextBlock("heading", ("Running", "Head")),
                                TextBlock("p", ("old", "mission", "at", "dawn."))))


def test_document_offsets_point_at_words() -> None:
    """Every span slices its own word out of the joined text."""
    document_text = sample_document()
    assert document_text.full_text == "He rode to the\n\nRunning Head\n\nold mission at dawn."
    assert [document_text.full_text[word_span.start_offset:word_span.end_offset]
            for word_span in document_text.word_spans][4:6] == ["Running", "Head"]
    assert document_text.span_containing(document_text.full_text.index("\n")) is None


def test_word_check_scores_the_same_continuation() -> None:
    """Both variants are measured from the separator before the following words."""
    check_request = word_check(sample_document(), 1, 200, 3)
    (kept_text, kept_start), (dropped_text, dropped_start) = check_request.scored_variants
    assert kept_text[kept_start:] == " to the\n\nRunning"
    assert dropped_text[dropped_start:] == " to the\n\nRunning"
    assert "rode" not in dropped_text


def test_block_check_offers_separate_and_merged_variants() -> None:
    """Removing the heading yields a blank-line and a merged variant of the same continuation."""
    check_request = block_check(sample_document(), 1, 200, 2)
    variant_texts = [scored_text for scored_text, _ in check_request.scored_variants]
    assert variant_texts[0] == "He rode to the\n\nRunning Head\n\nold mission"
    assert variant_texts[1] == "He rode to the\n\nold mission"
    assert variant_texts[2] == "He rode to the old mission"


def test_break_check_compares_blank_line_with_space() -> None:
    """A paragraph break is tested against joining the paragraphs."""
    check_request = break_check(sample_document(), 2, 200, 2)
    assert [scored_text for scored_text, _ in check_request.scored_variants] == [
        "He rode to the\n\nRunning Head\n\nold mission", "He rode to the\n\nRunning Head old mission"]


def test_impossible_checks_are_skipped() -> None:
    """The last word and the last block cannot be tested."""
    document_text = sample_document()
    assert word_check(document_text, len(document_text.word_spans) - 1, 200, 3) is None
    assert block_check(document_text, 2, 200, 3) is None
    assert break_check(document_text, 0, 200, 3) is None
