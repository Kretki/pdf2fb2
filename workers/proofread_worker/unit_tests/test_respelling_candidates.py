"""Unit tests of the respelling candidates and their rewrite variants (no model needed)."""

from proofread_worker.document_lexicon import DocumentLexicon, edit_distance
from proofread_worker.document_text import build_document_text
from proofread_worker.job_execution import respell_requests
from proofread_worker.respelling_candidates import candidate_spellings, confusion_candidates
from worker_protocol.proofread_job_schema import ProofreadSettings, TextBlock


def test_edit_distance_counts_swaps_and_caps() -> None:
    """Neighbour swaps cost one edit; distances above the limit are capped."""
    assert edit_distance("form", "from", 2) == 1
    assert edit_distance("kitten", "sitting", 2) == 3
    assert edit_distance("слово", "слово", 2) == 0


def test_confusion_pairs_cover_the_typical_misreadings() -> None:
    """с/е, ы/ые, rn/m and long s read as f are proposed; upper case is derived."""
    assert "время" in confusion_candidates("врсмя")
    assert "новые" in confusion_candidates("новы")
    assert "modern" in confusion_candidates("rnodern")
    assert "Sunday" in confusion_candidates("Funday")
    assert "сила" in confusion_candidates("cила")  # Latin c in a Cyrillic word


def test_book_forms_outrank_unseen_confusions() -> None:
    """A form the book prints twice comes before a confusion that never occurs; punctuation is kept."""
    document_lexicon = DocumentLexicon(["mission", "mission", "the", "the", "the"])
    spellings = [respell_candidate.spelling
                 for respell_candidate in candidate_spellings("misslon,", document_lexicon, 3)]
    assert spellings[0] == "mission,"
    assert len(spellings) == 3


def test_frequent_and_symbol_words_are_not_respelled() -> None:
    """Forms printed three times and words without letters get no candidates."""
    document_lexicon = DocumentLexicon(["rnodern"] * 3)
    assert candidate_spellings("rnodern", document_lexicon, 6) == []
    assert candidate_spellings("1856", document_lexicon, 6) == []


def test_respell_request_scores_each_spelling_on_the_same_continuation() -> None:
    """Every variant ends with the same following words and the target carries spellings and costs."""
    document_text = build_document_text((TextBlock("p", ("The", "rnodern", "mission", "of", "the", "modern", "age")),))
    proofread_settings = ProofreadSettings(-9.0, 200, 2, 10, max_respell_checks=5, respell_candidates=2)
    (check_request,) = respell_requests(document_text, proofread_settings, [1])
    assert check_request.check_kind == "respell"
    assert check_request.variant_names == ("kept", "alt_a", "alt_b")
    assert check_request.check_target["spellings"]["alt_a"] == "modern"
    measured_texts = [variant_text[measure_from:] for variant_text, measure_from in check_request.scored_variants]
    assert measured_texts[0] == " rnodern mission of"
    assert all(measured_text.endswith(" mission of") for measured_text in measured_texts)


def test_respelling_is_off_without_a_budget() -> None:
    """Old jobs (max_respell_checks = 0) run no respelling tests."""
    document_text = build_document_text((TextBlock("p", ("rnodern", "times")),))
    assert respell_requests(document_text, ProofreadSettings(-9.0, 200, 2, 10), [0]) == []
