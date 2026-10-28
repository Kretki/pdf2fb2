"""Unit tests of the respelling decisions and how they are applied to the flow."""

from pdf2fb2.proofreading.edit_decisions import decide_edits
from pdf2fb2.proofreading.flow_blocks import collect_flow_blocks
from pdf2fb2.proofreading.flow_editing import apply_flow_edits
from pdf2fb2.proofreading.word_respelling import decide_respellings, respelled_raw_token


def respell_entry(kept_score: float, candidate_score: float, candidate_cost: float = 0.5) -> dict:
    """Worker respelling entry for word 1 of block 0.

    Args:
        kept_score: Score of the spelling read.
        candidate_score: Score of the single candidate.
        candidate_cost: Edit cost of the candidate.

    Returns:
        Result entry.
    """
    return {"block": 0, "word": 1, "spellings": {"alt_a": "modern"}, "costs": {"alt_a": candidate_cost},
            "kept": kept_score, "alt_a": candidate_score}


def worker_result(respell_entries: list[dict], word_count: int) -> dict:
    """Worker result with one block and no other tests.

    Args:
        respell_entries: Respelling entries.
        word_count: Words of the block.

    Returns:
        Result document.
    """
    return {"blocks": [{"words": [{"tokens": [["w", -2.0]], "min_logprob": -2.0, "mean_logprob": -2.0}
                                  for _ in range(word_count)]}],
            "word_checks": [], "block_checks": [], "break_checks": [], "rewrite_checks": [],
            "respell_checks": respell_entries}


def test_candidate_must_beat_margin_after_edit_penalty() -> None:
    """A gain of 2.5 nats does not pay for a 0.5 edit (penalty 1 + margin 2); a gain of 4 does."""
    assert decide_respellings(worker_result([respell_entry(-20.0, -17.5)], 3), frozenset())[0] == []
    respellings, decision_notes = decide_respellings(worker_result([respell_entry(-20.0, -16.0)], 3), frozenset())
    assert respellings[0].spelling == "modern"
    assert decision_notes[0]["action"] == "respell_word"


def test_deleted_words_are_not_respelled() -> None:
    """A word removed as noise keeps no respelling."""
    assert decide_respellings(worker_result([respell_entry(-20.0, -10.0)], 3), frozenset({(0, 1)}))[0] == []


def test_markup_and_punctuation_are_kept() -> None:
    """Style tags and trailing punctuation survive the replacement; words split by tags are skipped."""
    assert respelled_raw_token("<i>rnodern</i>,", "rnodern,", "modern,") == "<i>modern</i>,"
    assert respelled_raw_token("<i>rno</i>dern", "rnodern", "modern") is None


def test_respelling_is_applied_to_the_flow() -> None:
    """The accepted spelling replaces the token in the edited flow."""
    flow_items = [{"kind": "p", "text": "The <b>rnodern</b> age.", "style": "p", "list": False, "pages": {1}}]
    flow_blocks = collect_flow_blocks(flow_items)
    flow_edits = decide_edits(flow_blocks, worker_result([respell_entry(-20.0, -10.0)], 3))
    assert apply_flow_edits(flow_items, flow_blocks, flow_edits)[0]["text"] == "The <b>modern</b> age."
