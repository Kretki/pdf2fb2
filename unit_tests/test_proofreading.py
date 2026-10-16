"""Unit tests of the proofreading decisions and flow edits (worker results are synthetic)."""

from pdf2fb2.proofreading.check_selection import block_check_indexes, break_check_indexes
from pdf2fb2.proofreading.edit_decisions import decide_edits, deletable_token, noise_log_odds
from pdf2fb2.proofreading.flow_blocks import collect_flow_blocks
from pdf2fb2.proofreading.flow_editing import apply_flow_edits


def sample_flow() -> list[dict]:
    """A sentence broken by a page break and a running head, plus a stray symbol line.

    Returns:
        Flow items.
    """
    return [{"kind": "page", "page": 1, "printed": None},
            {"kind": "p", "text": "He rode down the long valley to the", "style": "p", "list": False, "pages": {1}},
            {"kind": "page", "page": 2, "printed": None},
            {"kind": "heading", "level": 1, "text": "Running Head"},
            {"kind": "p", "text": "old mission at <i>dawn</i> and waited there.", "style": "p", "list": False,
             "pages": {2}},
            {"kind": "p", "text": "||", "style": "p", "list": False, "pages": {2}},
            {"kind": "p", "text": "Then he slept a long and quiet sleep.", "style": "p", "list": False, "pages": {2}}]


def word_result(word_count: int, token_logprob: float = -2.0) -> dict:
    """Worker block entry with one token per word.

    Args:
        word_count: Number of words.
        token_logprob: Log-probability of every token.

    Returns:
        Block entry.
    """
    return {"words": [{"tokens": [["w", token_logprob]], "min_logprob": token_logprob,
                       "mean_logprob": token_logprob} for _ in range(word_count)]}


def sample_result() -> dict:
    """Worker result: the heading and the symbol line read better removed.

    Returns:
        Result document.
    """
    return {"blocks": [word_result(8), word_result(2), word_result(7), word_result(1, -13.0), word_result(8)],
            "word_checks": [],
            "block_checks": [{"block": 1, "keep": -30.0, "drop_separate": -28.0, "drop_merged": -20.0},
                             {"block": 3, "keep": -25.0, "drop_separate": -21.0, "drop_merged": -40.0}],
            "break_checks": []}


def test_blocks_keep_the_way_back_to_the_flow() -> None:
    """Style tags are hidden from the model and the page-top position is recorded."""
    flow_blocks = collect_flow_blocks(sample_flow())
    assert flow_blocks[2].plain_words == ("old", "mission", "at", "dawn", "and", "waited", "there.")
    assert flow_blocks[1].page_top and not flow_blocks[2].page_top
    assert block_check_indexes(flow_blocks) == [1, 3]
    assert break_check_indexes(flow_blocks) == [4]


def test_wrong_heading_is_removed_and_the_sentence_rejoined() -> None:
    """The running head disappears and both halves of the sentence form one paragraph."""
    flow_items = sample_flow()
    flow_blocks = collect_flow_blocks(flow_items)
    flow_edits = decide_edits(flow_blocks, sample_result())
    assert flow_edits.deleted_blocks == frozenset({1, 3})
    edited_items = apply_flow_edits(flow_items, flow_blocks, flow_edits)
    paragraphs = [flow_item for flow_item in edited_items if flow_item["kind"] == "p"]
    assert [paragraph["text"] for paragraph in paragraphs] == [
        "He rode down the long valley to the old mission at <i>dawn</i> and waited there.",
        "Then he slept a long and quiet sleep."]
    assert paragraphs[0]["pages"] == {1, 2}
    assert not any(flow_item["kind"] == "heading" for flow_item in edited_items)


def test_misread_word_that_fits_is_kept() -> None:
    """A long real word is far more likely text than noise even when it reads oddly."""
    assert noise_log_odds("swarthy", 6.5, -9.3) < 0
    assert noise_log_odds("||", 4.7, -13.3) > 3


def test_markup_and_note_links_are_never_cut() -> None:
    """Unbalanced tags, note links and lone hyphens are not deletable."""
    assert not deletable_token("<i>half")
    assert not deletable_token("\x00NOTE:n1:1\x00")
    assert not deletable_token("-")
    assert deletable_token("<i>word</i>")
