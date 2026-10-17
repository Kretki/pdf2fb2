"""Unit tests of the hyphen-break repair: candidates, rewrite tests, decisions and flow edits."""

from pdf2fb2.proofreading.edit_decisions import decide_edits
from pdf2fb2.proofreading.flow_blocks import collect_flow_blocks
from pdf2fb2.proofreading.flow_editing import apply_flow_edits
from pdf2fb2.proofreading.hyphen_repair import find_hyphen_breaks, rewrite_checks_for


def paragraph_item(paragraph_text: str) -> dict:
    """Paragraph flow item.

    Args:
        paragraph_text: Text of the paragraph.

    Returns:
        Flow item.
    """
    return {"kind": "p", "text": paragraph_text, "style": "p", "list": False, "pages": {1}}


def word_entries(word_count: int) -> dict:
    """Worker block entry with one token per word.

    Args:
        word_count: Number of words.

    Returns:
        Block entry.
    """
    return {"words": [{"tokens": [["w", -2.0]], "min_logprob": -2.0, "mean_logprob": -2.0}
                      for _ in range(word_count)]}


def sample_flow() -> list[dict]:
    """Three kinds of hyphen breaks, one inside italics, and a real dash between numbers.

    Returns:
        Flow items.
    """
    return [paragraph_item("Мы дискути- ровали о <i>grand</i> <i>-</i> <i>father</i> и the -tall men."),
            paragraph_item("Spanish - Californian ranchos, years 1850 - 1853.")]


def test_breaks_and_their_spellings_are_found() -> None:
    """Letter-hyphen-letter spans become tests (separate words only for `the -tall`); numbers do not."""
    flow_blocks = collect_flow_blocks(sample_flow())
    rewrite_checks = rewrite_checks_for(flow_blocks, find_hyphen_breaks(flow_blocks))
    assert [(rewrite_check.block_index, rewrite_check.word_index, rewrite_check.word_count)
            for rewrite_check in rewrite_checks] == [(0, 1, 2), (0, 4, 3), (0, 8, 2), (1, 0, 3)]
    assert dict(rewrite_checks[0].variant_texts) == {"kept": "дискути- ровали", "hyphenated": "дискути-ровали",
                                                     "joined": "дискутировали"}
    assert dict(rewrite_checks[1].variant_texts)["joined"] == "grandfather"
    assert dict(rewrite_checks[2].variant_texts)["separate"] == "the tall"


def test_best_spelling_is_applied_with_markup_kept() -> None:
    """Joined, hyphenated and separate winners rewrite the flow; a break the model prefers as read stays."""
    flow_items = sample_flow()
    flow_blocks = collect_flow_blocks(flow_items)
    worker_result = {"blocks": [word_entries(12), word_entries(7)], "word_checks": [], "block_checks": [],
                     "break_checks": [], "rewrite_checks": [
                         {"block": 0, "word": 1, "count": 2, "kept": -30.0, "hyphenated": -25.0, "joined": -9.0},
                         {"block": 0, "word": 4, "count": 3, "kept": -20.0, "hyphenated": -12.0, "joined": -8.0},
                         {"block": 0, "word": 8, "count": 2, "kept": -12.0, "hyphenated": -14.0, "joined": -12.5,
                          "separate": -7.0},
                         {"block": 1, "word": 0, "count": 3, "kept": -21.0, "hyphenated": -11.0, "joined": -30.0},
                         {"block": 1, "word": 4, "count": 3, "kept": -5.0, "hyphenated": -11.0, "joined": -30.0}]}
    flow_edits = decide_edits(flow_blocks, worker_result)
    assert len(flow_edits.hyphen_fixes) == 4
    edited_texts = [flow_item["text"] for flow_item in apply_flow_edits(flow_items, flow_blocks, flow_edits)]
    assert edited_texts == ["Мы дискутировали о <i>grand</i><i>father</i> и the tall men.",
                            "Spanish-Californian ranchos, years 1850 - 1853."]
