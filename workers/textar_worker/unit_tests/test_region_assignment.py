"""Unit tests of the word -> text region assignment."""

from textar_worker.attribute_inference import WordAttributes
from textar_worker.region_assignment import containing_region, group_by_region, reading_order


def make_word(word_box: tuple[int, int, int, int], emphasis_class: int = 0) -> WordAttributes:
    """Build a word with fixed scores.

    Args:
        word_box: Pixel box.
        emphasis_class: Emphasis class 0..3.

    Returns:
        WordAttributes.
    """
    return WordAttributes(word_box=word_box, emphasis_class=emphasis_class, emphasis_score=0.9,
                          line_class=0, line_score=0.9)


def test_smallest_containing_region_wins() -> None:
    """A word inside nested regions belongs to the inner one."""
    text_regions = [(0, 0, 500, 500), (10, 10, 200, 100)]
    assert containing_region((20, 20, 60, 40), text_regions) == 1
    assert containing_region((300, 300, 340, 320), text_regions) == 0
    assert containing_region((600, 600, 640, 620), text_regions) is None


def test_group_by_region_keeps_payload_fields() -> None:
    """Words are distributed per region with classes and rounded scores."""
    grouped_words = group_by_region([make_word((20, 20, 60, 40), 2), make_word((900, 900, 950, 920))],
                                    [(0, 0, 100, 100)])
    assert len(grouped_words) == 1
    assert grouped_words[0] == [{"bbox": [20, 20, 60, 40], "emphasis_class": 2, "emphasis_score": 0.9,
                                 "line_class": 0, "line_score": 0.9}]


def test_reading_order_is_top_down_left_right() -> None:
    """Boxes are sorted by top edge, then left edge."""
    assert reading_order([(100, 50, 150, 70), (0, 50, 40, 70), (0, 0, 40, 20)]) == [2, 1, 0]
