"""Map the whitespace tokens of a segment's OCR text onto the words detected in its image.

Token and word sequences are aligned monotonically by dynamic programming: a group of 1..3
tokens is matched with a group of 1..3 word boxes when the boxes' width fits the tokens'
character count (``width = pixels_per_character * characters + box_padding``), and single tokens or boxes may be
skipped. The width model is fitted on the segment itself, then refined on the first alignment.
"""

import math
import re
from dataclasses import dataclass

TOKEN_PATTERN = re.compile(r"\S+")
MARKDOWN_ONLY_PATTERN = re.compile(r"^(?:#{1,6}|[-*+>|•]|\d+[.)])$")
GROUP_STEPS = ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (1, 0), (0, 1))
SKIP_PENALTY = 1.5
GROUP_PENALTY = 0.35
MIN_BAND_WIDTH = 12
BAND_RATIO = 0.15
MIN_COUNT_RATIO = 0.5


@dataclass(frozen=True)
class TextToken:
    """One whitespace-separated piece of the OCR text.

    Attributes:
        start_offset: Character offset of the token in the text.
        end_offset: Offset just after the token.
    """

    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class WidthModel:
    """Linear estimate of a word box width from its character count.

    Attributes:
        pixels_per_character: Pixels per character (> 0).
        box_padding: Box padding in pixels (>= 0).
    """

    pixels_per_character: float
    box_padding: float

    def step_cost(self, character_count: int, box_width: float, group_size: int) -> float:
        """Mismatch of a token group with a box group.

        Args:
            character_count: Characters in the token group.
            box_width: Summed width of the box group.
            group_size: Number of boxes in the group (each has its own padding).

        Returns:
            Absolute log ratio of predicted and observed width.
        """
        predicted_width = self.pixels_per_character * character_count + self.box_padding * group_size
        return abs(math.log(max(predicted_width, 1.0) / max(box_width, 1.0)))


def alignable_tokens(ocr_text: str) -> list[TextToken]:
    """Tokens that correspond to printed words (Markdown block markers are skipped).

    Args:
        ocr_text: OCR Markdown of one segment.

    Returns:
        Tokens in text order.
    """
    return [TextToken(token_match.start(), token_match.end()) for token_match in TOKEN_PATTERN.finditer(ocr_text)
            if not MARKDOWN_ONLY_PATTERN.match(token_match.group())]


def fit_width_model(token_lengths: list[int], word_widths: list[float],
                    matched_pairs: list[tuple[int, int]] | None = None) -> WidthModel:
    """Fit ``width = pixels_per_character * characters + box_padding`` on one-to-one matches (or on totals).

    Args:
        token_lengths: Characters per token.
        word_widths: Pixel width per word box.
        matched_pairs: ``(token_index, word_index)`` one-to-one matches of a previous alignment.

    Returns:
        Width model with a positive slope.
    """
    if not matched_pairs or len(matched_pairs) < 3:
        return WidthModel(pixels_per_character=sum(word_widths) / max(1, sum(token_lengths)), box_padding=0.0)
    pair_lengths = [float(token_lengths[token_index]) for token_index, _ in matched_pairs]
    pair_widths = [word_widths[word_index] for _, word_index in matched_pairs]
    length_mean, width_mean = sum(pair_lengths) / len(pair_lengths), sum(pair_widths) / len(pair_widths)
    variance = sum((length_value - length_mean) ** 2 for length_value in pair_lengths)
    if variance <= 0:
        return WidthModel(pixels_per_character=width_mean / max(1.0, length_mean), box_padding=0.0)
    fitted_slope = sum((length_value - length_mean) * (width_value - width_mean)
                       for length_value, width_value in zip(pair_lengths, pair_widths, strict=True)) / variance
    fitted_slope = max(fitted_slope, 0.25 * width_mean / max(1.0, length_mean))
    return WidthModel(pixels_per_character=fitted_slope, box_padding=max(0.0, width_mean - fitted_slope * length_mean))


def band_limits(token_index: int, token_count: int, word_count: int) -> range:
    """Word positions explored for a token position (diagonal band of the DP table).

    Args:
        token_index: Tokens consumed so far.
        token_count: Total tokens.
        word_count: Total words.

    Returns:
        Range of word positions.
    """
    band_width = max(MIN_BAND_WIDTH, int(BAND_RATIO * max(token_count, word_count)))
    diagonal_position = round(token_index * word_count / max(1, token_count))
    return range(max(0, diagonal_position - band_width), min(word_count, diagonal_position + band_width) + 1)


def step_cost(width_model: WidthModel, token_lengths: list[int], word_widths: list[float],
              cell_position: tuple[int, int], group_step: tuple[int, int]) -> float:
    """Cost of consuming ``group_step`` tokens/words after ``cell_position``.

    Args:
        width_model: Width estimate.
        token_lengths: Characters per token.
        word_widths: Width per word.
        cell_position: ``(tokens_consumed, words_consumed)`` before the step.
        group_step: ``(token_count, word_count)`` consumed by the step.

    Returns:
        Step cost.
    """
    token_step, word_step = group_step
    if token_step == 0 or word_step == 0:
        return SKIP_PENALTY
    token_start, word_start = cell_position
    character_count = sum(token_lengths[token_start:token_start + token_step]) + token_step - 1
    box_width = sum(word_widths[word_start:word_start + word_step])
    return width_model.step_cost(character_count, box_width, word_step) + GROUP_PENALTY * (token_step + word_step - 2)


def fill_alignment_table(width_model: WidthModel, token_lengths: list[int],
                         word_widths: list[float]) -> dict[tuple[int, int], tuple[float, tuple[int, int]]]:
    """Banded dynamic programming over (tokens consumed, words consumed).

    Args:
        width_model: Width estimate.
        token_lengths: Characters per token.
        word_widths: Width per word.

    Returns:
        Mapping cell -> ``(best cost, step that reached it)``.
    """
    token_count, word_count = len(token_lengths), len(word_widths)
    alignment_table = {(0, 0): (0.0, (0, 0))}
    for token_position in range(token_count + 1):
        for word_position in band_limits(token_position, token_count, word_count):
            current_cell = alignment_table.get((token_position, word_position))
            if current_cell is None:
                continue
            for group_step in GROUP_STEPS:
                next_cell = (token_position + group_step[0], word_position + group_step[1])
                if next_cell[0] > token_count or next_cell[1] > word_count:
                    continue
                total_cost = current_cell[0] + step_cost(width_model, token_lengths, word_widths,
                                                         (token_position, word_position), group_step)
                if next_cell not in alignment_table or total_cost < alignment_table[next_cell][0]:
                    alignment_table[next_cell] = (total_cost, group_step)
    return alignment_table


def trace_alignment(alignment_table: dict, token_count: int,
                    word_count: int) -> tuple[list[int | None], list[tuple[int, int]]] | None:
    """Follow the best path back and give every token its word (first box of its group).

    Args:
        alignment_table: Result of ``fill_alignment_table``.
        token_count: Total tokens.
        word_count: Total words.

    Returns:
        ``(token_words, one_to_one_pairs)``: word index (None for a skipped token) per token and
        the ``(token_index, word_index)`` pairs matched one-to-one; None when the end is unreachable.
    """
    if (token_count, word_count) not in alignment_table:
        return None
    token_words: list[int | None] = [None] * token_count
    one_to_one_pairs = []
    token_position, word_position = token_count, word_count
    while (token_position, word_position) != (0, 0):
        token_step, word_step = alignment_table[(token_position, word_position)][1]
        token_position, word_position = token_position - token_step, word_position - word_step
        for token_index in range(token_position, token_position + token_step):
            token_words[token_index] = word_position if word_step else None
        if (token_step, word_step) == (1, 1):
            one_to_one_pairs.append((token_position, word_position))
    return token_words, one_to_one_pairs


def align_tokens_to_words(text_tokens: list[TextToken], word_widths: list[int]) -> list[int | None] | None:
    """Index of the word that every token belongs to.

    Usage:
        Call per segment with the tokens from ``alignable_tokens`` and the pair_widths of the words in
        reading order.

    Args:
        text_tokens: OCR tokens.
        word_widths: Pixel pair_widths of the detected words in reading order.

    Returns:
        One word index (None for an unmatched token) per token, or None when no alignment is possible.
    """
    if not text_tokens or not word_widths \
            or min(len(text_tokens), len(word_widths)) / max(len(text_tokens), len(word_widths)) < MIN_COUNT_RATIO:
        return None
    token_lengths = [text_token.end_offset - text_token.start_offset for text_token in text_tokens]
    float_widths = [float(max(1, word_width)) for word_width in word_widths]
    first_alignment = trace_alignment(fill_alignment_table(fit_width_model(token_lengths, float_widths),
                                                           token_lengths, float_widths),
                                      len(token_lengths), len(float_widths))
    if first_alignment is None:
        return None
    refined_model = fit_width_model(token_lengths, float_widths, first_alignment[1])
    refined_alignment = trace_alignment(fill_alignment_table(refined_model, token_lengths, float_widths),
                                        len(token_lengths), len(float_widths))
    return refined_alignment[0] if refined_alignment is not None else first_alignment[0]
