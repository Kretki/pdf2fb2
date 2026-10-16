"""Reading order of the detected words of one segment: group into lines, lines top-down, words left-right."""

LINE_TOLERANCE = 0.5


def box_centre_y(word_entry: dict) -> float:
    """Vertical centre of a word box.

    Args:
        word_entry: Word with ``bbox``.

    Returns:
        Centre y in pixels.
    """
    return (word_entry["bbox"][1] + word_entry["bbox"][3]) / 2


def belongs_to_line(text_line: list[dict], word_entry: dict) -> bool:
    """Tell whether a word's centre lies within half a line height of the line's mean centre.

    Args:
        text_line: Words already in the line.
        word_entry: Candidate word.

    Returns:
        True if the word continues the line.
    """
    line_centre = sum(box_centre_y(line_word) for line_word in text_line) / len(text_line)
    line_heights = sorted(line_word["bbox"][3] - line_word["bbox"][1] for line_word in text_line)
    return abs(box_centre_y(word_entry) - line_centre) <= LINE_TOLERANCE * line_heights[len(line_heights) // 2]


def group_into_lines(segment_words: list[dict]) -> list[list[dict]]:
    """Split words into text lines by their vertical centres.

    Args:
        segment_words: Words with ``bbox``, in any order.

    Returns:
        Lines top-down, each with its words left-to-right.
    """
    text_lines: list[list[dict]] = []
    for word_entry in sorted(segment_words, key=box_centre_y):
        if text_lines and belongs_to_line(text_lines[-1], word_entry):
            text_lines[-1].append(word_entry)
        else:
            text_lines.append([word_entry])
    return [sorted(text_line, key=lambda line_word: line_word["bbox"][0]) for text_line in text_lines]


def words_in_reading_order(segment_words: list[dict]) -> list[dict]:
    """Flatten the lines of a segment into reading order.

    Usage:
        Call before aligning words with the OCR tokens of the same segment.

    Args:
        segment_words: Words with ``bbox`` (as returned by the style stage).

    Returns:
        Words in reading order.
    """
    return [line_word for text_line in group_into_lines(segment_words) for line_word in text_line]
