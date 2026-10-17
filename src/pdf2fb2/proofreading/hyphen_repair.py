"""Words split by a line-break hyphen that the OCR kept with a space: found here, decided by the language model.

OCR reads a word hyphenated at the end of a printed line as ``дискути- ровали``, ``grand - father`` or
``the -tall``. Each such spot becomes a rewrite test with the spellings ``kept`` (as read), ``hyphenated``
(``grand-father``) and ``joined`` (``grandfather``); a hyphen stuck to the start of the next word (``the -tall``)
may also be a speck, so that shape adds ``separate`` (``the tall``). Separate words are not offered for the other
shapes: the hyphen is printed there, and the model would favour the shorter spelling of real compounds. The
language model scores each spelling in its context; the best one wins, so real compounds keep their hyphen
(``Spanish-Californian``) and broken words are joined. The flow text is then rewritten on its raw tokens,
keeping the style tags around the pieces.
"""

from dataclasses import dataclass

from pdf2fb2.proofreading.flow_blocks import FlowBlock
from worker_protocol.rewrite_checks import RewriteCheck

HYPHEN_CHARACTERS = "-­‐‑"  # hyphen-minus, soft hyphen, hyphen, non-breaking hyphen
KEPT_VARIANT = "kept"
HYPHENATED_VARIANT = "hyphenated"
JOINED_VARIANT = "joined"
SEPARATE_VARIANT = "separate"
REWRITTEN_VARIANTS = (HYPHENATED_VARIANT, JOINED_VARIANT, SEPARATE_VARIANT)


@dataclass(frozen=True)
class HyphenBreak:
    """A word span split around a hyphen.

    Attributes:
        block_index: Block of the span.
        word_index: First word of the span.
        word_count: 2 (``left- right`` or ``left -right``) or 3 (``left - right``).
        left_text: Word before the hyphen, without it.
        right_text: Word after the hyphen, without it.
        may_be_speck: Whether the hyphen sticks to the next word (``left -right``) and may be a speck.
    """

    block_index: int
    word_index: int
    word_count: int
    left_text: str
    right_text: str
    may_be_speck: bool = False


@dataclass(frozen=True)
class HyphenFix:
    """Decided spelling of a hyphen break.

    Attributes:
        block_index: Block of the span.
        word_index: First word of the span.
        word_count: Words of the span.
        variant_name: ``hyphenated``, ``joined`` or ``separate``.
    """

    block_index: int
    word_index: int
    word_count: int
    variant_name: str


def is_hyphen(text_character: str) -> bool:
    """Tell whether a character is a hyphen.

    Args:
        text_character: One character (empty gives False).

    Returns:
        True for the characters of ``HYPHEN_CHARACTERS``.
    """
    return len(text_character) == 1 and text_character in HYPHEN_CHARACTERS


def break_at(block_index: int, plain_words: tuple[str, ...], word_index: int) -> HyphenBreak | None:
    """Hyphen break starting at a word, if any.

    Args:
        block_index: Block of the words.
        plain_words: Words of the block.
        word_index: Candidate first word.

    Returns:
        The break for ``left- right``, ``left - right`` or ``left -right`` with letters on both sides of the
        hyphen, else None.
    """
    left_word = plain_words[word_index]
    if word_index + 1 >= len(plain_words) or not left_word:
        return None
    next_word = plain_words[word_index + 1]
    if len(left_word) >= 2 and is_hyphen(left_word[-1]) and left_word[-2].isalpha() and next_word[0].isalpha():
        return HyphenBreak(block_index, word_index, 2, left_word[:-1], next_word)
    if not left_word[-1].isalpha():
        return None
    if len(next_word) >= 2 and is_hyphen(next_word[0]) and next_word[1].isalpha():
        return HyphenBreak(block_index, word_index, 2, left_word, next_word[1:], may_be_speck=True)
    if is_hyphen(next_word) and word_index + 2 < len(plain_words) and plain_words[word_index + 2][0].isalpha():
        return HyphenBreak(block_index, word_index, 3, left_word, plain_words[word_index + 2])
    return None


def find_hyphen_breaks(flow_blocks: list[FlowBlock]) -> list[HyphenBreak]:
    """All hyphen breaks of the flow, without overlaps.

    Args:
        flow_blocks: Blocks in reading order.

    Returns:
        Breaks in reading order.
    """
    hyphen_breaks = []
    for block_index, flow_block in enumerate(flow_blocks):
        word_index = 0
        while word_index < len(flow_block.plain_words):
            hyphen_break = break_at(block_index, flow_block.plain_words, word_index)
            if hyphen_break is not None:
                hyphen_breaks.append(hyphen_break)
            word_index += hyphen_break.word_count if hyphen_break is not None else 1
    return hyphen_breaks


def break_spellings(flow_block: FlowBlock, hyphen_break: HyphenBreak) -> tuple[tuple[str, str], ...]:
    """Spellings tested for one break.

    Args:
        flow_block: Block of the break.
        hyphen_break: The break.

    Returns:
        ``(name, spelling)`` pairs: ``kept``, ``hyphenated``, ``joined`` and, for possible specks, ``separate``.
    """
    kept_text = " ".join(flow_block.plain_words[hyphen_break.word_index:
                                                hyphen_break.word_index + hyphen_break.word_count])
    spellings = ((KEPT_VARIANT, kept_text), (HYPHENATED_VARIANT, f"{hyphen_break.left_text}-{hyphen_break.right_text}"),
                 (JOINED_VARIANT, hyphen_break.left_text + hyphen_break.right_text))
    if hyphen_break.may_be_speck:
        spellings += ((SEPARATE_VARIANT, f"{hyphen_break.left_text} {hyphen_break.right_text}"),)
    return spellings


def rewrite_checks_for(flow_blocks: list[FlowBlock], hyphen_breaks: list[HyphenBreak]) -> list[RewriteCheck]:
    """Rewrite tests of the hyphen breaks.

    Args:
        flow_blocks: Blocks the breaks refer to.
        hyphen_breaks: Breaks to test.

    Returns:
        One test per break with the spellings of ``break_spellings``.
    """
    return [RewriteCheck(hyphen_break.block_index, hyphen_break.word_index, hyphen_break.word_count,
                         break_spellings(flow_blocks[hyphen_break.block_index], hyphen_break))
            for hyphen_break in hyphen_breaks]


def decide_hyphen_fixes(worker_result: dict) -> tuple[list[HyphenFix], list[dict]]:
    """Pick the most probable spelling of every tested break.

    Args:
        worker_result: Worker result document (``rewrite_checks`` may be absent).

    Returns:
        ``(fixes, notes)``: a fix for every break whose best spelling is not the one read by the OCR.
    """
    hyphen_fixes, decision_notes = [], []
    for rewrite_result in worker_result.get("rewrite_checks", []):
        variant_scores = {variant_name: rewrite_result[variant_name]
                          for variant_name in (KEPT_VARIANT, *REWRITTEN_VARIANTS)
                          if isinstance(rewrite_result.get(variant_name), (int, float))}
        best_variant = max(variant_scores, key=variant_scores.__getitem__, default=KEPT_VARIANT)
        if best_variant == KEPT_VARIANT:
            continue
        hyphen_fixes.append(HyphenFix(rewrite_result["block"], rewrite_result["word"], rewrite_result["count"],
                                      best_variant))
        decision_notes.append({"action": "fix_hyphen", "block": rewrite_result["block"],
                               "word": rewrite_result["word"], "variant": best_variant,
                               "gain": round(variant_scores[best_variant] - variant_scores.get(KEPT_VARIANT, 0.0), 2)})
    return hyphen_fixes, decision_notes


def remove_hyphen(raw_token: str, from_end: bool) -> str:
    """Remove the last (or first) hyphen of a raw token, keeping its markup.

    Args:
        raw_token: Token of the flow text, e.g. ``<i>grand-</i>``.
        from_end: True for the last hyphen, False for the first.

    Returns:
        The token without that hyphen (unchanged when it has none).
    """
    hyphen_positions = [character_index for character_index, text_character in enumerate(raw_token)
                        if is_hyphen(text_character)]
    if not hyphen_positions:
        return raw_token
    hyphen_position = hyphen_positions[-1] if from_end else hyphen_positions[0]
    return raw_token[:hyphen_position] + raw_token[hyphen_position + 1:]


def repaired_token(span_tokens: list[str], variant_name: str, left_ends_with_hyphen: bool) -> str:
    """Raw text replacing the tokens of a break.

    Args:
        span_tokens: Raw tokens of the break's words (2 or 3).
        variant_name: ``hyphenated``, ``joined`` or ``separate``.
        left_ends_with_hyphen: Whether the hyphen ends the first word (``left- right``).

    Returns:
        The pieces joined directly, with a hyphen between them for ``hyphenated`` or a space for ``separate``.
    """
    if len(span_tokens) == 3:
        left_token, right_token = span_tokens[0], span_tokens[2]
    elif left_ends_with_hyphen:
        left_token, right_token = remove_hyphen(span_tokens[0], True), span_tokens[1]
    else:
        left_token, right_token = span_tokens[0], remove_hyphen(span_tokens[1], False)
    piece_separators = {HYPHENATED_VARIANT: "-", SEPARATE_VARIANT: " "}
    return left_token + piece_separators.get(variant_name, "") + right_token
