"""Respelling of misread words: accept a candidate spelling the language model reads clearly better.

The worker scores every low-probability word as read and in up to seven candidate spellings (OCR confusions,
nearby forms of the book), each followed by the same words from the same context, exactly like the hyphen
spellings of ``hyphen_repair``. A candidate replaces the word when its score beats the word as read by
``RESPELL_MARGIN`` after paying ``EDIT_PENALTY`` per unit of edit cost, so a real but rare word is not
"corrected" into a frequent one by a small gain. Style markup of the token is kept.
"""

import re
from dataclasses import dataclass

from pdf2fb2.fb2_output.text_markup import STYLE_TAG_PATTERN

KEPT_SPELLING = "kept"
RESPELL_MARGIN = 2.0   # nats the candidate must win by
EDIT_PENALTY = 2.0     # nats per unit of edit cost (one confusion substitution costs 0.5)
WORD_SHAPE_PATTERN = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)
TAG_SPLIT_PATTERN = re.compile(f"({STYLE_TAG_PATTERN.pattern})")  # keeps the tags as pieces


@dataclass(frozen=True)
class WordRespelling:
    """A misread word and its accepted spelling.

    Attributes:
        block_index: Block of the word.
        word_index: Word inside the block.
        spelling: Plain replacement of the whole word (punctuation included).
    """

    block_index: int
    word_index: int
    spelling: str


def best_candidate(respell_result: dict) -> tuple[str, float] | None:
    """Candidate with the highest penalised gain over the spelling read.

    Args:
        respell_result: Worker entry with ``kept``, candidate scores, ``spellings`` and ``costs``.

    Returns:
        ``(candidate name, penalised gain)`` or None when the entry has no scored candidate.
    """
    kept_score = respell_result.get(KEPT_SPELLING)
    spelling_costs = respell_result.get("costs", {})
    penalised_gains = {candidate_name: respell_result[candidate_name] - kept_score
                       - EDIT_PENALTY * spelling_costs.get(candidate_name, 0.0)
                       for candidate_name in respell_result.get("spellings", {})
                       if isinstance(respell_result.get(candidate_name), (int, float))}
    if not isinstance(kept_score, (int, float)) or not penalised_gains:
        return None
    winner_name = max(penalised_gains, key=penalised_gains.__getitem__)
    return winner_name, penalised_gains[winner_name]


def decide_respellings(worker_result: dict, deleted_words: frozenset[tuple[int, int]]
                       ) -> tuple[list[WordRespelling], list[dict]]:
    """Accepted respellings and their report notes.

    Usage:
        Called by ``edit_decisions.decide_edits`` after the word deletions are known.

    Args:
        worker_result: Worker result document (``respell_checks`` may be absent).
        deleted_words: Words already deleted (never respelled).

    Returns:
        ``(respellings, notes)``; notes have action ``respell_word``.
    """
    word_respellings, decision_notes = [], []
    for respell_result in worker_result.get("respell_checks", []):
        word_key = (respell_result["block"], respell_result["word"])
        candidate_choice = best_candidate(respell_result)
        if word_key in deleted_words or candidate_choice is None or candidate_choice[1] < RESPELL_MARGIN:
            continue
        candidate_name, penalised_gain = candidate_choice
        new_spelling = respell_result["spellings"][candidate_name]
        word_respellings.append(WordRespelling(*word_key, new_spelling))
        decision_notes.append({"action": "respell_word", "block": word_key[0], "word": word_key[1],
                               "spelling": new_spelling, "gain": round(penalised_gain, 2)})
    return word_respellings, decision_notes


def respelled_raw_token(raw_token: str, plain_word: str, new_spelling: str) -> str | None:
    """Raw token with the plain word replaced and its style tags kept.

    Args:
        raw_token: Token of the flow text, e.g. ``<i>слово</i>,``.
        plain_word: The token's plain word (markup removed), e.g. ``слово,``.
        new_spelling: Accepted plain spelling.

    Returns:
        The edited token, or None when the word is split by markup and cannot be replaced safely.
    """
    old_core = WORD_SHAPE_PATTERN.match(plain_word).group(2)
    new_core = WORD_SHAPE_PATTERN.match(new_spelling).group(2)
    if not old_core:
        return None
    text_pieces = TAG_SPLIT_PATTERN.split(raw_token)
    for piece_index, text_piece in enumerate(text_pieces):
        if old_core in text_piece and not STYLE_TAG_PATTERN.fullmatch(text_piece):
            text_pieces[piece_index] = text_piece.replace(old_core, new_core, 1)
            return "".join(text_pieces)
    return None
