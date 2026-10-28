"""Turn the language-model scores into edits of the book flow.

All tests compare the log-probability of the text that follows a unit with and without it
(``gain`` = without - with). Two kinds of evidence are combined with it:

* noise units (words, short paragraphs): ``gain - NOISE_COST * characters - own_logprob + prior``
  is the log-odds that the unit is OCR noise rather than text. A real word that is merely
  misread keeps a high own probability relative to its length and is never deleted.
* headings: a heading is wrong when the text reads on without it; repeating heading text
  (running heads, strongest when it heads many pages) and a heading inside an unfinished
  sentence add evidence.

Words split by a line-break hyphen get the spelling the model finds most probable (see ``hyphen_repair``);
misread low-probability words get a candidate spelling that reads clearly better (see ``word_respelling``).
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

from pdf2fb2.fb2_output.text_markup import NOTE_TOKEN_PATTERN, SENTENCE_END_PATTERN, STYLE_TAG_PATTERN
from pdf2fb2.proofreading.check_selection import SHORT_BLOCK_WORDS
from pdf2fb2.proofreading.flow_blocks import FlowBlock
from pdf2fb2.proofreading.hyphen_repair import HyphenFix, decide_hyphen_fixes
from pdf2fb2.proofreading.word_respelling import WordRespelling, decide_respellings

NOISE_COST = math.log(60.0)  # nats per character of a random OCR string
LETTER_PRIOR = -6.0          # a token made of letters is rarely noise
SYMBOL_PRIOR = -1.0
NOISE_MARGIN = 3.0
HEADING_MARGIN = 2.0
REPEAT_BONUS = 3.0
RUNNING_HEAD_BONUS = 12.0    # the same heading text on top of many pages is a running head
RUNNING_HEAD_COUNT = 10
MID_SENTENCE_BONUS = 2.0
REPEATED_HEADING_COUNT = 3
BREAK_MARGIN = 2.0
LETTER_PATTERN = re.compile(r"[^\W\d_]", re.UNICODE)
DASH_ONLY_PATTERN = re.compile(r"^[-‐‑‒–—―]+$")


@dataclass(frozen=True)
class FlowEdits:
    """Edits decided for the text blocks (indexes into the block list).

    Attributes:
        deleted_words: ``(block_index, word_index)`` pairs.
        deleted_blocks: Blocks removed entirely.
        merged_blocks: Blocks joined to the previous paragraph.
        decision_notes: Human-readable reasons for the report.
        hyphen_fixes: Respelled hyphen breaks.
        word_respellings: Misread words replaced by a better spelling.
    """

    deleted_words: frozenset[tuple[int, int]]
    deleted_blocks: frozenset[int]
    merged_blocks: frozenset[int]
    decision_notes: tuple[dict, ...]
    hyphen_fixes: tuple[HyphenFix, ...] = ()
    word_respellings: tuple[WordRespelling, ...] = ()


def own_logprob(result_blocks: list[dict], block_index: int, word_indexes: range) -> float:
    """Summed token log-probability of some words of a block.

    Args:
        result_blocks: ``blocks`` of the worker result.
        block_index: Block.
        word_indexes: Words.

    Returns:
        Sum of the token log-probabilities.
    """
    return sum(token_entry[1] for word_index in word_indexes
               for token_entry in result_blocks[block_index]["words"][word_index]["tokens"])


def unit_logprob(check_entry: dict, result_blocks: list[dict], block_index: int, word_indexes: range) -> float:
    """Log-probability of a tested unit in its context.

    Args:
        check_entry: Word or block test result (targeted jobs carry ``own``).
        result_blocks: ``blocks`` of the worker result (scored tokens in whole-text jobs).
        block_index: Block of the unit.
        word_indexes: Words of the unit.

    Returns:
        ``own - keep`` when the test has an ``own`` score, otherwise the sum of the unit's token log-probabilities.
    """
    if isinstance(check_entry.get("own"), (int, float)):
        return check_entry["own"] - check_entry["keep"]
    return own_logprob(result_blocks, block_index, word_indexes)


def noise_log_odds(unit_text: str, continuation_gain: float, unit_logprob: float) -> float:
    """Log-odds that a unit is OCR noise rather than book text.

    Args:
        unit_text: Plain text of the unit.
        continuation_gain: Log-probability gain of the following text when the unit is removed.
        unit_logprob: Log-probability of the unit itself in its context.

    Returns:
        Log-odds; above ``NOISE_MARGIN`` the unit is deleted.
    """
    unit_prior = LETTER_PRIOR if LETTER_PATTERN.search(unit_text) else SYMBOL_PRIOR
    return continuation_gain - NOISE_COST * len(unit_text) - unit_logprob + unit_prior


def deletable_token(raw_token: str) -> bool:
    """Tell whether a raw token can be removed without breaking markup or note links.

    Args:
        raw_token: Token of the flow text.

    Returns:
        False for note links, unbalanced style tags and lone dashes (mis-spaced hyphens are book text).
    """
    if NOTE_TOKEN_PATTERN.search(raw_token) or DASH_ONLY_PATTERN.match(raw_token):
        return False
    opened_tags = [tag_match.group() for tag_match in STYLE_TAG_PATTERN.finditer(raw_token)]
    return sum(1 if not tag_text.startswith("</") else -1 for tag_text in opened_tags) == 0


def word_deletions(flow_blocks: list[FlowBlock], worker_result: dict) -> tuple[set, list[dict]]:
    """Words that the model explains better as noise.

    Args:
        flow_blocks: Blocks sent to the worker.
        worker_result: Worker result document.

    Returns:
        ``(deleted (block, word) pairs, notes)``.
    """
    deleted_words, decision_notes = set(), []
    for word_check in worker_result["word_checks"]:
        block_index, word_index = word_check["block"], word_check["word"]
        flow_block = flow_blocks[block_index]
        if len(flow_block.plain_words) <= SHORT_BLOCK_WORDS or flow_block.block_kind != "p":
            continue
        raw_token = flow_block.raw_tokens[flow_block.word_tokens[word_index]]
        unit_text = flow_block.plain_words[word_index]
        log_odds = noise_log_odds(unit_text, word_check["drop"] - word_check["keep"],
                                  unit_logprob(word_check, worker_result["blocks"], block_index,
                                               range(word_index, word_index + 1)))
        if log_odds > NOISE_MARGIN and deletable_token(raw_token):
            deleted_words.add((block_index, word_index))
            decision_notes.append({"action": "delete_word", "block": block_index, "word": unit_text,
                                   "noise_log_odds": round(log_odds, 2)})
    return deleted_words, decision_notes


def ends_sentence(flow_block: FlowBlock | None) -> bool:
    """Tell whether a block ends a sentence (or there is no block).

    Args:
        flow_block: Block or None.

    Returns:
        True when the last visible characters close a sentence.
    """
    return flow_block is None or bool(SENTENCE_END_PATTERN.search(" ".join(flow_block.plain_words)))


def heading_evidence(flow_blocks: list[FlowBlock], block_check: dict, heading_counts: Counter) -> float:
    """Evidence that a heading does not belong where it stands.

    Args:
        flow_blocks: Blocks sent to the worker.
        block_check: Worker result of the block test.
        heading_counts: How often every heading text occurs.

    Returns:
        Evidence score; at or above ``HEADING_MARGIN`` the heading is deleted.
    """
    block_index = block_check["block"]
    previous_block = flow_blocks[block_index - 1] if block_index else None
    inside_sentence = not ends_sentence(previous_block)
    dropped_score = block_check["drop_merged"] if inside_sentence else block_check["drop_separate"]
    return (dropped_score - block_check["keep"] + repetition_bonus(flow_blocks[block_index], heading_counts)
            + (MID_SENTENCE_BONUS if inside_sentence else 0.0))


def repetition_bonus(flow_block: FlowBlock, heading_counts: Counter) -> float:
    """Evidence from how often the heading text occurs.

    Args:
        flow_block: Heading block.
        heading_counts: How often every heading text occurs.

    Returns:
        ``RUNNING_HEAD_BONUS`` for frequent page-top headings, ``REPEAT_BONUS`` for repeated ones, else 0.
    """
    occurrence_count = heading_counts[" ".join(flow_block.plain_words)]
    if flow_block.page_top and occurrence_count >= RUNNING_HEAD_COUNT:
        return RUNNING_HEAD_BONUS
    return REPEAT_BONUS if occurrence_count >= REPEATED_HEADING_COUNT else 0.0


def block_decision(flow_blocks: list[FlowBlock], worker_result: dict, block_check: dict,
                   heading_counts: Counter) -> dict | None:
    """Decide one block test.

    Args:
        flow_blocks: Blocks sent to the worker.
        worker_result: Worker result document.
        block_check: Worker result of the test.
        heading_counts: How often every heading text occurs.

    Returns:
        Note ``{"action": "delete_block", "block", "text", "merge", "evidence"}`` or None to keep.
    """
    block_index = block_check["block"]
    flow_block = flow_blocks[block_index]
    if any(NOTE_TOKEN_PATTERN.search(raw_token) for raw_token in flow_block.raw_tokens):
        return None
    block_text = " ".join(flow_block.plain_words)
    if flow_block.block_kind == "heading":
        evidence, required_margin = heading_evidence(flow_blocks, block_check, heading_counts), HEADING_MARGIN
    else:
        best_drop = max(block_check["drop_separate"], block_check["drop_merged"])
        evidence = noise_log_odds(block_text.replace(" ", ""), best_drop - block_check["keep"],
                                  unit_logprob(block_check, worker_result["blocks"], block_index,
                                               range(len(flow_block.plain_words))))
        required_margin = NOISE_MARGIN
    if evidence < required_margin:
        return None
    merge_neighbours = block_check["drop_merged"] > block_check["drop_separate"] and not ends_sentence(
        flow_blocks[block_index - 1] if block_index else None)
    return {"action": "delete_block", "block": block_index, "kind": flow_block.block_kind, "text": block_text,
            "merge": merge_neighbours, "evidence": round(evidence, 2)}


def decide_edits(flow_blocks: list[FlowBlock], worker_result: dict) -> FlowEdits:
    """All edits of the flow.

    Usage:
        Call with the blocks of ``collect_flow_blocks`` and the validated worker result.

    Args:
        flow_blocks: Blocks sent to the worker.
        worker_result: Worker result document.

    Returns:
        The edits with their reasons.
    """
    deleted_words, decision_notes = word_deletions(flow_blocks, worker_result)
    heading_counts = Counter(" ".join(flow_block.plain_words) for flow_block in flow_blocks
                             if flow_block.block_kind == "heading")
    block_notes = [block_note for block_check in worker_result["block_checks"]
                   if (block_note := block_decision(flow_blocks, worker_result, block_check, heading_counts))]
    deleted_blocks = {block_note["block"] for block_note in block_notes}
    merged_blocks = {block_note["block"] + 1 for block_note in block_notes if block_note["merge"]}
    break_notes = [{"action": "merge_paragraphs", "block": break_check["block"],
                    "gain": round(break_check["merged"] - break_check["keep"], 2)}
                   for break_check in worker_result["break_checks"]
                   if break_check["merged"] - break_check["keep"] >= BREAK_MARGIN]
    merged_blocks |= {break_note["block"] for break_note in break_notes}
    hyphen_fixes, hyphen_notes = decide_hyphen_fixes(worker_result)
    word_respellings, respell_notes = decide_respellings(worker_result, frozenset(deleted_words))
    return FlowEdits(frozenset(deleted_words), frozenset(deleted_blocks), frozenset(merged_blocks - deleted_blocks),
                     tuple(decision_notes + block_notes + break_notes + hyphen_notes + respell_notes),
                     tuple(hyphen_fixes), tuple(word_respellings))
