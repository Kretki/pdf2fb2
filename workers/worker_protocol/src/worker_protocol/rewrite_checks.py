"""Rewrite tests of a proofreading job: which spelling of a word span reads best in its context.

The orchestrator names a span of words (``block``, first ``word``, ``count``) and two or more spellings of it
(e.g. ``kept`` "дискути- ровали", ``hyphenated`` "дискути-ровали", ``joined`` "дискутировали"). The worker
replaces the span with every spelling and scores the spelling plus the words after it, from the same context.

Result entry: ``{"block": int, "word": int, "count": int, <variant name>: float, ...}``.
"""

import re
from dataclasses import dataclass

from worker_protocol.field_validation import require_integer, require_list

MAX_REWRITE_CHECKS = 200_000
MAX_SPAN_WORDS = 8
MAX_VARIANTS = 8
MAX_VARIANT_LENGTH = 4096
VARIANT_NAME_PATTERN = re.compile(r"^[a-z_]{1,32}$")


@dataclass(frozen=True)
class RewriteCheck:
    """One rewrite test.

    Attributes:
        block_index: Block of the span.
        word_index: First word of the span inside the block.
        word_count: Number of words in the span, 1..8.
        variant_texts: ``(name, spelling)`` pairs in order; names are lowercase identifiers.
    """

    block_index: int
    word_index: int
    word_count: int
    variant_texts: tuple[tuple[str, str], ...]

    def to_payload(self) -> dict:
        """Serialise the test.

        Returns:
            JSON-compatible dictionary.
        """
        return {"block": self.block_index, "word": self.word_index, "count": self.word_count,
                "variants": dict(self.variant_texts)}


def validate_variants(variant_payload: object) -> tuple[tuple[str, str], ...]:
    """Decode the spellings of a rewrite test.

    Args:
        variant_payload: Decoded JSON value (object name -> spelling).

    Returns:
        ``(name, spelling)`` pairs.

    Raises:
        ValueError: For a non-object, too few or too many variants, bad names or unbounded spellings.
    """
    if not isinstance(variant_payload, dict) or not 2 <= len(variant_payload) <= MAX_VARIANTS:
        raise ValueError(f"Rewrite variants must be an object with 2..{MAX_VARIANTS} entries.")
    for variant_name, variant_text in variant_payload.items():
        if not VARIANT_NAME_PATTERN.fullmatch(variant_name) or not isinstance(variant_text, str) \
                or not 0 < len(variant_text) <= MAX_VARIANT_LENGTH or variant_text.isspace() or "\n" in variant_text:
            raise ValueError("Rewrite variants need lowercase names and one-line, non-blank spellings.")
    return tuple(variant_payload.items())


def validate_rewrite_check(check_entry: object, block_word_counts: list[int]) -> RewriteCheck:
    """Decode one rewrite test.

    Args:
        check_entry: Decoded JSON value.
        block_word_counts: Number of words of every block of the job.

    Returns:
        The validated test.

    Raises:
        ValueError: If a field is missing or the span lies outside its block.
    """
    if not isinstance(check_entry, dict):
        raise ValueError("Every rewrite check must be a JSON object.")
    block_index = require_integer(check_entry, "block", 0, max(0, len(block_word_counts) - 1))
    word_count = require_integer(check_entry, "count", 1, MAX_SPAN_WORDS)
    word_index = require_integer(check_entry, "word", 0, max(0, block_word_counts[block_index] - word_count))
    return RewriteCheck(block_index, word_index, word_count, validate_variants(check_entry.get("variants")))


def validate_rewrite_checks(job_payload: dict, block_word_counts: list[int]) -> tuple[RewriteCheck, ...]:
    """Read the optional ``rewrite_checks`` list of a job.

    Args:
        job_payload: Decoded job object.
        block_word_counts: Number of words of every block of the job.

    Returns:
        The tests (empty when the field is absent).

    Raises:
        ValueError: If an entry is invalid.
    """
    if "rewrite_checks" not in job_payload:
        return ()
    if not block_word_counts:
        raise ValueError("Rewrite checks need text blocks.")
    return tuple(validate_rewrite_check(check_entry, block_word_counts)
                 for check_entry in require_list(job_payload, "rewrite_checks", MAX_REWRITE_CHECKS))
