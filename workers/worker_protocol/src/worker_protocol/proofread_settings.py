"""Tunable limits of a proofreading job (part of ``proofread_job_schema``)."""

from dataclasses import dataclass

from worker_protocol.field_validation import require_integer, require_number

DEFAULT_RESPELL_CANDIDATES = 6
MAX_RESPELL_CANDIDATES = 7  # with the spelling read: the rewrite limit of 8 variants


def optional_integer(settings_payload: dict, field_name: str, default_value: int, lower_bound: int,
                     upper_bound: int) -> int:
    """Read an integer setting that older job files may lack.

    Args:
        settings_payload: Decoded settings object.
        field_name: Key to read.
        default_value: Value used when the key is absent.
        lower_bound: Smallest allowed value.
        upper_bound: Largest allowed value.

    Returns:
        The value, or ``default_value`` when absent.

    Raises:
        ValueError: If the value is present but not an integer within the bounds.
    """
    if field_name not in settings_payload:
        return default_value
    return require_integer(settings_payload, field_name, lower_bound, upper_bound)


@dataclass(frozen=True)
class ProofreadSettings:
    """Tunable limits of the analysis.

    Attributes:
        candidate_logprob: A word is tested when its least probable token is below this (natural log).
        context_characters: Characters of preceding text given to every deletion test.
        continuation_words: Words after the tested unit whose probability is compared.
        max_word_checks: Upper bound of word tests (the least probable words are tested first).
        max_respell_checks: Upper bound of respelling tests of the same words (0 disables them).
        respell_candidates: Spellings tried per respelled word besides the one read, 1..7.
    """

    candidate_logprob: float
    context_characters: int
    continuation_words: int
    max_word_checks: int
    max_respell_checks: int = 0
    respell_candidates: int = DEFAULT_RESPELL_CANDIDATES

    def to_payload(self) -> dict:
        """Serialise the settings.

        Returns:
            JSON-compatible dictionary.
        """
        return {"candidate_logprob": self.candidate_logprob, "context_characters": self.context_characters,
                "continuation_words": self.continuation_words, "max_word_checks": self.max_word_checks,
                "max_respell_checks": self.max_respell_checks, "respell_candidates": self.respell_candidates}

    @classmethod
    def from_payload(cls, settings_payload: object) -> "ProofreadSettings":
        """Decode and bound the settings.

        Args:
            settings_payload: Decoded JSON value.

        Returns:
            The validated settings.

        Raises:
            ValueError: If a value is missing or out of range.
        """
        if not isinstance(settings_payload, dict):
            raise ValueError("Field 'settings' must be an object.")
        return cls(candidate_logprob=require_number(settings_payload, "candidate_logprob", -100.0, 0.0),
                   context_characters=require_integer(settings_payload, "context_characters", 64, 20_000),
                   continuation_words=require_integer(settings_payload, "continuation_words", 1, 128),
                   max_word_checks=require_integer(settings_payload, "max_word_checks", 0, 1_000_000),
                   max_respell_checks=optional_integer(settings_payload, "max_respell_checks", 0, 0, 1_000_000),
                   respell_candidates=optional_integer(settings_payload, "respell_candidates",
                                                       DEFAULT_RESPELL_CANDIDATES, 1, MAX_RESPELL_CANDIDATES))
