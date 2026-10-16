"""Proofreading job (orchestrator -> proofread worker): book text blocks to score with a language model.

The worker joins the blocks into one text (blocks separated by a blank line, words by a space),
gives every token a log-probability from its left context, and runs deletion tests:

* ``word`` tests for every word whose least probable token is below ``candidate_logprob``;
* ``block`` tests for the blocks listed in ``block_checks`` (suspected wrong headings / stray lines);
* ``break`` tests for the paragraph breaks listed in ``break_checks`` (block index of the break's
  second block: is it one paragraph wrongly split in two?).

A test compares the log-probability of the text that follows the unit with and without it.

Result: JSON document ``{"blocks": [{"words": [{"tokens": [[text, logprob], ...],
"min_logprob": float, "mean_logprob": float}]}], "word_checks": [{"block": int, "word": int,
"keep": float, "drop": float}], "block_checks": [{"block": int, "keep": float, "drop_separate":
float, "drop_merged": float}], "break_checks": [{"block": int, "keep": float, "merged": float}]}``.
"""

from dataclasses import dataclass
from pathlib import Path

from worker_protocol.field_validation import (
    require_absolute_directory,
    require_integer,
    require_list,
    require_number,
    require_string,
    validate_device_name,
)
from worker_protocol.job_schema import checked_relative_path
from worker_protocol.path_safety import resolve_inside_directory

BLOCK_KINDS = frozenset({"p", "heading", "caption"})
MAX_BLOCKS = 500_000
MAX_WORDS_PER_BLOCK = 50_000
MAX_WORD_LENGTH = 1024


@dataclass(frozen=True)
class TextBlock:
    """One paragraph, heading or caption of the book flow.

    Attributes:
        block_kind: ``p``, ``heading`` or ``caption``.
        block_words: Plain words (no markup) in reading order.
    """

    block_kind: str
    block_words: tuple[str, ...]

    def to_payload(self) -> dict:
        """Serialise the block.

        Returns:
            JSON-compatible dictionary.
        """
        return {"kind": self.block_kind, "words": list(self.block_words)}


def validate_text_block(block_entry: object) -> TextBlock:
    """Decode one block.

    Args:
        block_entry: Decoded JSON value.

    Returns:
        The validated block.

    Raises:
        ValueError: If the kind is unknown or the words are not bounded non-empty strings.
    """
    if not isinstance(block_entry, dict) or block_entry.get("kind") not in BLOCK_KINDS:
        raise ValueError(f"Each block must be an object with kind in {sorted(BLOCK_KINDS)}.")
    word_entries = require_list(block_entry, "words", MAX_WORDS_PER_BLOCK)
    if not all(isinstance(word_text, str) and 0 < len(word_text) <= MAX_WORD_LENGTH and not word_text.isspace()
               for word_text in word_entries):
        raise ValueError(f"Block words must be non-blank strings of at most {MAX_WORD_LENGTH} characters.")
    return TextBlock(block_kind=block_entry["kind"], block_words=tuple(word_entries))


def validate_block_indexes(job_payload: dict, field_name: str, block_count: int) -> tuple[int, ...]:
    """Read a list of block indexes.

    Args:
        job_payload: Decoded JSON object.
        field_name: Key to read.
        block_count: Number of blocks in the job.

    Returns:
        The indexes.

    Raises:
        ValueError: If an entry is not an integer index of a block.
    """
    index_entries = require_list(job_payload, field_name, MAX_BLOCKS)
    if not all(isinstance(block_index, int) and not isinstance(block_index, bool) and 0 <= block_index < block_count
               for block_index in index_entries):
        raise ValueError(f"Field {field_name!r} must hold block indexes 0..{block_count - 1}.")
    return tuple(index_entries)


@dataclass(frozen=True)
class ProofreadSettings:
    """Tunable limits of the analysis.

    Attributes:
        candidate_logprob: A word is tested when its least probable token is below this (natural log).
        context_characters: Characters of preceding text given to every deletion test.
        continuation_words: Words after the tested unit whose probability is compared.
        max_word_checks: Upper bound of word tests (the least probable words are tested first).
    """

    candidate_logprob: float
    context_characters: int
    continuation_words: int
    max_word_checks: int

    def to_payload(self) -> dict:
        """Serialise the settings.

        Returns:
            JSON-compatible dictionary.
        """
        return {"candidate_logprob": self.candidate_logprob, "context_characters": self.context_characters,
                "continuation_words": self.continuation_words, "max_word_checks": self.max_word_checks}

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
                   max_word_checks=require_integer(settings_payload, "max_word_checks", 0, 1_000_000))


@dataclass(frozen=True)
class ProofreadJob:
    """Request to score the book text and run deletion tests.

    Usage:
        Built by the orchestrator, serialised with ``to_payload`` and decoded by the proofread worker
        with ``from_payload``.

    Attributes:
        work_directory: Absolute directory the result path refers to.
        model_directory: Absolute directory of the local Qwen3.5 weights.
        device_name: Torch device, e.g. ``cuda:0`` / ``gpu:0`` / ``cpu``.
        text_blocks: Blocks of the book in reading order.
        block_checks: Indexes of blocks to test for deletion.
        break_checks: Indexes of blocks whose break from the previous block is tested.
        proofread_settings: Analysis limits.
        result_file: Relative path of the JSON result document.
    """

    work_directory: Path
    model_directory: Path
    device_name: str
    text_blocks: tuple[TextBlock, ...]
    block_checks: tuple[int, ...]
    break_checks: tuple[int, ...]
    proofread_settings: ProofreadSettings
    result_file: str

    def to_payload(self) -> dict:
        """Serialise the job to a JSON-compatible object.

        Usage:
            Call before writing the job file.

        Returns:
            Dictionary ready for ``json.dumps``.
        """
        return {"work_directory": str(self.work_directory), "model_directory": str(self.model_directory),
                "device_name": self.device_name, "blocks": [text_block.to_payload() for text_block in self.text_blocks],
                "block_checks": list(self.block_checks), "break_checks": list(self.break_checks),
                "settings": self.proofread_settings.to_payload(), "result_file": self.result_file}

    @classmethod
    def from_payload(cls, job_payload: dict) -> "ProofreadJob":
        """Decode and validate a proofreading job.

        Usage:
            Call in the proofread worker right after reading the job file.

        Args:
            job_payload: Decoded JSON object produced by ``to_payload``.

        Returns:
            The validated job.

        Raises:
            ValueError: If any field is missing, out of range or escapes the working directory.
        """
        work_directory = require_absolute_directory(require_string(job_payload, "work_directory"), "work_directory")
        text_blocks = tuple(validate_text_block(block_entry)
                            for block_entry in require_list(job_payload, "blocks", MAX_BLOCKS))
        return cls(
            work_directory=work_directory,
            model_directory=require_absolute_directory(require_string(job_payload, "model_directory"),
                                                       "model_directory"),
            device_name=validate_device_name(require_string(job_payload, "device_name", 16)),
            text_blocks=text_blocks,
            block_checks=validate_block_indexes(job_payload, "block_checks", len(text_blocks)),
            break_checks=validate_block_indexes(job_payload, "break_checks", len(text_blocks)),
            proofread_settings=ProofreadSettings.from_payload(job_payload.get("settings")),
            result_file=checked_relative_path(work_directory, require_string(job_payload, "result_file")),
        )

    def resolve_inside(self, relative_path: str) -> Path:
        """Resolve a relative job path against the working directory.

        Args:
            relative_path: Path relative to ``work_directory``.

        Returns:
            Absolute path inside ``work_directory``.

        Raises:
            ValueError: If the path escapes ``work_directory``.
        """
        return resolve_inside_directory(self.work_directory, relative_path)
