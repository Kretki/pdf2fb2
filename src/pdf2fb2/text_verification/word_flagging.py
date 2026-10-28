"""Words of the book flow that the token verifier flags: the only words the language model then examines.

The flow blocks are the exact text the proofreader sends to Qwen3.5. Each block is split into verifier units
(``verify_units``); a plain word is flagged when any of its units is labelled ``misread`` or ``inserted``.
"""

from pathlib import Path

from pdf2fb2.progress.progress_contracts import StageProgress
from pdf2fb2.text_verification.unit_flags import flag_units
from pdf2fb2.text_verification.verification_contracts import UnitLabeler
from pdf2fb2.text_verification.verify_units import verify_units

FLAG_JOB_NAME = "proofread_verify_job"


def unit_word_indexes(plain_words: tuple[str, ...]) -> tuple[list[str], list[int]]:
    """Verifier units of a block and the word each one belongs to.

    Args:
        plain_words: Plain words of the block.

    Returns:
        ``(unit texts, word index per unit)``.
    """
    unit_texts, owner_words = [], []
    for word_index, word_text in enumerate(plain_words):
        for verify_unit in verify_units(word_text):
            unit_texts.append(verify_unit.unit_text)
            owner_words.append(word_index)
    return unit_texts, owner_words


class VerifierWordFlagger:
    """Selects the words worth a language-model test.

    Usage:
        Build with the verification stage and pass to ``LanguageModelProofreader``.
    """

    def __init__(self, unit_labeler: UnitLabeler) -> None:
        """Store the verifier.

        Args:
            unit_labeler: Verification stage.
        """
        self._unit_labeler = unit_labeler

    def flag_words(self, block_words: list[tuple[str, tuple[str, ...]]], work_directory: Path,
                   stage_progress: StageProgress) -> list[tuple[int, int]]:
        """Flagged words of the flow.

        Args:
            block_words: ``(kind, plain words)`` per block, as sent to the proofreader.
            work_directory: Output directory (job files).
            stage_progress: Progress sink.

        Returns:
            ``(block, word)`` pairs in reading order.
        """
        block_units = {str(block_index): unit_word_indexes(plain_words)
                       for block_index, (_, plain_words) in enumerate(block_words)}
        verify_outcome = self._unit_labeler.label_units(
            {block_key: unit_texts for block_key, (unit_texts, _) in block_units.items()}, work_directory,
            FLAG_JOB_NAME, stage_progress)
        flagged_words: set[tuple[int, int]] = set()
        for block_key, (_, owner_words) in block_units.items():
            unit_flags = flag_units(verify_outcome.unit_probabilities.get(block_key, []),
                                    verify_outcome.class_thresholds)
            flagged_words.update((int(block_key), owner_words[unit_index])
                                 for unit_index in unit_flags.misread_units + unit_flags.inserted_units)
        return sorted(flagged_words)
