"""Language-model plausibility of every unit, from the masked-LM head of the multilingual base model.

One unmasked forward pass gives, for every sub-word, the log-probability the model assigns to the sub-word it
sees; implausible spellings (OCR errors) get low values in any language. Per unit the lowest and the mean value
of its pieces are kept, normalised to -1..0, with a presence flag.
"""

from pathlib import Path

import numpy
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

LANGUAGE_MODEL_WIDTH = 3
WINDOW_UNITS = 96
MAX_PIECES = 256
SCORE_BATCH = 32
LOGPROB_FLOOR = -20.0


class LanguageModelScorer:
    """Masked language model used as a plausibility scorer.

    Usage:
        Built by ``feature_assembly`` when ``language_model_directory`` is set (``models/xlm-roberta-base``).
    """

    def __init__(self, model_directory: Path, torch_device: torch.device) -> None:
        """Load tokenizer and masked-LM model from local files.

        Args:
            model_directory: Local checkpoint with an MLM head.
            torch_device: Device.

        Raises:
            OSError: If the directory does not hold a model.
        """
        # B615: local directory only (local_files_only=True), never downloaded.
        self.text_tokenizer = AutoTokenizer.from_pretrained(str(model_directory), local_files_only=True)  # nosec B615
        self.masked_model = AutoModelForMaskedLM.from_pretrained(  # nosec B615
            str(model_directory), local_files_only=True).to(torch_device).eval()
        self.torch_device = torch_device

    @torch.inference_mode()
    def _piece_logprobs(self, unit_windows: list[list[str]]) -> list[tuple[list[int | None], list[float]]]:
        """Log-probability of every piece of a batch of unit windows.

        Args:
            unit_windows: Up to ``SCORE_BATCH`` lists of units.

        Returns:
            Per window: (unit index per piece, log-probability per piece).
        """
        encoding = self.text_tokenizer(unit_windows, is_split_into_words=True, truncation=True,
                                       max_length=MAX_PIECES, padding=True, return_tensors="pt").to(self.torch_device)
        log_probabilities = torch.log_softmax(self.masked_model(**encoding).logits.float(), dim=-1)
        seen_logprobs = log_probabilities.gather(-1, encoding["input_ids"].unsqueeze(-1)).squeeze(-1).cpu().tolist()
        return [(encoding.word_ids(window_index), seen_logprobs[window_index])
                for window_index in range(len(unit_windows))]

    def unit_scores(self, unit_texts: tuple[str, ...]) -> numpy.ndarray:
        """Plausibility features of the units of one text.

        Usage:
            Called once per text by ``feature_assembly``.

        Args:
            unit_texts: Units of the text.

        Returns:
            ``(units, 3)`` float32 matrix: lowest and mean normalised log-probability, presence flag.
        """
        score_matrix = numpy.zeros((len(unit_texts), LANGUAGE_MODEL_WIDTH), dtype=numpy.float32)
        window_starts = list(range(0, len(unit_texts), WINDOW_UNITS))
        for batch_start in range(0, len(window_starts), SCORE_BATCH):
            batch_starts = window_starts[batch_start:batch_start + SCORE_BATCH]
            piece_results = self._piece_logprobs([list(unit_texts[window_start:window_start + WINDOW_UNITS])
                                                  for window_start in batch_starts])
            for window_start, (piece_units, piece_logprobs) in zip(batch_starts, piece_results, strict=True):
                fill_unit_scores(score_matrix, window_start, piece_units, piece_logprobs)
        return score_matrix


def fill_unit_scores(score_matrix: numpy.ndarray, window_start: int, piece_units: list[int | None],
                     piece_logprobs: list[float]) -> None:
    """Write the per-unit minimum and mean of piece log-probabilities into the matrix.

    Args:
        score_matrix: ``(units, 3)`` matrix of the text (mutated).
        window_start: Index of the window's first unit.
        piece_units: Unit index within the window per piece (None for special pieces).
        piece_logprobs: Log-probability per piece.

    Returns:
        None.
    """
    unit_values: dict[int, list[float]] = {}
    for unit_offset, piece_logprob in zip(piece_units, piece_logprobs, strict=False):
        if unit_offset is not None:
            unit_values.setdefault(unit_offset, []).append(max(LOGPROB_FLOOR, piece_logprob) / -LOGPROB_FLOOR)
    for unit_offset, value_list in unit_values.items():
        score_matrix[window_start + unit_offset] = (min(value_list), sum(value_list) / len(value_list), 1.0)
