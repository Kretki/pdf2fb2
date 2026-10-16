"""Word attribute prediction for one page: windows -> TexTAR -> softmax averaged over windows."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy
import torch

from textar_worker.context_windows import build_context_windows, window_positions
from textar_worker.textar_network import TexTarNetwork

WINDOWS_PER_BATCH = 8
PixelWordBox = Sequence[int]


@dataclass(frozen=True)
class WordAttributes:
    """Predicted attributes of one word.

    Attributes:
        word_box: ``(x0, y0, x1, y1)`` pixel box.
        emphasis_class: 0 normal, 1 bold, 2 italic, 3 bold+italic.
        emphasis_score: Averaged softmax probability of ``emphasis_class``.
        line_class: 0 none, 1 underline, 2 strikeout, 3 underline+strikeout.
        line_score: Averaged softmax probability of ``line_class``.
    """

    word_box: tuple[int, int, int, int]
    emphasis_class: int
    emphasis_score: float
    line_class: int
    line_score: float


class AttributePredictor:
    """Runs TexTAR over all context windows of a page.

    Usage:
        Build once per job with the loaded network and call ``predict_page`` per page.
    """

    def __init__(self, textar_network: TexTarNetwork, torch_device: torch.device) -> None:
        """Store the network.

        Args:
            textar_network: Loaded network in eval mode.
            torch_device: Device the network lives on.
        """
        self._textar_network = textar_network
        self._torch_device = torch_device

    @torch.inference_mode()
    def predict_page(self, word_boxes: Sequence[PixelWordBox], word_crops: torch.Tensor) -> list[WordAttributes]:
        """Predict the attributes of every word on a page.

        Args:
            word_boxes: Word boxes in reading order.
            word_crops: ``(N, 3, 128, 96)`` crops of the same words.

        Returns:
            One WordAttributes per word, in input order (empty for no words).
        """
        context_windows = build_context_windows(word_boxes)
        if not context_windows:
            return []
        emphasis_sums = numpy.zeros((len(word_boxes), 4))
        line_sums = numpy.zeros((len(word_boxes), 4))
        for batch_start in range(0, len(context_windows), WINDOWS_PER_BATCH):
            window_batch = context_windows[batch_start:batch_start + WINDOWS_PER_BATCH]
            emphasis_probabilities, line_probabilities = self._run_windows(word_boxes, word_crops, window_batch)
            for window_indexes, window_emphasis, window_line in zip(window_batch, emphasis_probabilities,
                                                                    line_probabilities, strict=True):
                unique_indexes, first_positions = numpy.unique(window_indexes, return_index=True)
                emphasis_sums[unique_indexes] += window_emphasis[first_positions]
                line_sums[unique_indexes] += window_line[first_positions]
        return [summarise_word(word_box, emphasis_row, line_row)
                for word_box, emphasis_row, line_row in zip(word_boxes, emphasis_sums, line_sums, strict=True)]

    def _run_windows(self, word_boxes: Sequence[PixelWordBox], word_crops: torch.Tensor,
                     window_batch: list[numpy.ndarray]) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Forward a batch of windows.

        Args:
            word_boxes: Word boxes of the page.
            word_crops: Crops of the page's words.
            window_batch: Word indexes of each window.

        Returns:
            ``(emphasis, line)`` softmax probabilities, each ``(windows, window_size, 4)``.
        """
        index_tensor = torch.from_numpy(numpy.concatenate(window_batch)).long()
        batch_crops = word_crops[index_tensor].to(self._torch_device)
        batch_positions = torch.from_numpy(numpy.stack([window_positions(word_boxes, window_indexes)
                                                        for window_indexes in window_batch])).to(self._torch_device)
        emphasis_logits, line_logits = self._textar_network(batch_crops, batch_positions)
        result_shape = (len(window_batch), -1, 4)
        return (emphasis_logits.softmax(dim=-1).reshape(result_shape).cpu().numpy(),
                line_logits.softmax(dim=-1).reshape(result_shape).cpu().numpy())


def summarise_word(word_box: PixelWordBox, emphasis_sum: numpy.ndarray, line_sum: numpy.ndarray) -> WordAttributes:
    """Pick the most probable class of each attribute group from summed probabilities.

    Args:
        word_box: Word box.
        emphasis_sum: Summed emphasis probabilities over the windows containing the word.
        line_sum: Summed line probabilities over the same windows.

    Returns:
        The word's attributes with scores normalised to 0..1.
    """
    emphasis_mean = emphasis_sum / max(float(emphasis_sum.sum()), 1e-9)
    line_mean = line_sum / max(float(line_sum.sum()), 1e-9)
    return WordAttributes(word_box=tuple(int(coordinate) for coordinate in word_box),
                          emphasis_class=int(emphasis_mean.argmax()), emphasis_score=float(emphasis_mean.max()),
                          line_class=int(line_mean.argmax()), line_score=float(line_mean.max()))
