"""Pad encoded windows into batch tensors (shared by training and prediction)."""

from dataclasses import dataclass

import numpy
import torch

from token_verifier.datasets.example_files import IGNORE_INDEX
from token_verifier.datasets.token_encoding import EncodedWindow
from token_verifier.networks.backbone_adapters import NO_UNIT_INDEX


@dataclass(frozen=True)
class WindowBatch:
    """Tensors of one batch.

    Attributes:
        input_tensor: ``(batch, pieces)`` ids.
        mask_tensor: ``(batch, pieces)`` attention mask.
        piece_units: ``(batch, pieces)`` unit index per piece (-1 for none).
        feature_tensor: ``(batch, units, feature_width)`` evidence.
        visual_tensor: ``(batch, units, visual_width)`` crop rows, or None.
        label_tensor: ``(batch, units)`` class indexes (``IGNORE_INDEX`` for padding / ignore).
    """

    input_tensor: torch.Tensor
    mask_tensor: torch.Tensor
    piece_units: torch.Tensor
    feature_tensor: torch.Tensor
    visual_tensor: torch.Tensor | None
    label_tensor: torch.Tensor


def padded_rows(row_blocks: list[numpy.ndarray], unit_capacity: int) -> numpy.ndarray:
    """Stack per-window unit matrices, padding units with zeros.

    Args:
        row_blocks: ``(units, width)`` matrices.
        unit_capacity: Units after padding.

    Returns:
        ``(windows, unit_capacity, width)`` float32 array.
    """
    row_width = row_blocks[0].shape[1] if row_blocks else 0
    stacked_rows = numpy.zeros((len(row_blocks), unit_capacity, row_width), dtype=numpy.float32)
    for window_index, row_block in enumerate(row_blocks):
        stacked_rows[window_index, :row_block.shape[0]] = row_block
    return stacked_rows


def collate_windows(encoded_windows: list[EncodedWindow], padding_identifier: int,
                    torch_device: torch.device) -> WindowBatch:
    """Pad windows into batch tensors.

    Usage:
        Called for every batch of training and prediction.

    Args:
        encoded_windows: Windows of one batch.
        padding_identifier: Pad id of the backbone.
        torch_device: Target device.

    Returns:
        The batch.
    """
    longest_input = max(len(encoded_window.input_identifiers) for encoded_window in encoded_windows)
    unit_capacity = max(max(encoded_window.unit_count for encoded_window in encoded_windows), 1)
    input_rows, mask_rows, piece_rows, label_rows = [], [], [], []
    for encoded_window in encoded_windows:
        padding_size = longest_input - len(encoded_window.input_identifiers)
        input_rows.append(list(encoded_window.input_identifiers) + [padding_identifier] * padding_size)
        mask_rows.append([1] * len(encoded_window.input_identifiers) + [0] * padding_size)
        piece_rows.append(list(encoded_window.piece_units) + [NO_UNIT_INDEX] * padding_size)
        label_rows.append(list(encoded_window.unit_labels) + [IGNORE_INDEX] * (unit_capacity -
                                                                               encoded_window.unit_count))
    has_visual = encoded_windows[0].visual_rows is not None
    visual_tensor = torch.from_numpy(padded_rows([encoded_window.visual_rows for encoded_window in encoded_windows],
                                                 unit_capacity)).to(torch_device) if has_visual else None
    return WindowBatch(torch.tensor(input_rows, device=torch_device), torch.tensor(mask_rows, device=torch_device),
                       torch.tensor(piece_rows, device=torch_device),
                       torch.from_numpy(padded_rows([encoded_window.feature_rows for encoded_window in encoded_windows],
                                                    unit_capacity)).to(torch_device),
                       visual_tensor, torch.tensor(label_rows, device=torch_device))
