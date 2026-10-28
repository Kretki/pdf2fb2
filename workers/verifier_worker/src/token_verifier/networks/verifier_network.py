"""Unit classifier on top of any backbone: pooled sub-word states + page evidence + optional crop embedding.

Every unit is represented by the max and the mean of the states of all its pieces (an OCR error inside a word
changes later pieces, not only the first), concatenated with a projection of its evidence features (OCR
confidence, word boxes, second read, language-model plausibility) and, when crops are used, a projection of the
image embedding of its word crop. A small MLP gives the three class logits.
"""

import torch
from torch import nn

CLASS_COUNT = 3
FEATURE_PROJECTION = 64
VISUAL_PROJECTION = 128
HEAD_DROPOUT = 0.1


def pooled_units(piece_states: torch.Tensor, piece_units: torch.Tensor, unit_capacity: int) -> torch.Tensor:
    """Max- and mean-pool piece states into unit states.

    Usage:
        Called by ``VerifierNetwork.forward``; pieces with unit index -1 are ignored.

    Args:
        piece_states: ``(batch, pieces, hidden)``.
        piece_units: ``(batch, pieces)`` long unit index per piece, -1 for none.
        unit_capacity: Units per window after padding.

    Returns:
        ``(batch, unit_capacity, 2 * hidden)``; units without pieces get zeros.
    """
    batch_size, _, hidden_size = piece_states.shape
    valid_pieces = piece_units >= 0
    bucket_index = torch.where(valid_pieces, piece_units, torch.full_like(piece_units, unit_capacity))
    expanded_index = bucket_index.unsqueeze(-1).expand(-1, -1, hidden_size)
    state_values = piece_states.float()
    unit_maxima = torch.full((batch_size, unit_capacity + 1, hidden_size), float("-inf"), device=piece_states.device)
    unit_maxima = unit_maxima.scatter_reduce(1, expanded_index, state_values, reduce="amax", include_self=True)
    unit_sums = torch.zeros_like(unit_maxima).scatter_add(1, expanded_index, state_values)
    piece_counts = torch.zeros((batch_size, unit_capacity + 1, 1), device=piece_states.device).scatter_add(
        1, bucket_index.unsqueeze(-1), valid_pieces.unsqueeze(-1).float())
    has_pieces = piece_counts > 0
    unit_maxima = torch.where(has_pieces, unit_maxima, torch.zeros_like(unit_maxima))
    unit_means = unit_sums / piece_counts.clamp(min=1.0)
    return torch.cat([unit_maxima, unit_means], dim=-1)[:, :unit_capacity]


class VerifierNetwork(nn.Module):
    """Backbone plus unit head.

    Usage:
        Built by ``model_loading`` with the backbone adapter's encoder; ``forward`` returns unit logits.
    """

    def __init__(self, encoder_module: nn.Module, hidden_size: int, feature_width: int, visual_width: int) -> None:
        """Create the head.

        Args:
            encoder_module: Backbone encoder (registered as a submodule).
            hidden_size: Backbone state width.
            feature_width: Width of the per-unit evidence vector (0 disables it).
            visual_width: Width of the per-unit crop embedding vector (0 disables crops).
        """
        super().__init__()
        self.encoder_module = encoder_module
        self.feature_width, self.visual_width = feature_width, visual_width
        self.feature_layer = nn.Sequential(nn.Linear(feature_width, FEATURE_PROJECTION), nn.GELU()) \
            if feature_width else None
        self.visual_layer = nn.Sequential(nn.Linear(visual_width, VISUAL_PROJECTION), nn.GELU()) \
            if visual_width else None
        joined_width = 2 * hidden_size + (FEATURE_PROJECTION if feature_width else 0) + \
            (VISUAL_PROJECTION if visual_width else 0)
        self.classifier = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(joined_width, hidden_size), nn.GELU(),
                                        nn.Dropout(HEAD_DROPOUT), nn.Linear(hidden_size, CLASS_COUNT))

    def head_state(self) -> dict[str, torch.Tensor]:
        """Weights of the head only (the backbone is saved by its adapter).

        Returns:
            State dictionary without encoder entries.
        """
        return {parameter_name: parameter_value for parameter_name, parameter_value in self.state_dict().items()
                if not parameter_name.startswith("encoder_module.")}

    def forward(self, piece_states: torch.Tensor, piece_units: torch.Tensor, unit_features: torch.Tensor,
                visual_rows: torch.Tensor | None) -> torch.Tensor:
        """Unit logits from backbone states.

        Args:
            piece_states: ``(batch, pieces, hidden)`` encoder states (from the adapter).
            piece_units: ``(batch, pieces)`` unit index per piece.
            unit_features: ``(batch, units, feature_width)`` evidence.
            visual_rows: ``(batch, units, visual_width)`` crop embeddings, or None.

        Returns:
            ``(batch, units, 3)`` logits.
        """
        unit_states = pooled_units(piece_states, piece_units, unit_features.shape[1])
        joined_parts = [unit_states]
        if self.feature_layer is not None:
            joined_parts.append(self.feature_layer(unit_features))
        if self.visual_layer is not None:
            visual_input = visual_rows if visual_rows is not None else unit_features.new_zeros(
                (*unit_features.shape[:2], self.visual_width))
            joined_parts.append(self.visual_layer(visual_input))
        return self.classifier(torch.cat(joined_parts, dim=-1))
