"""RoPE-Mixed attention layers of TexTAR (re-implemented from the official code, same parameter names).

The official model builds these blocks from DeiT/timm classes; only the inference path is needed
here, so the layers are written with plain torch while keeping the state-dict keys
(``norm1``, ``attn.qkv``, ``attn.proj``, ``norm2``, ``mlp.fc1``, ``mlp.fc2``) of the checkpoint.
"""

import torch
from torch import nn


def mixed_rotation_phases(frequency_table: torch.Tensor, x_positions: torch.Tensor, y_positions: torch.Tensor,
                          head_count: int) -> torch.Tensor:
    """Complex rotation factors of the learned 2D (mixed) rotary embedding.

    Usage:
        Called once per forward pass; slice the result per block with ``[:, block_index]``.

    Args:
        frequency_table: ``(2, depth, head_count * head_dim / 2)`` learned x/y frequencies.
        x_positions: ``(batch, tokens)`` scaled x coordinates of the word centres.
        y_positions: ``(batch, tokens)`` scaled y coordinates of the word centres.
        head_count: Number of attention heads.

    Returns:
        ``(batch, depth, head_count, tokens, head_dim / 2)`` complex64 tensor.
    """
    batch_size, token_count = x_positions.shape
    block_depth = frequency_table.shape[1]
    x_phases = (x_positions.float().unsqueeze(-1).unsqueeze(1) @ frequency_table[0].float().unsqueeze(-2))
    y_phases = (y_positions.float().unsqueeze(-1).unsqueeze(1) @ frequency_table[1].float().unsqueeze(-2))
    x_phases = x_phases.view(batch_size, block_depth, token_count, head_count, -1).permute(0, 1, 3, 2, 4)
    y_phases = y_phases.view(batch_size, block_depth, token_count, head_count, -1).permute(0, 1, 3, 2, 4)
    return torch.polar(torch.ones_like(x_phases), x_phases + y_phases)


def rotate_queries_keys(query_tensor: torch.Tensor, key_tensor: torch.Tensor,
                        rotation_phases: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary phases to queries and keys (pairs of channels as complex numbers).

    Args:
        query_tensor: ``(batch, heads, tokens, head_dim)``.
        key_tensor: Same shape as ``query_tensor``.
        rotation_phases: ``(batch, heads, tokens, head_dim / 2)`` complex factors.

    Returns:
        Rotated ``(query, key)`` with the input dtype.
    """
    complex_queries = torch.view_as_complex(query_tensor.float().reshape(*query_tensor.shape[:-1], -1, 2))
    complex_keys = torch.view_as_complex(key_tensor.float().reshape(*key_tensor.shape[:-1], -1, 2))
    rotated_queries = torch.view_as_real(complex_queries * rotation_phases).flatten(3)
    rotated_keys = torch.view_as_real(complex_keys * rotation_phases).flatten(3)
    return rotated_queries.type_as(query_tensor), rotated_keys.type_as(key_tensor)


class FeedForward(nn.Module):
    """Two-layer MLP with ReLU (timm ``Mlp`` with ``act_layer=nn.ReLU``, no dropout at inference)."""

    def __init__(self, embedding_size: int, hidden_size: int) -> None:
        """Create the layers.

        Args:
            embedding_size: Input/output width.
            hidden_size: Hidden width.
        """
        super().__init__()
        self.fc1 = nn.Linear(embedding_size, hidden_size)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, embedding_size)

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        """Apply the MLP.

        Args:
            token_features: ``(batch, tokens, embedding_size)``.

        Returns:
            Tensor of the same shape.
        """
        return self.fc2(self.activation(self.fc1(token_features)))


class RotaryAttention(nn.Module):
    """Multi-head self-attention with 2D rotary position embeddings on queries and keys."""

    def __init__(self, embedding_size: int, head_count: int) -> None:
        """Create the projections.

        Args:
            embedding_size: Token width (divisible by ``head_count``).
            head_count: Number of heads.
        """
        super().__init__()
        self.head_count = head_count
        self.scale_factor = (embedding_size // head_count) ** -0.5
        self.qkv = nn.Linear(embedding_size, embedding_size * 3)
        self.proj = nn.Linear(embedding_size, embedding_size)

    def forward(self, token_features: torch.Tensor, rotation_phases: torch.Tensor) -> torch.Tensor:
        """Attend over the tokens of each context window.

        Args:
            token_features: ``(batch, tokens, embedding_size)``.
            rotation_phases: ``(batch, heads, tokens, head_dim / 2)`` complex factors.

        Returns:
            Tensor of the same shape as ``token_features``.
        """
        batch_size, token_count, channel_count = token_features.shape
        projected = self.qkv(token_features).reshape(batch_size, token_count, 3, self.head_count,
                                                     channel_count // self.head_count).permute(2, 0, 3, 1, 4)
        query_tensor, key_tensor = rotate_queries_keys(projected[0], projected[1], rotation_phases)
        attention_weights = ((query_tensor * self.scale_factor) @ key_tensor.transpose(-2, -1)).softmax(dim=-1)
        attended = (attention_weights @ projected[2]).transpose(1, 2).reshape(batch_size, token_count, channel_count)
        return self.proj(attended)


class RotaryBlock(nn.Module):
    """Pre-norm transformer block (attention + MLP, residual, no layer scale) of the RoPE-MixAB module."""

    def __init__(self, embedding_size: int, head_count: int) -> None:
        """Create the block (MLP ratio 1, as in the official model).

        Args:
            embedding_size: Token width.
            head_count: Number of heads.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_size)
        self.attn = RotaryAttention(embedding_size, head_count)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.mlp = FeedForward(embedding_size, embedding_size)

    def forward(self, token_features: torch.Tensor, rotation_phases: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            token_features: ``(batch, tokens, embedding_size)``.
            rotation_phases: Complex factors for this block.

        Returns:
            Tensor of the same shape.
        """
        token_features = token_features + self.attn(self.norm1(token_features), rotation_phases)
        return token_features + self.mlp(self.norm2(token_features))
