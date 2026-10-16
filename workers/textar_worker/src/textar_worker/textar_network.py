"""TexTAR network (ICDAR 2025, MIT licence): ResNet-18 word features, transformer context, RoPE-MixAB, two heads.

Architecture and hyper-parameters follow ``src/architectures/textar.py`` of github.com/tex-tar/tex-tar:
256-d word embeddings, 6 encoder layers with 8 heads, 2 RoPE-Mixed blocks, coordinate weights
60 (x) and 14 (y), context windows of 125 words, word crops resized to 128x96 (H x W).
"""

from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18

from textar_worker.rotary_layers import RotaryBlock, mixed_rotation_phases

EMBEDDING_SIZE = 256
ENCODER_LAYERS = 6
HEAD_COUNT = 8
ROTARY_BLOCKS = 2
CLASS_COUNT = 4
X_POSITION_WEIGHT = 60.0
Y_POSITION_WEIGHT = 14.0
CHECKPOINT_FILE = "TexTAR-trained.pt"


def classification_head() -> nn.Sequential:
    """One attribute-group head (dropout layers kept so the checkpoint indexes match).

    Returns:
        MLP 512 -> 128 -> 64 -> 32 -> 4.
    """
    return nn.Sequential(
        nn.Linear(2 * EMBEDDING_SIZE, 128), nn.Dropout(0.2), nn.ReLU(),
        nn.Linear(128, 64), nn.Dropout(0.2), nn.ReLU(),
        nn.Linear(64, 32), nn.Dropout(0.2), nn.ReLU(),
        nn.Linear(32, CLASS_COUNT))


class ContextEncoder(nn.Module):
    """Transformer encoder over the word embeddings of a context window (``model2`` in the checkpoint)."""

    def __init__(self) -> None:
        """Create the encoder (post-norm, ReLU, feed-forward 256)."""
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=EMBEDDING_SIZE, nhead=HEAD_COUNT,
                                                   dim_feedforward=EMBEDDING_SIZE, batch_first=True, dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=ENCODER_LAYERS,
                                                         enable_nested_tensor=False)

    def forward(self, word_embeddings: torch.Tensor) -> torch.Tensor:
        """Encode the window.

        Args:
            word_embeddings: ``(windows, tokens, 256)``.

        Returns:
            Tensor of the same shape.
        """
        return self.transformer_encoder(word_embeddings)


class TexTarNetwork(nn.Module):
    """Predicts the bold/italic group and the underline/strikeout group of every word in a window.

    Usage:
        Build with ``load_textar_network`` and call with word crops plus their window coordinates.
    """

    def __init__(self) -> None:
        """Create all sub-networks with the checkpoint's parameter names."""
        super().__init__()
        self.model1 = resnet18(weights=None)
        self.model1.fc = nn.Sequential(nn.Linear(self.model1.fc.in_features, EMBEDDING_SIZE), nn.ReLU())
        self.model2 = ContextEncoder()
        self.model3_head = classification_head()
        self.model4_head = classification_head()
        self.blocks = nn.ModuleList([RotaryBlock(EMBEDDING_SIZE, HEAD_COUNT) for _ in range(ROTARY_BLOCKS)])
        self.norm = nn.LayerNorm(EMBEDDING_SIZE, eps=1e-6)
        self.freqs = nn.Parameter(torch.zeros(2, ROTARY_BLOCKS, EMBEDDING_SIZE // 2))

    def forward(self, word_crops: torch.Tensor, word_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Classify the words of a batch of context windows.

        Args:
            word_crops: ``(windows * tokens, 3, 128, 96)`` float crops in 0..1.
            word_positions: ``(windows, tokens, 2)`` word centres normalised to 0..1 inside each window.

        Returns:
            ``(emphasis_logits, line_logits)``, each ``(windows * tokens, 4)``.
        """
        window_count, token_count, _ = word_positions.shape
        word_embeddings = self.model1(word_crops).view(window_count, token_count, EMBEDDING_SIZE)
        context_features = self.model2(word_embeddings)
        rotation_phases = mixed_rotation_phases(self.freqs, X_POSITION_WEIGHT * word_positions[..., 0],
                                                Y_POSITION_WEIGHT * word_positions[..., 1], HEAD_COUNT)
        rotary_features = context_features
        for block_index, rotary_block in enumerate(self.blocks):
            rotary_features = rotary_block(rotary_features, rotation_phases[:, block_index])
        rotary_features = self.norm(rotary_features).reshape(window_count * token_count, EMBEDDING_SIZE)
        joined_features = torch.cat([rotary_features, context_features.reshape(-1, EMBEDDING_SIZE)], dim=-1)
        return self.model3_head(joined_features), self.model4_head(joined_features)


def load_textar_network(textar_directory: Path, torch_device: torch.device) -> TexTarNetwork:
    """Load the fine-tuned TexTAR checkpoint (tensors only, no pickled code).

    Args:
        textar_directory: Directory containing ``TexTAR-trained.pt``.
        torch_device: Device to run on.

    Returns:
        Network in eval mode on ``torch_device``.

    Raises:
        FileNotFoundError: If the checkpoint is missing.
        RuntimeError: If the checkpoint does not match the architecture.
    """
    checkpoint_path = textar_directory / CHECKPOINT_FILE
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"TexTAR checkpoint not found: {checkpoint_path}")
    # weights_only=True: torch's restricted unpickler accepts tensors and plain containers only, no code.
    checkpoint_content = torch.load(  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        checkpoint_path, map_location="cpu", weights_only=True)
    textar_network = TexTarNetwork()
    textar_network.load_state_dict(checkpoint_content["model_state_dict"], strict=True)
    return textar_network.eval().to(torch_device)
