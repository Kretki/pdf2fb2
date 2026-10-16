"""Checkpoint round trip through safetensors (no pickle) keeps weights and metadata."""

from pathlib import Path

import torch

from readermarks.networks.checkpoint_store import load_checkpoint, save_checkpoint
from readermarks.networks.spotter_head import SpotterHead


def test_round_trip(tmp_path: Path) -> None:
    """Saved and reloaded heads produce identical outputs."""
    spotter_head = SpotterHead(16, 32, 2, 9).eval()
    checkpoint_path = tmp_path / "spotter.safetensors"
    save_checkpoint(checkpoint_path, spotter_head, {"feature_channels": 16, "hidden_channels": 32})
    loaded_state, loaded_meta = load_checkpoint(checkpoint_path)
    reloaded_head = SpotterHead(16, 32, 2, 9).eval()
    reloaded_head.load_state_dict(loaded_state)
    patch_features, valid_mask = torch.randn(1, 16, 5, 7), torch.ones(1, 1, 5, 7)
    original_logits, _ = spotter_head(patch_features, valid_mask)
    reloaded_logits, _ = reloaded_head(patch_features, valid_mask)
    assert loaded_meta == {"feature_channels": 16, "hidden_channels": 32}
    assert torch.equal(original_logits, reloaded_logits)
