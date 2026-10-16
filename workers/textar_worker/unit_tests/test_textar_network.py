"""Shape and checkpoint-compatibility tests of the re-implemented TexTAR network."""

import torch

from textar_worker.context_windows import WINDOW_SIZE
from textar_worker.textar_network import TexTarNetwork


def test_forward_returns_two_heads_per_word() -> None:
    """One window of 125 words yields 125 x 4 logits per attribute group."""
    textar_network = TexTarNetwork().eval()
    with torch.inference_mode():
        emphasis_logits, line_logits = textar_network(torch.rand(WINDOW_SIZE, 3, 128, 96),
                                                      torch.rand(1, WINDOW_SIZE, 2))
    assert emphasis_logits.shape == (WINDOW_SIZE, 4)
    assert line_logits.shape == (WINDOW_SIZE, 4)


def test_parameter_names_match_the_released_checkpoint() -> None:
    """The state dict exposes the key families of ``TexTAR-trained.pt``."""
    parameter_names = set(TexTarNetwork().state_dict())
    for expected_name in ("freqs", "model1.fc.0.weight", "model2.transformer_encoder.layers.5.linear2.bias",
                          "model3_head.9.weight", "model4_head.0.bias", "blocks.1.attn.qkv.weight",
                          "blocks.0.mlp.fc2.bias", "norm.weight"):
        assert expected_name in parameter_names
