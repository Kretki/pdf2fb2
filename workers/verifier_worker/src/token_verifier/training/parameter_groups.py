"""Optimizer parameter groups: frozen word embeddings and layer-wise learning-rate decay.

Freezing the pretrained word embeddings keeps the multilingual embedding space as pretrained (similar words of
different languages stay close, and languages absent from the training data keep working). Lower layers, which
hold general language knowledge, get smaller learning rates than upper layers and the head.
"""

import re

from torch import nn

LAYER_PATTERN = re.compile(r"\.(?:layer|block)\.(\d+)\.")
NO_DECAY_MARKERS = ("bias", "LayerNorm", "layer_norm", "norm.weight")


def freeze_word_embeddings(encoder_module: nn.Module) -> int:
    """Stop training the input word embeddings.

    Args:
        encoder_module: Backbone encoder.

    Returns:
        Number of frozen parameters.
    """
    embedding_weight = encoder_module.get_input_embeddings().weight
    embedding_weight.requires_grad_(False)
    return embedding_weight.numel()


def parameter_depth(parameter_name: str, layer_count: int) -> int:
    """Depth of a parameter: 0 embeddings, 1..layers transformer layers, layers+1 head and top norms.

    Args:
        parameter_name: Name within the verifier network.
        layer_count: Transformer layers of the backbone.

    Returns:
        Depth.
    """
    if not parameter_name.startswith("encoder_module."):
        return layer_count + 1
    layer_match = LAYER_PATTERN.search(parameter_name)
    if layer_match:
        return int(layer_match.group(1)) + 1
    return 0 if "embed" in parameter_name or "shared" in parameter_name else layer_count


def parameter_groups(verifier_network: nn.Module, layer_count: int, learning_rate: float, weight_decay: float,
                     layer_decay: float) -> list[dict]:
    """AdamW groups with per-depth learning rates; biases and norms get no weight decay.

    Usage:
        Called by ``training_loop.optimizer_state`` for every training stage.

    Args:
        verifier_network: Network (frozen parameters are skipped).
        layer_count: Transformer layers of the backbone.
        learning_rate: Learning rate of the head.
        weight_decay: AdamW weight decay.
        layer_decay: Factor per depth step below the head (1 = same rate everywhere).

    Returns:
        Parameter groups.
    """
    grouped_parameters: dict[tuple[int, bool], list[nn.Parameter]] = {}
    for parameter_name, parameter_value in verifier_network.named_parameters():
        if not parameter_value.requires_grad:
            continue
        group_key = (parameter_depth(parameter_name, layer_count),
                     not any(marker_text in parameter_name for marker_text in NO_DECAY_MARKERS))
        grouped_parameters.setdefault(group_key, []).append(parameter_value)
    return [{"params": group_parameters,
             "lr": learning_rate * layer_decay ** (layer_count + 1 - depth_value),
             "weight_decay": weight_decay if uses_decay else 0.0}
            for (depth_value, uses_decay), group_parameters in sorted(grouped_parameters.items())]
