"""One epoch of fine-tuning: weighted unit cross-entropy, optional R-Drop, AdamW, warm-up, EMA of the weights.

* R-Drop: every batch runs twice with different dropout; a symmetric KL term makes both predictions agree, which
  regularises against noisy labels.
* EMA: an exponential moving average of the weights is kept and used for validation and saving; it is smoother
  than the last weights.
"""

import copy
import dataclasses
import random
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from transformers import get_linear_schedule_with_warmup

from token_verifier.datasets.example_files import IGNORE_INDEX
from token_verifier.datasets.token_encoding import EncodedWindow
from token_verifier.datasets.window_batches import collate_windows
from token_verifier.inference.unit_prediction import network_logits
from token_verifier.networks.model_loading import LoadedVerifier
from token_verifier.training.parameter_groups import parameter_groups
from token_verifier.verifier_settings import TrainingSettings

ProgressCallback = Callable[[int, int, float], None]
GRADIENT_CLIP = 1.0


@dataclass
class OptimizerState:
    """Optimizer and schedule of one training stage.

    Attributes:
        weight_optimizer: AdamW over the trainable parameters.
        rate_schedule: Learning-rate schedule (stepped once per update).
        loss_function: Weighted cross-entropy over units.
        rdrop_weight: Weight of the R-Drop term (0 disables the second pass).
    """

    weight_optimizer: torch.optim.Optimizer
    rate_schedule: torch.optim.lr_scheduler.LRScheduler
    loss_function: torch.nn.Module
    rdrop_weight: float = 0.0


def optimizer_state(loaded_verifier: LoadedVerifier, training_settings: TrainingSettings, learning_rate: float,
                    update_count: int) -> OptimizerState:
    """AdamW with layer-wise rates, warm-up schedule and weighted loss for a stage.

    Args:
        loaded_verifier: Network to train.
        training_settings: Weight decay, layer decay, warm-up, class weights, R-Drop weight.
        learning_rate: Peak learning rate of the stage.
        update_count: Optimizer updates of the stage.

    Returns:
        Optimizer state.
    """
    weight_optimizer = torch.optim.AdamW(parameter_groups(
        loaded_verifier.verifier_network, loaded_verifier.backbone_adapter.layer_count, learning_rate,
        training_settings.weight_decay, training_settings.layer_decay))
    warmup_updates = int(update_count * training_settings.warmup_ratio)
    rate_schedule = get_linear_schedule_with_warmup(weight_optimizer, warmup_updates, max(update_count, 1))
    class_weights = torch.tensor(training_settings.class_weights, device=loaded_verifier.torch_device)
    return OptimizerState(weight_optimizer, rate_schedule,
                          torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX),
                          training_settings.rdrop_weight)


def averaged_weights(loaded_verifier: LoadedVerifier, ema_decay: float) -> AveragedModel | None:
    """EMA copy of the network.

    Args:
        loaded_verifier: Network being trained.
        ema_decay: Decay per update (0 disables the average).

    Returns:
        The averaged model, or None.
    """
    if ema_decay <= 0:
        return None
    return AveragedModel(loaded_verifier.verifier_network, multi_avg_fn=get_ema_multi_avg_fn(ema_decay),
                         use_buffers=True)


def averaged_view(loaded_verifier: LoadedVerifier, ema_model: AveragedModel | None) -> LoadedVerifier:
    """Verifier that runs the EMA weights (or the live weights without EMA).

    Usage:
        Used for validation and saving.

    Args:
        loaded_verifier: Live verifier.
        ema_model: Averaged copy, or None.

    Returns:
        A verifier sharing the tokenizer but using the averaged encoder and head.
    """
    if ema_model is None:
        return loaded_verifier
    averaged_network = ema_model.module
    viewed_adapter = copy.copy(loaded_verifier.backbone_adapter)
    viewed_adapter.encoder_module = averaged_network.encoder_module
    return dataclasses.replace(loaded_verifier, backbone_adapter=viewed_adapter, verifier_network=averaged_network)


def consistency_loss(first_logits: torch.Tensor, second_logits: torch.Tensor,
                     label_tensor: torch.Tensor) -> torch.Tensor:
    """Symmetric KL divergence between two dropout passes over the labelled units.

    Args:
        first_logits: ``(batch, units, 3)``.
        second_logits: ``(batch, units, 3)``.
        label_tensor: ``(batch, units)`` labels (``IGNORE_INDEX`` units are skipped).

    Returns:
        Scalar loss.
    """
    unit_mask = label_tensor != IGNORE_INDEX
    if not bool(unit_mask.any()):
        return first_logits.sum() * 0.0
    first_log = torch.log_softmax(first_logits[unit_mask], dim=-1)
    second_log = torch.log_softmax(second_logits[unit_mask], dim=-1)
    forward_term = torch.nn.functional.kl_div(first_log, second_log, log_target=True, reduction="batchmean")
    backward_term = torch.nn.functional.kl_div(second_log, first_log, log_target=True, reduction="batchmean")
    return (forward_term + backward_term) / 2


def batch_loss(loaded_verifier: LoadedVerifier, batch_windows: list[EncodedWindow], loop_state: OptimizerState,
               mixed_precision: bool) -> torch.Tensor:
    """Loss of one batch (forward passes only).

    Args:
        loaded_verifier: Network in training mode.
        batch_windows: Windows of the batch.
        loop_state: Loss function and R-Drop weight.
        mixed_precision: Use bfloat16 autocast on CUDA.

    Returns:
        Scalar loss tensor.
    """
    window_batch = collate_windows(batch_windows, loaded_verifier.backbone_adapter.padding_identifier,
                                   loaded_verifier.torch_device)
    flat_labels = window_batch.label_tensor.reshape(-1)
    first_logits = network_logits(loaded_verifier, window_batch, mixed_precision)
    first_loss = loop_state.loss_function(first_logits.reshape(-1, first_logits.shape[-1]), flat_labels)
    if loop_state.rdrop_weight <= 0:
        return first_loss
    second_logits = network_logits(loaded_verifier, window_batch, mixed_precision)
    second_loss = loop_state.loss_function(second_logits.reshape(-1, second_logits.shape[-1]), flat_labels)
    return (first_loss + second_loss) / 2 + loop_state.rdrop_weight * consistency_loss(
        first_logits, second_logits, window_batch.label_tensor)


def optimizer_step(loaded_verifier: LoadedVerifier, loop_state: OptimizerState,
                   ema_model: AveragedModel | None) -> None:
    """Clip, step, schedule, update the EMA and clear gradients.

    Args:
        loaded_verifier: Network being trained.
        loop_state: Optimizer and schedule.
        ema_model: EMA copy, or None.

    Returns:
        None.
    """
    torch.nn.utils.clip_grad_norm_(loaded_verifier.verifier_network.parameters(), GRADIENT_CLIP)
    loop_state.weight_optimizer.step()
    loop_state.rate_schedule.step()
    loop_state.weight_optimizer.zero_grad(set_to_none=True)
    if ema_model is not None:
        ema_model.update_parameters(loaded_verifier.verifier_network)


def train_epoch(loaded_verifier: LoadedVerifier, training_windows: list[EncodedWindow], loop_state: OptimizerState,
                training_settings: TrainingSettings, shuffle_generator: random.Random,
                ema_model: AveragedModel | None = None,
                progress_callback: ProgressCallback | None = None) -> float:
    """Run one pass over the training windows.

    Usage:
        Called by ``training_stages`` once per epoch.

    Args:
        loaded_verifier: Network (switched to training mode here).
        training_windows: Encoded training windows.
        loop_state: Optimizer, schedule, loss.
        training_settings: Batch size, accumulation, precision.
        shuffle_generator: Shuffles the window order.
        ema_model: EMA copy updated after every optimizer step, or None.
        progress_callback: Receives ``(done batches, total batches, running loss)``.

    Returns:
        Mean loss of the epoch.
    """
    loaded_verifier.verifier_network.train()
    window_order = list(range(len(training_windows)))
    shuffle_generator.shuffle(window_order)
    batch_starts = range(0, len(window_order), training_settings.batch_size)
    summed_loss = 0.0
    for batch_number, batch_start in enumerate(batch_starts, start=1):
        batch_windows = [training_windows[window_index]
                         for window_index in window_order[batch_start:batch_start + training_settings.batch_size]]
        loss_value = batch_loss(loaded_verifier, batch_windows, loop_state, training_settings.mixed_precision)
        (loss_value / training_settings.gradient_accumulation).backward()
        summed_loss += float(loss_value.detach())
        if batch_number % training_settings.gradient_accumulation == 0 or batch_number == len(batch_starts):
            optimizer_step(loaded_verifier, loop_state, ema_model)
        if progress_callback is not None:
            progress_callback(batch_number, len(batch_starts), summed_loss / batch_number)
    loaded_verifier.verifier_network.eval()
    return summed_loss / max(len(batch_starts), 1)
