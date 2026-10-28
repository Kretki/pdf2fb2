"""One epoch of fine-tuning: weighted cross-entropy over first sub-words, AdamW, linear warm-up, bf16 autocast."""

import random
from collections.abc import Callable
from dataclasses import dataclass

import torch

from token_verifier.datasets.example_files import IGNORE_INDEX
from token_verifier.datasets.token_encoding import EncodedWindow
from token_verifier.networks.model_loading import LoadedVerifier
from token_verifier.verifier_settings import TrainingSettings

ProgressCallback = Callable[[int, int, float], None]


@dataclass
class OptimizerState:
    """Optimizer and schedule shared by all epochs.

    Attributes:
        weight_optimizer: AdamW over the model parameters.
        rate_schedule: Learning-rate schedule (stepped once per update).
        loss_function: Weighted cross-entropy.
    """

    weight_optimizer: torch.optim.Optimizer
    rate_schedule: torch.optim.lr_scheduler.LRScheduler
    loss_function: torch.nn.Module


def collate_windows(encoded_windows: list[EncodedWindow], padding_identifier: int,
                    torch_device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad windows into training tensors.

    Args:
        encoded_windows: Windows of one batch.
        padding_identifier: Tokenizer pad id.
        torch_device: Target device.

    Returns:
        ``(input_ids, attention_mask, labels)``; padding is ignored by the loss.
    """
    longest_window = max(len(encoded_window.input_identifiers) for encoded_window in encoded_windows)
    padding_sizes = [longest_window - len(encoded_window.input_identifiers) for encoded_window in encoded_windows]
    input_rows = [list(encoded_window.input_identifiers) + [padding_identifier] * padding_size
                  for encoded_window, padding_size in zip(encoded_windows, padding_sizes, strict=True)]
    mask_rows = [[1] * len(encoded_window.input_identifiers) + [0] * padding_size
                 for encoded_window, padding_size in zip(encoded_windows, padding_sizes, strict=True)]
    label_rows = [list(encoded_window.label_identifiers) + [IGNORE_INDEX] * padding_size
                  for encoded_window, padding_size in zip(encoded_windows, padding_sizes, strict=True)]
    return (torch.tensor(input_rows, device=torch_device), torch.tensor(mask_rows, device=torch_device),
            torch.tensor(label_rows, device=torch_device))


def batch_loss(loaded_verifier: LoadedVerifier, batch_windows: list[EncodedWindow], optimizer_state: OptimizerState,
               mixed_precision: bool) -> torch.Tensor:
    """Loss of one batch (forward pass only).

    Args:
        loaded_verifier: Model in training mode.
        batch_windows: Windows of the batch.
        optimizer_state: Holds the loss function.
        mixed_precision: Use bfloat16 autocast on CUDA.

    Returns:
        Scalar loss tensor.
    """
    padding_identifier = loaded_verifier.text_tokenizer.pad_token_id or 0
    input_tensor, mask_tensor, label_tensor = collate_windows(batch_windows, padding_identifier,
                                                              loaded_verifier.torch_device)
    autocast_enabled = mixed_precision and loaded_verifier.torch_device.type == "cuda"
    with torch.autocast(device_type=loaded_verifier.torch_device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
        batch_logits = loaded_verifier.encoder_model(input_ids=input_tensor, attention_mask=mask_tensor).logits
    return optimizer_state.loss_function(batch_logits.float().reshape(-1, batch_logits.shape[-1]),
                                         label_tensor.reshape(-1))


def train_epoch(loaded_verifier: LoadedVerifier, training_windows: list[EncodedWindow],
                optimizer_state: OptimizerState, training_settings: TrainingSettings,
                shuffle_generator: random.Random, progress_callback: ProgressCallback) -> float:
    """Run one pass over the training windows.

    Args:
        loaded_verifier: Model (switched to training mode here).
        training_windows: Encoded training windows.
        optimizer_state: Optimizer, schedule and loss.
        training_settings: Batch size, accumulation, precision.
        shuffle_generator: Shuffles the window order.
        progress_callback: Receives ``(done batches, total batches, running loss)``.

    Returns:
        Mean loss of the epoch.
    """
    loaded_verifier.encoder_model.train()
    window_order = list(range(len(training_windows)))
    shuffle_generator.shuffle(window_order)
    batch_starts = range(0, len(window_order), training_settings.batch_size)
    summed_loss = 0.0
    for batch_number, batch_start in enumerate(batch_starts, start=1):
        batch_windows = [training_windows[window_index]
                         for window_index in window_order[batch_start:batch_start + training_settings.batch_size]]
        loss_value = batch_loss(loaded_verifier, batch_windows, optimizer_state, training_settings.mixed_precision)
        (loss_value / training_settings.gradient_accumulation).backward()
        summed_loss += float(loss_value.detach())
        if batch_number % training_settings.gradient_accumulation == 0 or batch_number == len(batch_starts):
            torch.nn.utils.clip_grad_norm_(loaded_verifier.encoder_model.parameters(), 1.0)
            optimizer_state.weight_optimizer.step()
            optimizer_state.rate_schedule.step()
            optimizer_state.weight_optimizer.zero_grad(set_to_none=True)
        progress_callback(batch_number, len(batch_starts), summed_loss / batch_number)
    loaded_verifier.encoder_model.eval()
    return summed_loss / max(len(batch_starts), 1)
