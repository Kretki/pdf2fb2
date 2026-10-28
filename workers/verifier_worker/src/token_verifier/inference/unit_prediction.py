"""Class probabilities of every unit of many texts, in batches (used by evaluation and pipeline jobs)."""

from collections.abc import Callable

import torch

from token_verifier.datasets.token_encoding import EncodedWindow, merge_window_predictions, prediction_windows
from token_verifier.networks.model_loading import LoadedVerifier

PREDICTION_BATCH = 32
ProgressCallback = Callable[[int, int], None]


def padded_batch(encoded_windows: list[EncodedWindow], padding_identifier: int,
                 torch_device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad windows into input and attention tensors.

    Args:
        encoded_windows: Windows of one batch.
        padding_identifier: Tokenizer pad id.
        torch_device: Target device.

    Returns:
        ``(input_ids, attention_mask)``.
    """
    longest_window = max(len(encoded_window.input_identifiers) for encoded_window in encoded_windows)
    input_rows = [list(encoded_window.input_identifiers) + [padding_identifier] *
                  (longest_window - len(encoded_window.input_identifiers)) for encoded_window in encoded_windows]
    mask_rows = [[1] * len(encoded_window.input_identifiers) + [0] *
                 (longest_window - len(encoded_window.input_identifiers)) for encoded_window in encoded_windows]
    return (torch.tensor(input_rows, device=torch_device), torch.tensor(mask_rows, device=torch_device))


@torch.inference_mode()
def window_probabilities(loaded_verifier: LoadedVerifier, encoded_windows: list[EncodedWindow],
                         progress_callback: ProgressCallback | None = None) -> list[list[list[float]]]:
    """Softmax class probabilities of every sub-word of every window.

    Args:
        loaded_verifier: Model and tokenizer.
        encoded_windows: Windows to run.
        progress_callback: Receives ``(done windows, total windows)``.

    Returns:
        Per window, per sub-word: three probabilities.
    """
    padding_identifier = loaded_verifier.text_tokenizer.pad_token_id or 0
    window_results: list[list[list[float]]] = []
    for batch_start in range(0, len(encoded_windows), PREDICTION_BATCH):
        batch_windows = encoded_windows[batch_start:batch_start + PREDICTION_BATCH]
        input_tensor, mask_tensor = padded_batch(batch_windows, padding_identifier, loaded_verifier.torch_device)
        batch_logits = loaded_verifier.encoder_model(input_ids=input_tensor, attention_mask=mask_tensor).logits
        batch_probabilities = torch.softmax(batch_logits.float(), dim=-1).cpu().tolist()
        window_results.extend(token_rows[:len(batch_window.input_identifiers)]
                              for token_rows, batch_window in zip(batch_probabilities, batch_windows, strict=True))
        if progress_callback is not None:
            progress_callback(len(window_results), len(encoded_windows))
    return window_results


def predict_units(loaded_verifier: LoadedVerifier, unit_lists: list[list[str]], window_units: int,
                  max_tokens: int, progress_callback: ProgressCallback | None = None) -> list[list[list[float]]]:
    """Probabilities of every unit of every text.

    Usage:
        Called by the pipeline job and by evaluation / threshold tuning.

    Args:
        loaded_verifier: Model and tokenizer.
        unit_lists: Units per text.
        window_units: Units per prediction window.
        max_tokens: Sub-word limit per window.
        progress_callback: Receives ``(done windows, total windows)``.

    Returns:
        Per text, per unit: ``[correct, misread, inserted]``.
    """
    text_windows = [prediction_windows(loaded_verifier.text_tokenizer, unit_texts, window_units, max_tokens)
                    for unit_texts in unit_lists]
    flat_windows = [encoded_window for window_list in text_windows for encoded_window in window_list]
    flat_probabilities = window_probabilities(loaded_verifier, flat_windows, progress_callback)
    text_results, read_offset = [], 0
    for unit_texts, window_list in zip(unit_lists, text_windows, strict=True):
        text_results.append(merge_window_predictions(len(unit_texts), window_list,
                                                     flat_probabilities[read_offset:read_offset + len(window_list)]))
        read_offset += len(window_list)
    return text_results
