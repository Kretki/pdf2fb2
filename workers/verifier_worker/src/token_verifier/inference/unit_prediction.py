"""Class probabilities of every unit of many texts, averaged over the members of a verifier bundle.

Used by pipeline jobs, evaluation, threshold tuning and the relabelling (confident learning) command.
"""

from collections.abc import Callable

import torch

from token_verifier.datasets.token_encoding import EncodedWindow, merge_window_predictions, text_windows
from token_verifier.datasets.verify_texts import VerifyText
from token_verifier.datasets.window_batches import WindowBatch, collate_windows
from token_verifier.features.feature_assembly import UnitEvidence
from token_verifier.networks.model_loading import LoadedVerifier, VerifierBundle, WindowShape

PREDICTION_BATCH = 32
ProgressCallback = Callable[[int, int], None]


def network_logits(loaded_verifier: LoadedVerifier, window_batch: WindowBatch,
                   mixed_precision: bool = True) -> torch.Tensor:
    """Unit logits of a batch.

    Usage:
        Shared by training and prediction.

    Args:
        loaded_verifier: Backbone adapter and network.
        window_batch: Batch tensors.
        mixed_precision: Use bfloat16 autocast on CUDA.

    Returns:
        ``(batch, units, 3)`` float logits.
    """
    autocast_enabled = mixed_precision and loaded_verifier.torch_device.type == "cuda"
    with torch.autocast(device_type=loaded_verifier.torch_device.type, dtype=torch.bfloat16,
                        enabled=autocast_enabled):
        piece_states = loaded_verifier.backbone_adapter.hidden_states(window_batch.input_tensor,
                                                                      window_batch.mask_tensor)
        unit_logits = loaded_verifier.verifier_network(piece_states, window_batch.piece_units,
                                                       window_batch.feature_tensor, window_batch.visual_tensor)
    return unit_logits.float()


@torch.inference_mode()
def window_probabilities(loaded_verifier: LoadedVerifier, encoded_windows: list[EncodedWindow],
                         progress_callback: ProgressCallback | None = None) -> list[list[list[float]]]:
    """Softmax class probabilities of every unit of every window.

    Args:
        loaded_verifier: Network.
        encoded_windows: Windows to run.
        progress_callback: Receives ``(done windows, total windows)``.

    Returns:
        Per window, per unit: three probabilities.
    """
    window_results: list[list[list[float]]] = []
    for batch_start in range(0, len(encoded_windows), PREDICTION_BATCH):
        batch_windows = encoded_windows[batch_start:batch_start + PREDICTION_BATCH]
        window_batch = collate_windows(batch_windows, loaded_verifier.backbone_adapter.padding_identifier,
                                       loaded_verifier.torch_device)
        batch_probabilities = torch.softmax(network_logits(loaded_verifier, window_batch), dim=-1).cpu().tolist()
        window_results.extend(unit_rows[:batch_window.unit_count]
                              for unit_rows, batch_window in zip(batch_probabilities, batch_windows, strict=True))
        if progress_callback is not None:
            progress_callback(len(window_results), len(encoded_windows))
    return window_results


def member_predictions(loaded_verifier: LoadedVerifier, verify_texts: list[VerifyText],
                       text_evidence: list[UnitEvidence], window_shape: WindowShape,
                       progress_callback: ProgressCallback | None) -> list[list[list[float]]]:
    """Probabilities of one member.

    Args:
        loaded_verifier: Member network.
        verify_texts: Texts.
        text_evidence: Their evidence.
        window_shape: Window settings.
        progress_callback: Progress receiver.

    Returns:
        Per text, per unit: three probabilities.
    """
    window_settings = (window_shape.window_units, window_shape.max_tokens, window_shape.context_units)
    window_lists = [text_windows(loaded_verifier.backbone_adapter, verify_text, unit_evidence, window_settings, True)
                    for verify_text, unit_evidence in zip(verify_texts, text_evidence, strict=True)]
    flat_probabilities = window_probabilities(loaded_verifier, [encoded_window for window_list in window_lists
                                                                for encoded_window in window_list], progress_callback)
    text_results, read_offset = [], 0
    for verify_text, window_list in zip(verify_texts, window_lists, strict=True):
        text_results.append(merge_window_predictions(len(verify_text.unit_texts), window_list,
                                                     flat_probabilities[read_offset:read_offset + len(window_list)]))
        read_offset += len(window_list)
    return text_results


def averaged_rows(member_results: list[list[list[list[float]]]]) -> list[list[list[float]]]:
    """Average member probabilities unit by unit.

    Args:
        member_results: Per member, per text, per unit: probabilities.

    Returns:
        Per text, per unit: mean probabilities.
    """
    member_count = len(member_results)
    return [[[sum(class_values) / member_count for class_values in zip(*unit_rows, strict=True)]
             for unit_rows in zip(*text_rows, strict=True)]
            for text_rows in zip(*member_results, strict=True)]


def predict_units(verifier_bundle: VerifierBundle, verify_texts: list[VerifyText],
                  progress_callback: ProgressCallback | None = None) -> list[list[list[float]]]:
    """Probabilities of every unit of every text.

    Usage:
        Called by the pipeline job and by evaluation, tuning and relabelling.

    Args:
        verifier_bundle: Members, evidence providers and window settings.
        verify_texts: Texts to label.
        progress_callback: Receives ``(done windows, total windows)`` of the current member.

    Returns:
        Per text, per unit: ``[correct, misread, inserted]``.
    """
    text_evidence = [verifier_bundle.feature_assembler.assemble(verify_text) for verify_text in verify_texts]
    return averaged_rows([member_predictions(loaded_verifier, verify_texts, text_evidence,
                                             verifier_bundle.window_shape, progress_callback)
                          for loaded_verifier in verifier_bundle.verifier_members])
