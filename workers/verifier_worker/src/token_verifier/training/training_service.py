"""``train`` command: fine-tune the base encoder (one model or an ensemble of seeds) and keep the best epoch.

Every member gets a fresh head on the pretrained backbone (word embeddings frozen when configured), runs the
two-stage schedule of ``training_stages`` and saves its best (EMA) weights with thresholds tuned on the
selection data. With ``ensemble_members > 1`` the members go to ``member_<n>/`` sub-directories, ``ensemble.json``
lists them and the thresholds of the averaged ensemble are tuned at the end.
"""

import random
import shutil
from pathlib import Path

import torch

from token_verifier.evaluation.evaluation_service import run_tuning
from token_verifier.networks.model_loading import (
    ENSEMBLE_FILE,
    EvidenceSources,
    WindowShape,
    build_assembler,
    create_verifier,
    save_verifier,
    torch_device_of,
    write_ensemble,
    write_thresholds,
)
from token_verifier.training.parameter_groups import freeze_word_embeddings
from token_verifier.training.training_data import training_corpus
from token_verifier.training.training_stages import MemberRun, fit_member
from token_verifier.verifier_settings import VerifierSettings


def evidence_sources(verifier_settings: VerifierSettings) -> EvidenceSources:
    """Evidence provider directories of the settings.

    Args:
        verifier_settings: All settings.

    Returns:
        Language-model and visual model directories (None when disabled).
    """
    path_settings = verifier_settings.path_settings
    return EvidenceSources(path_settings.language_model, path_settings.visual_model)


def window_shape_of(verifier_settings: VerifierSettings) -> WindowShape:
    """Window settings of the settings.

    Args:
        verifier_settings: All settings.

    Returns:
        Window shape.
    """
    training_settings = verifier_settings.training_settings
    return WindowShape(training_settings.window_units, training_settings.max_tokens, training_settings.context_units)


def train_member(verifier_settings: VerifierSettings, output_directory: Path, random_seed: int) -> float:
    """Train one model and save its best weights.

    Args:
        verifier_settings: All settings.
        output_directory: Where the member is saved.
        random_seed: Seed of initialisation, shuffling and evidence dropout.

    Returns:
        Best validation mean F1.
    """
    training_settings = verifier_settings.training_settings
    torch.manual_seed(random_seed)
    torch_device = torch_device_of(training_settings.device_name)
    evidence_dirs = evidence_sources(verifier_settings)
    feature_assembler = build_assembler(evidence_dirs, torch_device, training_settings.evidence_dropout, random_seed)
    loaded_verifier = create_verifier(verifier_settings.path_settings.base_model, torch_device, feature_assembler,
                                      training_settings.backbone_kind)
    if training_settings.freeze_word_embeddings:
        freeze_word_embeddings(loaded_verifier.backbone_adapter.encoder_module)
    training_corpus_value = training_corpus(verifier_settings.path_settings.data_directory,
                                            loaded_verifier.backbone_adapter, feature_assembler, training_settings)
    window_shape = window_shape_of(verifier_settings)

    def save_best(best_weights: object, class_thresholds: dict[str, float]) -> None:
        """Save the best weights and their thresholds.

        Args:
            best_weights: Verifier with the best (EMA) weights.
            class_thresholds: Thresholds tuned for them.

        Returns:
            None.
        """
        save_verifier(best_weights, output_directory, evidence_dirs, window_shape)
        write_thresholds(output_directory, class_thresholds)

    member_run = MemberRun(training_corpus_value, training_settings, feature_assembler, window_shape, save_best,
                           random.Random(random_seed), None)  # nosec B311  # noqa: S311 - shuffling, not cryptography
    return fit_member(loaded_verifier, member_run)


def run_training(verifier_settings: VerifierSettings) -> Path:
    """Fine-tune and save the best epoch of every member.

    Usage:
        ``python -m token_verifier train`` (GPU recommended).

    Args:
        verifier_settings: All settings.

    Returns:
        The output model directory.
    """
    output_directory = verifier_settings.path_settings.output_model
    member_count = max(1, verifier_settings.training_settings.ensemble_members)
    base_seed = verifier_settings.training_settings.random_seed
    if member_count == 1:
        (output_directory / ENSEMBLE_FILE).unlink(missing_ok=True)
        train_member(verifier_settings, output_directory, base_seed)
        return output_directory
    member_names = [f"member_{member_index}" for member_index in range(member_count)]
    for member_index, member_name in enumerate(member_names):
        print(f"training ensemble member {member_index + 1}/{member_count}", flush=True)
        shutil.rmtree(output_directory / member_name, ignore_errors=True)
        train_member(verifier_settings, output_directory / member_name, base_seed + member_index)
    write_ensemble(output_directory, member_names)
    print(f"ensemble thresholds: {run_tuning(verifier_settings)}", flush=True)
    return output_directory
