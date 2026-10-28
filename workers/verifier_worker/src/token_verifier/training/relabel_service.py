"""``relabel`` command: confident-learning cleanup of the real training labels.

The books of ``aligned_train.jsonl`` are split into book_fold_sets. For every fold a model is trained on synthetic data
and the real examples of the other book_fold_sets, then predicts the held-out books. A unit whose label the fold model
contradicts with probability of at least ``relabel_confidence`` takes the predicted label (most such units are
errors of a machine-OCR reference, not of the pipeline). ``ignore`` units are never changed. The result is
written to ``aligned_train_relabelled.jsonl``; ``train`` prefers it when ``use_relabelled_real`` is set.
"""

import dataclasses
import gc
import hashlib
import random
from collections import Counter
from pathlib import Path

import torch

from token_verifier.datasets.example_files import IGNORE_LABEL, TokenExample, read_examples, write_examples
from token_verifier.datasets.verify_texts import text_from_example
from token_verifier.inference.unit_prediction import predict_units
from token_verifier.networks.model_loading import (
    LoadedVerifier,
    VerifierBundle,
    build_assembler,
    create_verifier,
    torch_device_of,
)
from token_verifier.training.parameter_groups import freeze_word_embeddings
from token_verifier.training.training_data import RELABELLED_FILE, training_corpus
from token_verifier.training.training_service import evidence_sources, window_shape_of
from token_verifier.training.training_stages import MemberRun, fit_member
from token_verifier.verifier_settings import VerifierSettings
from worker_protocol.verify_job_schema import LABEL_NAMES


def book_folds(token_examples: list[TokenExample], fold_count: int) -> list[set[str]]:
    """Book keys of every fold (stable hash order, so reruns give the same book_fold_sets).

    Args:
        token_examples: Real training examples.
        fold_count: Number of book_fold_sets (at least 2).

    Returns:
        One set of book keys per fold.
    """
    book_keys = sorted({token_example.book_key for token_example in token_examples},
                       key=lambda book_key: hashlib.sha256(book_key.encode("utf-8")).hexdigest())
    book_fold_sets: list[set[str]] = [set() for _ in range(max(2, fold_count))]
    for book_position, book_key in enumerate(book_keys):
        book_fold_sets[book_position % len(book_fold_sets)].add(book_key)
    return book_fold_sets


def relabelled_example(token_example: TokenExample, probability_rows: list[list[float]],
                       relabel_confidence: float, change_counter: Counter) -> TokenExample:
    """Example with confidently contradicted labels replaced.

    Args:
        token_example: Real example.
        probability_rows: Fold-model probabilities per unit.
        relabel_confidence: Probability needed to overrule a label.
        change_counter: Counts ``old->new`` changes (mutated).

    Returns:
        The (possibly) relabelled example.
    """
    new_labels = []
    for unit_label, probability_row in zip(token_example.unit_labels, probability_rows, strict=True):
        best_index = max(range(len(LABEL_NAMES)), key=lambda class_index: probability_row[class_index])
        predicted_name = LABEL_NAMES[best_index]
        if unit_label != IGNORE_LABEL and predicted_name != unit_label and \
                probability_row[best_index] >= relabel_confidence:
            change_counter[f"{unit_label}->{predicted_name}"] += 1
            new_labels.append(predicted_name)
        else:
            new_labels.append(unit_label)
    return dataclasses.replace(token_example, unit_labels=tuple(new_labels))


def fold_model(verifier_settings: VerifierSettings, kept_examples: list[TokenExample],
               random_seed: int) -> tuple[LoadedVerifier, object]:
    """Train the model of one fold (kept in memory, never saved).

    Args:
        verifier_settings: Settings with the fold's epoch budget.
        kept_examples: Real examples of the other book_fold_sets.
        random_seed: Seed.

    Returns:
        ``(final weights, feature assembler)``.
    """
    training_settings = verifier_settings.training_settings
    torch.manual_seed(random_seed)
    torch_device = torch_device_of(training_settings.device_name)
    feature_assembler = build_assembler(evidence_sources(verifier_settings), torch_device,
                                        training_settings.evidence_dropout, random_seed)
    loaded_verifier = create_verifier(verifier_settings.path_settings.base_model, torch_device, feature_assembler,
                                      training_settings.backbone_kind)
    if training_settings.freeze_word_embeddings:
        freeze_word_embeddings(loaded_verifier.backbone_adapter.encoder_module)
    fold_corpus = training_corpus(verifier_settings.path_settings.data_directory, loaded_verifier.backbone_adapter,
                             feature_assembler, training_settings, kept_examples)
    final_weights: dict[str, LoadedVerifier] = {}
    member_run = MemberRun(fold_corpus, training_settings, feature_assembler, window_shape_of(verifier_settings),
                           lambda best_weights, _: final_weights.update(latest=best_weights),
                           random.Random(random_seed), None)  # nosec B311  # noqa: S311 - shuffling, not cryptography
    fit_member(loaded_verifier, member_run)
    return final_weights.get("latest", loaded_verifier), feature_assembler


def fold_settings(verifier_settings: VerifierSettings) -> VerifierSettings:
    """Settings of the fold models: short, single-stage, no early stop.

    Args:
        verifier_settings: All settings.

    Returns:
        Adjusted settings.
    """
    training_settings = dataclasses.replace(
        verifier_settings.training_settings, epoch_count=verifier_settings.relabel_settings.fold_epochs,
        real_stage_epochs=0, early_stop_patience=0, ensemble_members=1)
    return dataclasses.replace(verifier_settings, training_settings=training_settings)


def run_relabelling(verifier_settings: VerifierSettings) -> Path:
    """Relabel the real training examples by cross-predictions.

    Usage:
        ``python -m token_verifier relabel`` after ``export-aligned`` and ``synth``, before ``train``.

    Args:
        verifier_settings: All settings.

    Returns:
        Path of ``aligned_train_relabelled.jsonl``.

    Raises:
        FileNotFoundError: If ``aligned_train.jsonl`` is missing or empty.
    """
    data_directory = verifier_settings.path_settings.data_directory
    aligned_examples = read_examples(data_directory / "aligned_train.jsonl")
    if not aligned_examples:
        raise FileNotFoundError(f"No aligned_train.jsonl in {data_directory}.")
    short_settings = fold_settings(verifier_settings)
    change_counter: Counter = Counter()
    relabelled: dict[int, TokenExample] = {}
    book_fold_sets = book_folds(aligned_examples, verifier_settings.relabel_settings.fold_count)
    for fold_index, held_books in enumerate(book_fold_sets):
        print(f"relabel fold {fold_index + 1}: {len(held_books)} held-out books", flush=True)
        held_positions = [example_position for example_position, token_example in enumerate(aligned_examples)
                          if token_example.book_key in held_books]
        fold_weights, feature_assembler = fold_model(short_settings, [
            token_example for token_example in aligned_examples if token_example.book_key not in held_books],
            short_settings.training_settings.random_seed + fold_index)
        probability_lists = predict_units(VerifierBundle([fold_weights], feature_assembler,
                                                         window_shape_of(short_settings)),
                                          [text_from_example(aligned_examples[example_position], data_directory)
                                           for example_position in held_positions])
        for example_position, probability_rows in zip(held_positions, probability_lists, strict=True):
            relabelled[example_position] = relabelled_example(aligned_examples[example_position], probability_rows,
                                                              verifier_settings.relabel_settings.relabel_confidence,
                                                              change_counter)
        del fold_weights, feature_assembler
        gc.collect()
        torch.cuda.empty_cache()
    output_path = data_directory / RELABELLED_FILE
    write_examples(output_path, [relabelled.get(example_position, token_example)
                                 for example_position, token_example in enumerate(aligned_examples)])
    print(f"relabelled units: {dict(change_counter)}", flush=True)
    return output_path
