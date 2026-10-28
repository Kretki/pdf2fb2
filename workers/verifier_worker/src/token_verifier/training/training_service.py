"""``train`` command: fine-tune the base encoder on synthetic and real examples and keep the best epoch.

Training data: ``synthetic_train.jsonl`` (all catalog languages, balanced at export) plus
``aligned_train.jsonl`` (real pipeline output of all languages, repeated ``real_repeat`` times and oversampled
per language towards the language with most examples, by at most ``language_balance_cap``).
Selection data: ``aligned_valid.jsonl`` when it exists, else ``synthetic_valid.jsonl``.
After every epoch the thresholds are tuned on the selection data and the epoch with the best mean F1 of the two
error classes is saved to ``output_model`` (weights, tokenizer, ``thresholds.json``, ``window_settings.json``).
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
from transformers import get_linear_schedule_with_warmup

from token_verifier.datasets.example_files import TokenExample, read_examples
from token_verifier.datasets.token_encoding import training_windows
from token_verifier.evaluation.label_metrics import ERROR_CLASSES, class_scores
from token_verifier.evaluation.threshold_tuning import tune_thresholds
from token_verifier.inference.unit_prediction import predict_units
from token_verifier.networks.model_loading import (
    THRESHOLDS_FILE,
    WINDOW_SETTINGS_FILE,
    LoadedVerifier,
    load_verifier,
)
from token_verifier.training.training_loop import OptimizerState, train_epoch
from token_verifier.verifier_settings import TrainingSettings, VerifierSettings


def language_balanced(token_examples: list[TokenExample], balance_cap: int) -> list[TokenExample]:
    """Oversample the languages with few examples towards the largest one.

    Args:
        token_examples: Examples of all languages.
        balance_cap: Largest repetition factor of one language.

    Returns:
        Examples with every language repeated ``min(cap, ceil(largest / own count))`` times.
    """
    examples_by_language: dict[str, list[TokenExample]] = defaultdict(list)
    for token_example in token_examples:
        examples_by_language[token_example.language_name].append(token_example)
    largest_count = max((len(language_examples) for language_examples in examples_by_language.values()), default=0)
    return [token_example for language_examples in examples_by_language.values()
            for token_example in language_examples * min(balance_cap,
                                                          math.ceil(largest_count / len(language_examples)))]


def training_examples(data_directory: Path, training_settings: TrainingSettings) -> list[TokenExample]:
    """Synthetic and repeated, language-balanced real training examples.

    Args:
        data_directory: Directory of the JSONL files.
        training_settings: ``real_repeat`` and ``language_balance_cap``.

    Returns:
        Examples.

    Raises:
        FileNotFoundError: If there is no training data at all.
    """
    real_examples = language_balanced(read_examples(data_directory / "aligned_train.jsonl"),
                                      training_settings.language_balance_cap)
    collected_examples = read_examples(data_directory / "synthetic_train.jsonl") + \
        real_examples * training_settings.real_repeat
    if not collected_examples:
        raise FileNotFoundError(f"No synthetic_train.jsonl / aligned_train.jsonl in {data_directory}.")
    return collected_examples


def selection_examples(data_directory: Path) -> list[TokenExample]:
    """Examples used to pick the best epoch and its thresholds.

    Args:
        data_directory: Directory of the JSONL files.

    Returns:
        Real validation output when available, else synthetic validation examples.
    """
    return read_examples(data_directory / "aligned_valid.jsonl") or \
        read_examples(data_directory / "synthetic_valid.jsonl")


def optimizer_for(loaded_verifier: LoadedVerifier, training_settings: TrainingSettings,
                  update_count: int) -> OptimizerState:
    """AdamW, warm-up schedule and weighted loss.

    Args:
        loaded_verifier: Model to train.
        training_settings: Hyper-parameters.
        update_count: Optimizer updates over all epochs.

    Returns:
        Optimizer state.
    """
    weight_optimizer = torch.optim.AdamW(loaded_verifier.encoder_model.parameters(),
                                         lr=training_settings.learning_rate,
                                         weight_decay=training_settings.weight_decay)
    rate_schedule = get_linear_schedule_with_warmup(weight_optimizer,
                                                    int(update_count * training_settings.warmup_ratio), update_count)
    class_weights = torch.tensor(training_settings.class_weights, device=loaded_verifier.torch_device)
    return OptimizerState(weight_optimizer, rate_schedule, torch.nn.CrossEntropyLoss(weight=class_weights))


def selection_score(loaded_verifier: LoadedVerifier, validation_examples: list[TokenExample],
                    training_settings: TrainingSettings) -> tuple[float, dict[str, float]]:
    """Mean F1 of the error classes at tuned thresholds.

    Args:
        loaded_verifier: Model after an epoch.
        validation_examples: Selection examples.
        training_settings: Window size.

    Returns:
        ``(mean F1, thresholds)``.
    """
    probability_lists = predict_units(loaded_verifier, [list(token_example.unit_texts)
                                                        for token_example in validation_examples],
                                      training_settings.window_units, training_settings.max_tokens)
    gold_labels = [unit_label for token_example in validation_examples for unit_label in token_example.unit_labels]
    probability_rows = [probability_row for probability_list in probability_lists
                        for probability_row in probability_list]
    class_thresholds = tune_thresholds(gold_labels, probability_rows)
    f1_values = [class_scores(gold_labels, probability_rows, class_name,
                              class_thresholds[class_name]).score_summary()["f1"]
                 for class_name in ERROR_CLASSES]
    return sum(f1_values) / len(f1_values), class_thresholds


def save_verifier(loaded_verifier: LoadedVerifier, output_directory: Path, class_thresholds: dict[str, float],
                  training_settings: TrainingSettings) -> None:
    """Write the fine-tuned model with its thresholds and window size.

    Args:
        loaded_verifier: Model to save.
        output_directory: ``models/token_verifier``.
        class_thresholds: Tuned thresholds.
        training_settings: Window size.

    Returns:
        None.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    loaded_verifier.encoder_model.save_pretrained(str(output_directory))
    loaded_verifier.text_tokenizer.save_pretrained(str(output_directory))
    (output_directory / THRESHOLDS_FILE).write_text(json.dumps(class_thresholds, indent=1), encoding="utf-8")
    (output_directory / WINDOW_SETTINGS_FILE).write_text(json.dumps(
        {"window_units": training_settings.window_units, "max_tokens": training_settings.max_tokens}), encoding="utf-8")


def run_training(verifier_settings: VerifierSettings) -> Path:
    """Fine-tune and save the best epoch.

    Usage:
        ``python -m token_verifier train`` (GPU recommended; about 1-2 GB of VRAM for the base model).

    Args:
        verifier_settings: All settings.

    Returns:
        The output model directory.
    """
    path_settings, training_settings = verifier_settings.path_settings, verifier_settings.training_settings
    torch.manual_seed(training_settings.random_seed)
    loaded_verifier = load_verifier(path_settings.base_model, training_settings.device_name, new_head=True)
    encoded_windows = [encoded_window for token_example in training_examples(path_settings.data_directory,
                                                                             training_settings)
                       for encoded_window in training_windows(loaded_verifier.text_tokenizer, token_example,
                                                              training_settings.window_units,
                                                              training_settings.max_tokens)]
    validation_examples = selection_examples(path_settings.data_directory)
    batches_per_epoch = -(-len(encoded_windows) // training_settings.batch_size)
    optimizer_state = optimizer_for(loaded_verifier, training_settings, training_settings.epoch_count *
                                    -(-batches_per_epoch // training_settings.gradient_accumulation))
    shuffle_generator = random.Random(training_settings.random_seed)  # nosec B311  # noqa: S311 - shuffling, not cryptography
    best_score = -1.0
    for epoch_number in range(1, training_settings.epoch_count + 1):
        epoch_loss = train_epoch(loaded_verifier, encoded_windows, optimizer_state, training_settings,
                                 shuffle_generator,
                                 lambda done_count, total_count, running_loss: None)
        epoch_score, class_thresholds = selection_score(loaded_verifier, validation_examples, training_settings)
        print(f"epoch {epoch_number}: loss {epoch_loss:.4f}, validation mean F1 {epoch_score:.4f}, "
              f"thresholds {class_thresholds}", flush=True)
        if epoch_score > best_score:
            best_score = epoch_score
            save_verifier(loaded_verifier, path_settings.output_model, class_thresholds, training_settings)
    return path_settings.output_model
