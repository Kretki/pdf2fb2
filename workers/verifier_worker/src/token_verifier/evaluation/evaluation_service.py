"""``tune`` and ``eval`` commands: thresholds on real validation output, detection scores on a split.

Real pipeline output (``aligned_<split>.jsonl``) is used when it exists; otherwise the synthetic examples of the
split (reported as such). The final number is ``eval --split test`` on real output of the test books, whose
``old_russian`` part is hand-proofread ground truth.
"""

import json
from collections import defaultdict
from pathlib import Path

from token_verifier.datasets.example_files import TokenExample, read_examples
from token_verifier.evaluation.label_metrics import ERROR_CLASSES, class_scores
from token_verifier.evaluation.threshold_tuning import tune_thresholds
from token_verifier.inference.unit_prediction import predict_units
from token_verifier.networks.model_loading import (
    THRESHOLDS_FILE,
    LoadedVerifier,
    load_verifier,
    read_thresholds,
    read_window_settings,
)
from token_verifier.verifier_settings import VerifierSettings


def split_examples(data_directory: Path, split_name: str) -> tuple[list[TokenExample], str]:
    """Examples of a split and where they come from.

    Args:
        data_directory: Directory of the JSONL files.
        split_name: ``valid`` or ``test``.

    Returns:
        ``(examples, "aligned" | "synthetic")``.

    Raises:
        FileNotFoundError: If the split has no examples.
    """
    aligned_examples = read_examples(data_directory / f"aligned_{split_name}.jsonl")
    if aligned_examples:
        return aligned_examples, "aligned"
    synthetic_examples = read_examples(data_directory / f"synthetic_{split_name}.jsonl")
    if not synthetic_examples:
        raise FileNotFoundError(f"No aligned_{split_name}.jsonl or synthetic_{split_name}.jsonl in {data_directory}.")
    return synthetic_examples, "synthetic"


def predicted_rows(loaded_verifier: LoadedVerifier, token_examples: list[TokenExample],
                   model_directory: Path, verifier_settings: VerifierSettings) -> list[list[list[float]]]:
    """Probabilities of every unit of every example.

    Args:
        loaded_verifier: Fine-tuned model.
        token_examples: Examples.
        model_directory: Model directory (window size).
        verifier_settings: Default window size.

    Returns:
        Per example, per unit: probabilities.
    """
    window_units, max_tokens = read_window_settings(model_directory, verifier_settings.training_settings.window_units,
                                                    verifier_settings.training_settings.max_tokens)
    return predict_units(loaded_verifier, [list(token_example.unit_texts) for token_example in token_examples],
                         window_units, max_tokens)


def run_tuning(verifier_settings: VerifierSettings) -> dict[str, float]:
    """Tune and store the thresholds of the fine-tuned model.

    Usage:
        ``python -m token_verifier tune`` after ``train`` (``train`` already tunes; rerun after new valid data).

    Args:
        verifier_settings: All settings.

    Returns:
        The thresholds written to ``thresholds.json``.
    """
    model_directory = verifier_settings.path_settings.output_model
    loaded_verifier = load_verifier(model_directory, verifier_settings.training_settings.device_name)
    token_examples, _ = split_examples(verifier_settings.path_settings.data_directory, "valid")
    probability_lists = predicted_rows(loaded_verifier, token_examples, model_directory, verifier_settings)
    class_thresholds = tune_thresholds(
        [unit_label for token_example in token_examples for unit_label in token_example.unit_labels],
        [probability_row for probability_list in probability_lists for probability_row in probability_list])
    (model_directory / THRESHOLDS_FILE).write_text(json.dumps(class_thresholds, indent=1), encoding="utf-8")
    return class_thresholds


def grouped_scores(token_examples: list[TokenExample], probability_lists: list[list[list[float]]],
                   class_thresholds: dict[str, float]) -> dict[str, dict]:
    """Scores per language and over all examples.

    Args:
        token_examples: Examples.
        probability_lists: Their probabilities.
        class_thresholds: Decision thresholds.

    Returns:
        ``{group: {class: summary}}`` with group ``ALL`` plus one per language.
    """
    grouped_rows: dict[str, tuple[list[str], list[list[float]]]] = defaultdict(lambda: ([], []))
    for token_example, probability_list in zip(token_examples, probability_lists, strict=True):
        for group_name in ("ALL", token_example.language_name):
            grouped_rows[group_name][0].extend(token_example.unit_labels)
            grouped_rows[group_name][1].extend(probability_list)
    return {group_name: {class_name: class_scores(gold_labels, probability_rows, class_name,
                                                  class_thresholds[class_name]).score_summary()
                         for class_name in ERROR_CLASSES}
            for group_name, (gold_labels, probability_rows) in sorted(grouped_rows.items())}


def markdown_scores(split_name: str, data_origin: str, score_groups: dict[str, dict]) -> str:
    """Markdown table of the scores.

    Args:
        split_name: Evaluated split.
        data_origin: ``aligned`` or ``synthetic``.
        score_groups: Result of ``grouped_scores``.

    Returns:
        Markdown text.
    """
    report_lines = [f"# Token verifier, {split_name} split ({data_origin} data)", "",
                    "| group | class | precision | recall | F1 | support |", "|---|---|---|---|---|---|"]
    for group_name, class_summaries in score_groups.items():
        report_lines.extend(f"| {group_name} | {class_name} | {class_summary['precision']:.3f} | "
                            f"{class_summary['recall']:.3f} | {class_summary['f1']:.3f} | {class_summary['support']} |"
                            for class_name, class_summary in class_summaries.items())
    return "\n".join(report_lines) + "\n"


def run_evaluation(verifier_settings: VerifierSettings, split_name: str) -> Path:
    """Evaluate the fine-tuned model on a split and write ``verifier_<split>.{json,md}``.

    Usage:
        ``python -m token_verifier eval --split test`` (the final score; never tune on it).

    Args:
        verifier_settings: All settings.
        split_name: ``valid`` or ``test``.

    Returns:
        Path of the Markdown report.
    """
    model_directory = verifier_settings.path_settings.output_model
    loaded_verifier = load_verifier(model_directory, verifier_settings.training_settings.device_name)
    token_examples, data_origin = split_examples(verifier_settings.path_settings.data_directory, split_name)
    score_groups = grouped_scores(token_examples,
                                  predicted_rows(loaded_verifier, token_examples, model_directory, verifier_settings),
                                  read_thresholds(model_directory))
    reports_directory = verifier_settings.path_settings.reports_directory
    reports_directory.mkdir(parents=True, exist_ok=True)
    (reports_directory / f"verifier_{split_name}.json").write_text(json.dumps(
        {"split": split_name, "data": data_origin, "scores": score_groups}, indent=1), encoding="utf-8")
    report_path = reports_directory / f"verifier_{split_name}.md"
    report_path.write_text(markdown_scores(split_name, data_origin, score_groups), encoding="utf-8")
    return report_path
