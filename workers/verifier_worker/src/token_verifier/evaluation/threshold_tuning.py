"""Choose the decision threshold of each error class on the validation split (highest F1)."""

from token_verifier.evaluation.label_metrics import ERROR_CLASSES, class_scores

# Fine low end: a rare class such as ``inserted`` can peak below 0.05.
LOW_THRESHOLDS = (0.005, 0.01, 0.015, 0.02, 0.03, 0.04)
THRESHOLD_GRID = LOW_THRESHOLDS + tuple(round(grid_step / 100, 2) for grid_step in range(5, 96, 5))


def best_threshold(gold_labels: list[str], probability_rows: list[list[float]], class_name: str) -> float:
    """Threshold with the highest F1 for one class (ties: the higher threshold, i.e. fewer flags).

    Args:
        gold_labels: True labels of the validation units.
        probability_rows: Predicted probabilities.
        class_name: ``misread`` or ``inserted``.

    Returns:
        Threshold from ``THRESHOLD_GRID``.
    """
    return max(THRESHOLD_GRID, key=lambda grid_threshold: (
        class_scores(gold_labels, probability_rows, class_name, grid_threshold).score_summary()["f1"], grid_threshold))


def tune_thresholds(gold_labels: list[str], probability_rows: list[list[float]]) -> dict[str, float]:
    """Thresholds of both error classes.

    Usage:
        Called by the ``tune`` command on real validation output (``aligned_valid.jsonl``).

    Args:
        gold_labels: True labels.
        probability_rows: Predicted probabilities.

    Returns:
        ``{"misread": float, "inserted": float}``.
    """
    return {class_name: best_threshold(gold_labels, probability_rows, class_name) for class_name in ERROR_CLASSES}
