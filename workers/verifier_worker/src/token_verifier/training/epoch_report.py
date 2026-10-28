"""Console report of one training epoch: loss, thresholds, per-class scores and per-language F1."""

from token_verifier.evaluation.label_metrics import ERROR_CLASSES


def class_line(group_name: str, class_summaries: dict[str, dict]) -> str:
    """Precision, recall, F1 and support of both error classes of one group.

    Args:
        group_name: ``ALL`` or a language.
        class_summaries: ``{class: summary}`` of ``grouped_scores``.

    Returns:
        One line of text.
    """
    class_parts = [f"{class_name} P {class_summaries[class_name]['precision']:.3f} "
                   f"R {class_summaries[class_name]['recall']:.3f} F1 {class_summaries[class_name]['f1']:.3f} "
                   f"(n={class_summaries[class_name]['support']})" for class_name in ERROR_CLASSES]
    return f"  {group_name:12s} " + " | ".join(class_parts)


def epoch_report(epoch_number: int, epoch_loss: float, epoch_score: float, class_thresholds: dict[str, float],
                 score_groups: dict[str, dict]) -> str:
    """Multi-line report printed after every epoch.

    Usage:
        Called by ``training_service`` after the validation pass of an epoch.

    Args:
        epoch_number: 1-based epoch.
        epoch_loss: Mean training loss of the epoch.
        epoch_score: Mean F1 of the error classes (the selection score).
        class_thresholds: Tuned thresholds.
        score_groups: ``grouped_scores`` result (group ``ALL`` plus one per language).

    Returns:
        Report text (no trailing newline).
    """
    rounded_thresholds = {class_name: round(threshold_value, 2)
                          for class_name, threshold_value in class_thresholds.items()}
    header_line = (f"epoch {epoch_number}: loss {epoch_loss:.4f}, validation mean F1 {epoch_score:.4f}, "
                   f"thresholds {rounded_thresholds}")
    return "\n".join([header_line, *(class_line(group_name, class_summaries)
                                     for group_name, class_summaries in score_groups.items())])
