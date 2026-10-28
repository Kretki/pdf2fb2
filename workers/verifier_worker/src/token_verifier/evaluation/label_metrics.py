"""Unit labels from probabilities and thresholds, and precision / recall / F1 of the error classes."""

from dataclasses import dataclass

ERROR_CLASSES = ("misread", "inserted")
CLASS_POSITIONS = {"correct": 0, "misread": 1, "inserted": 2}


@dataclass(frozen=True)
class ClassScores:
    """Detection scores of one class.

    Attributes:
        true_positives: Units of the class that were flagged.
        false_positives: Units flagged wrongly.
        false_negatives: Units of the class that were missed.
    """

    true_positives: int
    false_positives: int
    false_negatives: int

    def score_summary(self) -> dict:
        """Precision, recall, F1 and support.

        Returns:
            Mapping with the rates rounded to four digits.
        """
        precision = self.true_positives / max(self.true_positives + self.false_positives, 1)
        recall_rate = self.true_positives / max(self.true_positives + self.false_negatives, 1)
        f1_score = 2 * precision * recall_rate / max(precision + recall_rate, 1e-9)
        return {"precision": round(precision, 4), "recall": round(recall_rate, 4), "f1": round(f1_score, 4),
                "support": self.true_positives + self.false_negatives}


def predicted_label(unit_probabilities: list[float], class_thresholds: dict[str, float]) -> str:
    """Decision for one unit.

    Args:
        unit_probabilities: ``[correct, misread, inserted]``.
        class_thresholds: Thresholds of ``misread`` and ``inserted``.

    Returns:
        ``inserted`` when its threshold is reached, else ``misread`` when its threshold is reached, else ``correct``.
    """
    if unit_probabilities[CLASS_POSITIONS["inserted"]] >= class_thresholds["inserted"]:
        return "inserted"
    if unit_probabilities[CLASS_POSITIONS["misread"]] >= class_thresholds["misread"]:
        return "misread"
    return "correct"


def class_scores(gold_labels: list[str], probability_rows: list[list[float]], class_name: str,
                 class_threshold: float) -> ClassScores:
    """One-vs-rest scores of a class at a threshold (``ignore`` units are skipped).

    Args:
        gold_labels: True labels.
        probability_rows: Probabilities per unit.
        class_name: ``misread`` or ``inserted``.
        class_threshold: Decision threshold.

    Returns:
        Counts.
    """
    class_position = CLASS_POSITIONS[class_name]
    flagged_pairs = [(gold_label == class_name, probability_row[class_position] >= class_threshold)
                     for gold_label, probability_row in zip(gold_labels, probability_rows, strict=True)
                     if gold_label != "ignore"]
    return ClassScores(sum(gold_match and flag_match for gold_match, flag_match in flagged_pairs),
                       sum(flag_match and not gold_match for gold_match, flag_match in flagged_pairs),
                       sum(gold_match and not flag_match for gold_match, flag_match in flagged_pairs))
