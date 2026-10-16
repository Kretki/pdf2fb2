"""Binary detection metrics (precision, recall, F-score, IoU) from boolean arrays or counts."""

from dataclasses import asdict, dataclass

import numpy


@dataclass
class DetectionCounts:
    """Accumulated confusion counts.

    Attributes:
        true_positives: Predicted and expected.
        false_positives: Predicted, not expected.
        false_negatives: Expected, not predicted.
        true_negatives: Neither.
    """

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0

    def add_arrays(self, predicted: numpy.ndarray, expected: numpy.ndarray) -> None:
        """Accumulate counts from two boolean arrays of the same shape.

        Args:
            predicted: Predicted positives.
            expected: Ground-truth positives.

        Returns:
            None.
        """
        self.true_positives += int(numpy.count_nonzero(predicted & expected))
        self.false_positives += int(numpy.count_nonzero(predicted & ~expected))
        self.false_negatives += int(numpy.count_nonzero(~predicted & expected))
        self.true_negatives += int(numpy.count_nonzero(~predicted & ~expected))

    def summarize_counts(self) -> dict:
        """Precision, recall, F-score, IoU and accuracy plus the raw counts.

        Returns:
            JSON-ready dict; ratios are None when undefined.
        """

        def class_ratio(numerator: int, denominator: int) -> float | None:
            """Safe division rounded to 4 digits; None when the denominator is 0."""
            return round(numerator / denominator, 4) if denominator else None

        precision = class_ratio(self.true_positives, self.true_positives + self.false_positives)
        recall_value = class_ratio(self.true_positives, self.true_positives + self.false_negatives)
        f_measure = (
            round(2 * precision * recall_value / (precision + recall_value), 4)
            if precision is not None and recall_value is not None and precision + recall_value > 0
            else None
        )
        total_count = self.true_positives + self.false_positives + self.false_negatives + self.true_negatives
        return {
            "precision": precision,
            "recall": recall_value,
            "f_score": f_measure,
            "iou": class_ratio(self.true_positives, self.true_positives + self.false_positives + self.false_negatives),
            "accuracy": class_ratio(self.true_positives + self.true_negatives, total_count),
            **asdict(self),
        }
