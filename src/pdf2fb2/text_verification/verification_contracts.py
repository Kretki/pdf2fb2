"""Contracts of the verification loop: what labels units, and which optional evidence the loop collects."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pdf2fb2.model_stages.verify_stage import VerifyOutcome
from pdf2fb2.progress.progress_contracts import StageProgress
from worker_protocol.verify_job_schema import VerifyItem


class ItemLabeler(Protocol):
    """Labels verify items carrying context and page evidence (see ``VerifyStage.label_items``)."""

    def label_items(self, verify_items: list[VerifyItem], work_directory: Path, job_name: str,
                    stage_progress: StageProgress) -> VerifyOutcome:
        """Probabilities of every unit.

        Args:
            verify_items: Items.
            work_directory: Output directory.
            job_name: Job file stem.
            stage_progress: Progress sink.

        Returns:
            Probabilities and thresholds.
        """
        ...


@dataclass(frozen=True)
class FeedbackOptions:
    """Optional evidence of the verification loop.

    Attributes:
        write_crops: Write word crops for the image-and-text features.
        second_read: Read every segment a second time from a larger crop.
        judge_band: Error-probability band of units sent to the judge (used only with a judge).
    """

    write_crops: bool = False
    second_read: bool = False
    judge_band: tuple[float, float] = (0.15, 0.6)


class UnitLabeler(Protocol):
    """Labels text units (see ``model_stages.verify_stage.VerifyStage``)."""

    def label_units(self, unit_lists: dict[str, list[str]], work_directory: Path, job_name: str,
                    stage_progress: StageProgress) -> VerifyOutcome:
        """Probabilities of every unit.

        Args:
            unit_lists: Item key -> units.
            work_directory: Output directory.
            job_name: Job file stem.
            stage_progress: Progress sink.

        Returns:
            Probabilities and thresholds.
        """
        ...
