"""Units the verifier is unsure about, their line crops for the vision-language judge, and how answers combine.

A unit is uncertain when its error probability (misread + inserted) lies inside the judge band. Its printed line
(from the word-box placement) is cropped once per line; the judge's answer and the verifier's probability are
averaged, keeping the verifier's split between misread and inserted.
"""

from pathlib import Path
from typing import Protocol

from PIL import Image

from pdf2fb2.progress.progress_contracts import StageProgress
from pdf2fb2.text_verification.region_mapping import UnitPlacement, padded_box
from pdf2fb2.text_verification.verify_units import VerifyUnit
from worker_protocol.judge_job_schema import MAX_LINE_LENGTH, MAX_TARGET_LENGTH, JudgeItem

JUDGE_DIRECTORY = "judge_lines"
JUDGE_WEIGHT = 0.5
KEY_SEPARATOR = "#"


def is_uncertain(probability_row: list[float], judge_band: tuple[float, float]) -> bool:
    """Tell whether the verifier is unsure about a unit.

    Args:
        probability_row: ``[correct, misread, inserted]``.
        judge_band: ``(lowest, highest)`` error probability that is judged.

    Returns:
        True inside the band.
    """
    return judge_band[0] <= probability_row[1] + probability_row[2] <= judge_band[1]


def line_crop(page_image: Image.Image, unit_placement: UnitPlacement, line_index: int, crop_path: Path) -> None:
    """Write the crop of one printed line.

    Args:
        page_image: Page image.
        unit_placement: Placement with the line's words.
        line_index: Line of the unit.
        crop_path: Output PNG.

    Returns:
        None.
    """
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    page_image.crop(padded_box(unit_placement.word_lines[line_index], page_image.size)).save(crop_path)


def segment_judge_items(segment_key: str, segment_units: list[VerifyUnit], unit_placement: UnitPlacement | None,
                        probability_rows: list[list[float]], judge_inputs: tuple[Image.Image, Path, tuple]) -> \
        list[JudgeItem]:
    """Judge items of the uncertain units of one segment (line crops are written here).

    Usage:
        Called by ``VerificationFeedback`` after the first verification pass when a judge is configured.

    Args:
        segment_key: Segment key.
        segment_units: Units of the segment.
        unit_placement: Placement on the word boxes (units without a line are not judged).
        probability_rows: Verifier probabilities per unit.
        judge_inputs: ``(page image, work directory, judge band)``.

    Returns:
        Items keyed ``<segment key>#<unit index>``.
    """
    page_image, work_directory, judge_band = judge_inputs
    if unit_placement is None:
        return []
    judge_items, written_lines = [], set()
    for unit_index, probability_row in enumerate(probability_rows):
        line_index = unit_placement.unit_lines[unit_index]
        if line_index is None or not is_uncertain(probability_row, judge_band):
            continue
        relative_path = f"{JUDGE_DIRECTORY}/{segment_key.replace(':', '_')}_{line_index:03d}.png"
        if line_index not in written_lines:
            line_crop(page_image, unit_placement, line_index, work_directory / relative_path)
            written_lines.add(line_index)
        line_text = " ".join(segment_units[member_index].unit_text for member_index, member_line
                             in enumerate(unit_placement.unit_lines) if member_line == line_index)
        target_text = segment_units[unit_index].unit_text[:MAX_TARGET_LENGTH]
        judge_items.append(JudgeItem(f"{segment_key}{KEY_SEPARATOR}{unit_index}", relative_path,
                                     line_text[:MAX_LINE_LENGTH], target_text))
    return judge_items


def judged_row(probability_row: list[float], judge_error: float) -> list[float]:
    """Combine the verifier's probabilities with the judge's error probability.

    Args:
        probability_row: ``[correct, misread, inserted]``.
        judge_error: Judge's probability that the unit is not printed as read.

    Returns:
        New row: error mass averaged, misread/inserted split kept.
    """
    verifier_error = probability_row[1] + probability_row[2]
    combined_error = (1 - JUDGE_WEIGHT) * verifier_error + JUDGE_WEIGHT * judge_error
    misread_share = probability_row[1] / verifier_error if verifier_error > 0 else 1.0
    return [1.0 - combined_error, combined_error * misread_share, combined_error * (1 - misread_share)]


class UnitJudge(Protocol):
    """Judges units against their line images (see ``model_stages.judge_stage.JudgeStage``)."""

    def judge_units(self, judge_items: list[JudgeItem], work_directory: Path,
                    stage_progress: StageProgress) -> dict[str, float]:
        """Error probability of every item.

        Args:
            judge_items: Units with line crops.
            work_directory: Output directory.
            stage_progress: Progress sink.

        Returns:
            Item key -> probability.
        """
        ...


def judged_probabilities(cases_by_key: dict, unit_probabilities: dict[str, list[list[float]]],
                         unit_judge: UnitJudge, judge_band: tuple[float, float], work_directory: Path,
                         stage_progress: StageProgress) -> dict[str, list[list[float]]]:
    """Verifier probabilities with the uncertain units re-weighted by the judge.

    Usage:
        Called by ``VerificationFeedback`` between the first verification pass and planning.

    Args:
        cases_by_key: Segment cases with ``page_record``, ``segment_units`` and ``unit_placement``.
        unit_probabilities: First-pass probabilities per segment key.
        unit_judge: Judge stage.
        judge_band: Error-probability band that is judged.
        work_directory: Output directory (line crops are written under it).
        stage_progress: Progress sink.

    Returns:
        Probabilities per segment key (rows of judged units replaced).
    """
    judge_items: list[JudgeItem] = []
    for segment_key, segment_case in cases_by_key.items():
        if segment_case.unit_placement is None or segment_key not in unit_probabilities:
            continue
        with Image.open(segment_case.page_record["image"]) as page_file:
            page_image = page_file.convert("RGB")
        judge_items.extend(segment_judge_items(segment_key, segment_case.segment_units, segment_case.unit_placement,
                                               unit_probabilities[segment_key],
                                               (page_image, work_directory, judge_band)))
    judge_answers = unit_judge.judge_units(judge_items, work_directory, stage_progress)
    refined_rows = {segment_key: [list(probability_row) for probability_row in probability_rows]
               for segment_key, probability_rows in unit_probabilities.items()}
    for item_key, judge_error in judge_answers.items():
        segment_key, unit_text = item_key.rsplit(KEY_SEPARATOR, 1)
        unit_index = int(unit_text)
        refined_rows[segment_key][unit_index] = judged_row(refined_rows[segment_key][unit_index], judge_error)
    return refined_rows
