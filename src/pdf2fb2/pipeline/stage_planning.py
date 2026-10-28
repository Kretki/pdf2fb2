"""Which progress stages a conversion runs: the optional stages and their order."""

from dataclasses import dataclass

from pdf2fb2.pipeline.stage_contracts import (
    FlowChecker,
    HandwritingDetector,
    LayoutReviewer,
    MarkedPageRescanner,
    PictureRecoverer,
    RescanProgress,
    ResidualTextFinder,
    TextStyleDetector,
    TextVerifier,
)
from pdf2fb2.progress.pipeline_progress import PipelineProgress
from pdf2fb2.progress.stage_names import (
    ASSEMBLING_STAGE,
    CLEANING_STAGE,
    DRAWING_STAGE,
    HANDWRITING_STAGE,
    PICTURE_STAGE,
    PROOFREADING_STAGE,
    RECOGNITION_STAGE,
    RENDERING_STAGE,
    RESCAN_SEGMENTATION_STAGE,
    RESIDUAL_STAGE,
    REVIEW_STAGE,
    SEGMENT_REVIEW_STAGE,
    SEGMENTATION_STAGE,
    STYLING_STAGE,
    VERIFICATION_STAGE,
)


@dataclass(frozen=True)
class OptionalStages:
    """Stages a conversion can run or skip; None skips the stage.

    Attributes:
        handwriting_detector: Marks handwritten segments before mark cleaning.
        mark_rescanner: Cleans pages with reader marks and re-segments them before OCR.
        picture_recoverer: Replaces lettering that does not fit the book's font size by pictures before OCR.
        residual_finder: Adds segments for text left outside every segment.
        segment_reviewer: Corrects labels that decide what is OCR'd (misplaced running heads), last step before OCR.
        layout_reviewer: Corrects labels (page numbers, headings) once the text is known, before word styles.
        style_detector: Detects word bold/italic/strikeout in text segments.
        text_verifier: Labels OCR units and re-reads flagged lines (after word boxes, before word styles).
        flow_checker: Proofreads and rearranges the book flow with a language model.
    """

    handwriting_detector: HandwritingDetector | None = None
    mark_rescanner: MarkedPageRescanner | None = None
    picture_recoverer: PictureRecoverer | None = None
    residual_finder: ResidualTextFinder | None = None
    segment_reviewer: LayoutReviewer | None = None
    layout_reviewer: LayoutReviewer | None = None
    style_detector: TextStyleDetector | None = None
    text_verifier: TextVerifier | None = None
    flow_checker: FlowChecker | None = None


def active_stage_names(draw_segments: bool, optional_stages: OptionalStages) -> list[str]:
    """List the progress stages of a conversion in execution order.

    Args:
        draw_segments: Whether segmentation drawings are produced.
        optional_stages: Optional stages of the conversion (None entries are skipped).

    Returns:
        Stage names for ``PipelineProgress``.
    """
    stage_switches = (
        (optional_stages.handwriting_detector is not None, [HANDWRITING_STAGE]),
        (optional_stages.mark_rescanner is not None, [CLEANING_STAGE, RESCAN_SEGMENTATION_STAGE]),
        (optional_stages.picture_recoverer is not None, [PICTURE_STAGE]),
        (optional_stages.residual_finder is not None, [RESIDUAL_STAGE]),
        (optional_stages.segment_reviewer is not None, [SEGMENT_REVIEW_STAGE]),
        (True, [RECOGNITION_STAGE]),
        (optional_stages.layout_reviewer is not None, [REVIEW_STAGE]),
        (optional_stages.style_detector is not None, [STYLING_STAGE]),
        (optional_stages.text_verifier is not None, [VERIFICATION_STAGE]),
        (draw_segments, [DRAWING_STAGE]),
        (optional_stages.flow_checker is not None, [PROOFREADING_STAGE]),
    )
    return [RENDERING_STAGE, SEGMENTATION_STAGE,
            *(stage_name for stage_enabled, stage_names in stage_switches if stage_enabled
              for stage_name in stage_names), ASSEMBLING_STAGE]


def rescan_trackers(pipeline_progress: PipelineProgress) -> RescanProgress:
    """Progress sinks of the reader-mark rescan.

    Args:
        pipeline_progress: Overall progress.

    Returns:
        Trackers of the cleaning and re-segmentation stages.
    """
    return RescanProgress(pipeline_progress.stage_tracker(CLEANING_STAGE),
                          pipeline_progress.stage_tracker(RESCAN_SEGMENTATION_STAGE))
