"""Full per-page pipeline: spot -> segment flagged regions -> fuse + geometric veto -> remove marks."""

import time
from dataclasses import dataclass, field

import numpy

from readermarks.inference.segmenter_inference import SegmenterRunner
from readermarks.inference.spotter_inference import SpotterOutput, SpotterRunner, flagged_regions
from readermarks.postprocess.mark_removal import remove_marks
from readermarks.postprocess.mask_fusion import FusedMarks, fuse_masks
from readermarks.settings.model_settings import CleaningSettings
from readermarks.synthesis.page_analysis import PixelBox


@dataclass
class PageVerdict:
    """Everything the cleaner decided about one page.

    Attributes:
        spotter_output: Stage 1 output.
        region_boxes: Flagged regions (empty = Stage 2 skipped).
        fused_marks: Stage 2 fusion result, or None when skipped.
        cleaned_image: Page without marks, or None when the page has no marks.
        stage_seconds: Wall time per stage (``spotter``, ``segmenter``, ``removal``).
    """

    spotter_output: SpotterOutput
    region_boxes: list[PixelBox]
    fused_marks: FusedMarks | None
    cleaned_image: numpy.ndarray | None
    stage_seconds: dict[str, float] = field(default_factory=dict)

    @property
    def has_marks(self) -> bool:
        """Return whether reader marks were found and removed.

        Returns:
            True when ``cleaned_image`` was produced.
        """
        return self.cleaned_image is not None


class PageCleaner:
    """Decides whether a page has reader marks and removes them.

    Usage:
        ``PageCleaner(spotter_runner, segmenter_runner, cleaning_settings).inspect_page(rgb_image)``.
    """

    def __init__(
        self, spotter_runner: SpotterRunner, segmenter_runner: SegmenterRunner, cleaning_settings: CleaningSettings
    ) -> None:
        """Store the runners and thresholds.

        Args:
            spotter_runner: Stage 1 runner.
            segmenter_runner: Stage 2 runner.
            cleaning_settings: Effective thresholds (defaults + tuned values).
        """
        self._spotter_runner = spotter_runner
        self._segmenter_runner = segmenter_runner
        self.cleaning_settings = cleaning_settings

    def inspect_page(self, rgb_image: numpy.ndarray) -> PageVerdict:
        """Run the pipeline on one page.

        Args:
            rgb_image: (H, W, 3) uint8 RGB page.

        Returns:
            Page verdict; ``cleaned_image`` is None when the page has no marks.
        """
        stage_start = time.perf_counter()
        spotter_output = self._spotter_runner.spot_page(rgb_image)
        region_boxes = flagged_regions(spotter_output, self.cleaning_settings)
        page_verdict = PageVerdict(
            spotter_output, region_boxes, None, None, {"spotter": time.perf_counter() - stage_start}
        )
        if region_boxes:
            self._segment_and_remove(rgb_image, page_verdict)
        return page_verdict

    def _segment_and_remove(self, rgb_image: numpy.ndarray, page_verdict: PageVerdict) -> None:
        """Stage 2 and removal for a page with flagged regions.

        Args:
            rgb_image: (H, W, 3) uint8 page.
            page_verdict: Verdict to complete (mutated).

        Returns:
            None.
        """
        stage_start = time.perf_counter()
        probability_maps = self._segmenter_runner.segment_regions(rgb_image, page_verdict.region_boxes)
        page_verdict.fused_marks = fuse_masks(probability_maps, self.cleaning_settings)
        page_verdict.stage_seconds["segmenter"] = time.perf_counter() - stage_start
        if int(page_verdict.fused_marks.mark_mask.sum()) < self.cleaning_settings.minimum_mark_pixels:
            return
        stage_start = time.perf_counter()
        page_verdict.cleaned_image = remove_marks(
            rgb_image,
            page_verdict.fused_marks.mark_mask,
            page_verdict.fused_marks.ink_mask,
            self.cleaning_settings.mark_dilation_pixels,
            self.cleaning_settings.background_tile_pixels,
        )
        page_verdict.stage_seconds["removal"] = time.perf_counter() - stage_start
