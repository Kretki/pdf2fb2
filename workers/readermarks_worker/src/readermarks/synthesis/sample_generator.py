"""Generate one synthetic (image, per-class pixel masks) sample from a clean base page."""

from dataclasses import dataclass

import numpy

from readermarks.settings.class_catalog import MARK_CLASSES, SPOTTER_CLASSES, spotter_channel
from readermarks.settings.model_settings import SynthesisSettings
from readermarks.synthesis.ink_compositing import choose_tool, composite_mark, textured_alpha
from readermarks.synthesis.mark_placement import PlannerEntry
from readermarks.synthesis.mark_registry import MARK_PLANNERS
from readermarks.synthesis.page_analysis import PageAnalysis
from readermarks.synthesis.printed_pictures import add_printed_picture
from readermarks.synthesis.printed_rules import add_printed_rules
from readermarks.synthesis.printed_text_copies import add_printed_text_copies
from readermarks.synthesis.scanner_effects import apply_geometric, apply_photometric, finish_scan

UNDER_PRINT_PROBABILITY = 0.3
SINGLE_TOOL_PROBABILITY = 0.7
GEOMETRIC_PROBABILITY = 0.6


@dataclass(frozen=True)
class SampleRecipe:
    """Random decisions of one sample, drawn up front so they can be logged in the manifest.

    Attributes:
        marked_page: Whether reader marks are added.
        rule_page: Whether synthetic printed rules are added.
        mark_count: Number of marks to draw.
    """

    marked_page: bool
    rule_page: bool
    mark_count: int


@dataclass(frozen=True)
class SyntheticSample:
    """One generated training sample.

    Attributes:
        page_image: (H, W, 3) uint8 RGB page as the scanner would deliver it.
        class_masks: (len(SPOTTER_CLASSES), H, W) uint8 masks, 255 = class present.
        recipe: Random decisions that produced the sample.
    """

    page_image: numpy.ndarray
    class_masks: numpy.ndarray
    sample_recipe: SampleRecipe

    @property
    def has_marks(self) -> bool:
        """Return whether any reader-mark pixel survived (after cropping/warping).

        Returns:
            True when at least one mark-class mask is non-empty.
        """
        return bool(self.class_masks[: len(MARK_CLASSES)].any())


def draw_recipe(
    base_has_rules: bool, synthesis_settings: SynthesisSettings, random_source: numpy.random.Generator
) -> SampleRecipe:
    """Draw the page-level decisions (marked or not, printed rules or not, number of marks).

    Args:
        base_has_rules: Whether the base page already contains detected printed rules.
        synthesis_settings: Mixing rules.
        random_source: Random generator.

    Returns:
        Sample recipe.
    """
    marked_page = random_source.random() >= synthesis_settings.unmarked_fraction
    rule_page = base_has_rules or random_source.random() < synthesis_settings.rule_page_fraction
    mark_count = int(random_source.integers(synthesis_settings.minimum_marks, synthesis_settings.maximum_marks + 1))
    return SampleRecipe(marked_page, rule_page, mark_count if marked_page else 0)


def choose_planners(mark_count: int, random_source: numpy.random.Generator) -> list[PlannerEntry]:
    """Sample mark planners according to their weights.

    Args:
        mark_count: Number of marks.
        random_source: Random generator.

    Returns:
        ``mark_count`` planner entries (with repetition).
    """
    sampling_weights = numpy.array([planner_entry.sampling_weight for planner_entry in MARK_PLANNERS])
    chosen_indices = random_source.choice(len(MARK_PLANNERS), mark_count, p=sampling_weights / sampling_weights.sum())
    return [MARK_PLANNERS[chosen_index] for chosen_index in chosen_indices]


def add_reader_marks(
    page_image: numpy.ndarray,
    class_masks: numpy.ndarray,
    analysis: PageAnalysis,
    mark_count: int,
    random_source: numpy.random.Generator,
) -> None:
    """Render and composite ``mark_count`` marks of random classes.

    Args:
        page_image: (H, W, 3) float32 page 0..1, mutated.
        class_masks: (C, H, W) uint8 masks in ``SPOTTER_CLASSES`` order, mutated.
        analysis: Page analysis of the base page.
        mark_count: Number of marks.
        random_source: Random generator.

    Returns:
        None.
    """
    page_tool = choose_tool(random_source)
    page_shape = page_image.shape[:2]
    for planner_entry in choose_planners(mark_count, random_source):
        drawing_tool = page_tool
        if planner_entry.tool_kind is not None or random_source.random() > SINGLE_TOOL_PROBABILITY:
            drawing_tool = choose_tool(random_source, planner_entry.tool_kind)
        under_print = random_source.random() < UNDER_PRINT_PROBABILITY
        class_mask = class_masks[spotter_channel(planner_entry.class_name)]
        for local_alpha in planner_entry.plan_function(analysis, page_shape, drawing_tool, random_source):
            composite_mark(
                page_image,
                class_mask,
                textured_alpha(local_alpha, drawing_tool, random_source),
                drawing_tool,
                analysis.ink_mask,
                under_print,
            )


def generate_sample(
    base_image: numpy.ndarray,
    analysis: PageAnalysis,
    verso_image: numpy.ndarray | None,
    synthesis_settings: SynthesisSettings,
    random_source: numpy.random.Generator,
) -> SyntheticSample:
    """Build one synthetic sample: printed rules, reader marks, then scanner realism.

    Usage:
        Called by the dataset writer once per sample with a per-sample seeded generator.

    Args:
        base_image: Clean page (H, W, 3) uint8 RGB.
        analysis: ``analyse_page(base_image)``.
        verso_image: Another clean page (float32 0..1) used for bleed-through, or None.
        synthesis_settings: Mixing rules.
        random_source: Random generator (seeded per sample for reproducibility).

    Returns:
        The synthetic sample.
    """
    base_has_rules = bool(analysis.rule_mask.any())
    sample_recipe = draw_recipe(base_has_rules, synthesis_settings, random_source)
    page_image = base_image.astype(numpy.float32) / 255.0
    class_masks = numpy.zeros((len(SPOTTER_CLASSES),) + base_image.shape[:2], numpy.uint8)
    class_masks[spotter_channel("printed_ink")] = analysis.ink_mask
    class_masks[spotter_channel("printed_rule")] = analysis.rule_mask
    if sample_recipe.rule_page and (
        not base_has_rules or random_source.random() < synthesis_settings.synthetic_rule_probability
    ):
        add_printed_rules(
            page_image,
            class_masks[spotter_channel("printed_ink")],
            class_masks[spotter_channel("printed_rule")],
            analysis,
            random_source,
        )
    if random_source.random() < synthesis_settings.picture_page_fraction:
        left_edge, top_edge, right_edge, bottom_edge = add_printed_picture(
            page_image, class_masks[spotter_channel("printed_ink")], analysis, random_source
        )
        class_masks[spotter_channel("printed_rule"), top_edge:bottom_edge, left_edge:right_edge] = 0
    if random_source.random() < synthesis_settings.text_copy_page_fraction:
        add_printed_text_copies(page_image, class_masks[spotter_channel("printed_ink")], analysis, random_source)
    if sample_recipe.marked_page:
        add_reader_marks(page_image, class_masks, analysis, sample_recipe.mark_count, random_source)
    page_image = apply_photometric(page_image, verso_image, random_source)
    if random_source.random() < GEOMETRIC_PROBABILITY:
        page_image, class_masks = apply_geometric(page_image, class_masks, random_source)
    return SyntheticSample(finish_scan(page_image, random_source), class_masks, sample_recipe)
