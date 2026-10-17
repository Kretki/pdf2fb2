"""Layout review: statistical rules that correct segment labels, before OCR and once the text is known.

Each rule is a function over the page records returning how many segments it relabelled, so new rules are added
to a rule list without touching the review. Rules before OCR (``SEGMENT_REVIEW_RULES``):

* running heads (``segments.running_heads``): ``header`` segments far from the running-head band become headings
  (running heads are not OCR'd, so this must happen before OCR).

Rules after OCR (``DEFAULT_LAYOUT_RULES``):

* page numbering (``segments.page_numbering``): short top/bottom texts that continue a numbering become ``number``;
* sparse headings (``segments.heading_statistics``): short, centred, isolated lines of near-empty pages become
  headings;
* opening headings (``segments.opening_headings``): a short centred line set apart above the first paragraph of
  a page becomes a heading.
"""

from collections.abc import Callable
from dataclasses import dataclass

from pdf2fb2.progress.progress_contracts import StageProgress
from pdf2fb2.segments.heading_statistics import mark_sparse_headings
from pdf2fb2.segments.opening_headings import mark_opening_headings
from pdf2fb2.segments.page_numbering import verify_page_numbers
from pdf2fb2.segments.running_heads import relabel_misplaced_running_heads


@dataclass(frozen=True)
class LayoutRule:
    """A named label correction.

    Attributes:
        rule_name: Short description used in the progress line, e.g. ``page numbers``.
        rule_function: Takes the OCR'd page records (mutating them) and returns the number of relabelled segments.
    """

    rule_name: str
    rule_function: Callable[[list[dict]], int]


DEFAULT_LAYOUT_RULES = (LayoutRule("page numbers", verify_page_numbers),
                        LayoutRule("headings", mark_sparse_headings),
                        LayoutRule("opening headings", mark_opening_headings))
SEGMENT_REVIEW_RULES = (LayoutRule("running heads", relabel_misplaced_running_heads),)


class LayoutReview:
    """Applies the layout rules in order.

    Usage:
        Built by the command line twice: with ``SEGMENT_REVIEW_RULES`` as the last step before OCR, and with
        ``DEFAULT_LAYOUT_RULES`` after OCR, before word styles.
    """

    def __init__(self, layout_rules: tuple[LayoutRule, ...] = DEFAULT_LAYOUT_RULES) -> None:
        """Store the rules.

        Args:
            layout_rules: Rules applied in this order.
        """
        self._layout_rules = layout_rules

    def review_layout(self, page_records: list[dict], stage_progress: StageProgress) -> int:
        """Apply every rule to the page records.

        Args:
            page_records: Page records (mutated by the rules).
            stage_progress: Receives one progress step per rule and the summary.

        Returns:
            Total number of relabelled segments.
        """
        rule_reports = []
        relabelled_total = 0
        for rule_index, layout_rule in enumerate(self._layout_rules):
            stage_progress.advance_stage(rule_index, len(self._layout_rules), f"checking {layout_rule.rule_name}")
            relabelled_count = layout_rule.rule_function(page_records)
            relabelled_total += relabelled_count
            rule_reports.append(f"{layout_rule.rule_name}: {relabelled_count}")
        stage_progress.advance_stage(1, 1, "relabelled " + ", ".join(rule_reports))
        return relabelled_total
