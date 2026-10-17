"""Layout review after OCR: statistical rules that correct segment labels once the text is known.

Each rule is a function over the OCR'd page records returning how many segments it relabelled, so new rules
are added to the list built by the command line without touching this module. Default rules:

* page numbering (``segments.page_numbering``): short top/bottom texts that continue a numbering become ``number``;
* sparse headings (``segments.heading_statistics``): short, centred, isolated lines of near-empty pages become
  headings.
"""

from collections.abc import Callable
from dataclasses import dataclass

from pdf2fb2.progress.progress_contracts import StageProgress
from pdf2fb2.segments.heading_statistics import mark_sparse_headings
from pdf2fb2.segments.page_numbering import verify_page_numbers


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
                        LayoutRule("headings", mark_sparse_headings))


class LayoutReview:
    """Applies the layout rules in order.

    Usage:
        Built by the command line; injected into the pipeline and run after OCR, before word styles.
    """

    def __init__(self, layout_rules: tuple[LayoutRule, ...] = DEFAULT_LAYOUT_RULES) -> None:
        """Store the rules.

        Args:
            layout_rules: Rules applied in this order.
        """
        self._layout_rules = layout_rules

    def review_layout(self, page_records: list[dict], stage_progress: StageProgress) -> int:
        """Apply every rule to the OCR'd page records.

        Args:
            page_records: OCR'd page records (mutated by the rules).
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
