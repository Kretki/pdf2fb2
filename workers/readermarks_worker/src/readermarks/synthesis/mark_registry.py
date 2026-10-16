"""Registry of all mark planners with their class and sampling weight.

New mark types are added by registering another ``PlannerEntry`` here (open/closed), not by
editing existing planners.
"""

from readermarks.synthesis.handwriting_placement import plan_inline_text, plan_margin_text
from readermarks.synthesis.ink_compositing import HIGHLIGHTER_TOOL
from readermarks.synthesis.mark_placement import (
    PlannerEntry,
    highlight_paths,
    plan_margin_mark,
    plan_span_mark,
    plan_underlines,
)
from readermarks.synthesis.stroke_geometry import ellipse_path, strike_paths

MARK_PLANNERS = (
    PlannerEntry("hand_underline", plan_underlines, 3.0),
    PlannerEntry("hand_encirclement", plan_span_mark(ellipse_path, 3, 1.0), 1.0),
    PlannerEntry("margin_mark", plan_margin_mark, 2.0),
    PlannerEntry("hand_text_inline", plan_inline_text, 1.5),
    PlannerEntry("hand_text_margin", plan_margin_text, 2.5),
    PlannerEntry("highlighter", plan_span_mark(highlight_paths, 6, 1.0), 0.8, HIGHLIGHTER_TOOL),
    PlannerEntry("strike_through", plan_span_mark(strike_paths, 3, 1.0), 1.0),
)
