"""Class catalogue shared by the generator, the models and the evaluation (STAGE_CLEAN.md §5).

All outputs are multi-label (independent sigmoids): one channel per class, a pixel or
patch may belong to several classes, background is all zeros.
"""

MARK_CLASSES = (
    "hand_underline",
    "hand_encirclement",
    "margin_mark",
    "hand_text_inline",
    "hand_text_margin",
    "highlighter",
    "strike_through",
)
PROTECTED_CLASSES = ("printed_ink", "printed_rule")
SPOTTER_CLASSES = MARK_CLASSES + PROTECTED_CLASSES
# The segmenter additionally predicts the union of all mark classes as an explicit channel.
MARK_ANY_CLASS = "mark_any"
SEGMENTER_CLASSES = (MARK_ANY_CLASS,) + SPOTTER_CLASSES

MARK_CLASS_COUNT = len(MARK_CLASSES)
SPOTTER_CLASS_COUNT = len(SPOTTER_CLASSES)
SEGMENTER_CLASS_COUNT = len(SEGMENTER_CLASSES)


def spotter_channel(class_name: str) -> int:
    """Return the channel index of a class in the spotter output.

    Usage:
        Use instead of hard-coded indices when reading spotter grids or labels.

    Args:
        class_name: One of ``SPOTTER_CLASSES``.

    Returns:
        Channel index, 0..SPOTTER_CLASS_COUNT-1.

    Raises:
        ValueError: If the class is unknown.
    """
    return SPOTTER_CLASSES.index(class_name)


def segmenter_channel(class_name: str) -> int:
    """Return the channel index of a class in the segmenter output.

    Usage:
        Use instead of hard-coded indices when reading segmenter masks.

    Args:
        class_name: One of ``SEGMENTER_CLASSES``.

    Returns:
        Channel index, 0..SEGMENTER_CLASS_COUNT-1.

    Raises:
        ValueError: If the class is unknown.
    """
    return SEGMENTER_CLASSES.index(class_name)
