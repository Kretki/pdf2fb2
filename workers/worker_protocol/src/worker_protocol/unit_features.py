"""Per-unit evidence the orchestrator passes to the token verifier, with its normalisation.

Every feature is language-independent and comes from the page, not from the text:

* ``ocr_min_logprob`` / ``ocr_mean_logprob``: lowest and mean log-probability of the OCR model's tokens that
  produced the unit (more negative = less certain);
* ``box_missing``: 1 when no detected word box lies under the unit (text that may not be on the page);
* ``box_shared_units``: how many units share the unit's word box (merged or split words);
* ``box_width_ratio``: width of that box divided by the width its units' characters need on this page (a
  hallucinated word mapped onto a short printed word gets a small ratio);
* ``reread_disagrees``: 1 when a second OCR read of the segment gives a different unit at this position.

A feature may be absent for a unit (``None``): the verifier receives a value and a presence flag per feature, so
the same model works with or without the evidence (synthetic text has none of it).
"""

import math

UNIT_FEATURE_NAMES = ("ocr_min_logprob", "ocr_mean_logprob", "box_missing", "box_shared_units", "box_width_ratio",
                      "reread_disagrees")
LOGPROB_FLOOR = -20.0
SHARED_UNITS_CEILING = 8.0
WIDTH_RATIO_CEILING = 2.0
MAX_FEATURE_MAGNITUDE = 1_000.0


def normalized_feature(feature_name: str, feature_value: float) -> float:
    """Scale a raw feature into 0..1 (or -1..0 for log-probabilities).

    Usage:
        Called by the verifier before features enter the network.

    Args:
        feature_name: One of ``UNIT_FEATURE_NAMES``.
        feature_value: Raw value.

    Returns:
        Normalised value.

    Raises:
        KeyError: For an unknown feature name.
    """
    if feature_name not in UNIT_FEATURE_NAMES:
        raise KeyError(feature_name)
    if feature_name.endswith("logprob"):
        return max(LOGPROB_FLOOR, min(0.0, feature_value)) / -LOGPROB_FLOOR
    if feature_name == "box_shared_units":
        return min(max(feature_value, 0.0), SHARED_UNITS_CEILING) / SHARED_UNITS_CEILING
    if feature_name == "box_width_ratio":
        return min(max(feature_value, 0.0), WIDTH_RATIO_CEILING) / WIDTH_RATIO_CEILING
    return 1.0 if feature_value >= 0.5 else 0.0


def validate_feature_column(feature_name: str, column_values: object, unit_count: int) -> tuple[float | None, ...]:
    """Check one feature column of a verify item or training example.

    Args:
        feature_name: Feature name.
        column_values: Decoded JSON value (list with one number or null per unit).
        unit_count: Units of the item.

    Returns:
        The column as a tuple.

    Raises:
        ValueError: If the name is unknown, the length differs from the units, or a value is not a finite number.
    """
    if feature_name not in UNIT_FEATURE_NAMES:
        raise ValueError(f"Unknown unit feature {feature_name!r}.")
    if not isinstance(column_values, list) or len(column_values) != unit_count:
        raise ValueError(f"Feature {feature_name!r} needs one value per unit.")
    for feature_value in column_values:
        if feature_value is None:
            continue
        if isinstance(feature_value, bool) or not isinstance(feature_value, int | float) \
                or not math.isfinite(feature_value) or abs(feature_value) > MAX_FEATURE_MAGNITUDE:
            raise ValueError(f"Feature {feature_name!r} holds a value that is not a bounded number.")
    return tuple(column_values)


def validate_feature_columns(feature_payload: object, unit_count: int) -> dict[str, tuple[float | None, ...]]:
    """Check all feature columns of an item.

    Args:
        feature_payload: Decoded JSON object ``{feature name: [value | null, ...]}`` or None.
        unit_count: Units of the item.

    Returns:
        Validated columns (empty when absent).

    Raises:
        ValueError: If the payload is not an object or a column is invalid.
    """
    if feature_payload is None:
        return {}
    if not isinstance(feature_payload, dict):
        raise ValueError("Unit features must be an object of columns.")
    return {feature_name: validate_feature_column(feature_name, column_values, unit_count)
            for feature_name, column_values in feature_payload.items()}
