"""Invariants of deterministic mark removal."""

import numpy

from readermarks.postprocess.mark_removal import removal_mask, remove_marks


def synthetic_page() -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Paper page with a printed glyph block and a pen stroke crossing it.

    Returns:
        ``(rgb_image, mark_mask, ink_mask)``.
    """
    rgb_image = numpy.full((120, 160, 3), 220, numpy.uint8)
    ink_mask = numpy.zeros((120, 160), bool)
    ink_mask[40:60, 50:90] = True
    rgb_image[ink_mask] = 30
    mark_mask = numpy.zeros((120, 160), bool)
    mark_mask[48:51, 20:140] = True
    rgb_image[mark_mask & ~ink_mask] = (20, 40, 150)
    return rgb_image, mark_mask, ink_mask


def test_pixels_outside_dilated_mask_are_untouched() -> None:
    """Nothing outside the dilated mark mask changes."""
    rgb_image, mark_mask, ink_mask = synthetic_page()
    cleaned_image = remove_marks(rgb_image, mark_mask, ink_mask, 2, 16)
    allowed_mask = removal_mask(mark_mask, ink_mask, 2)
    assert numpy.array_equal(cleaned_image[~allowed_mask], rgb_image[~allowed_mask])


def test_printed_ink_is_kept_under_marks() -> None:
    """Printed-ink pixels survive even where the mark overlaps them."""
    rgb_image, mark_mask, ink_mask = synthetic_page()
    cleaned_image = remove_marks(rgb_image, mark_mask, ink_mask, 2, 16)
    assert numpy.array_equal(cleaned_image[ink_mask], rgb_image[ink_mask])


def test_mark_only_pixels_become_paper() -> None:
    """Mark-only pixels are replaced by the local paper colour."""
    rgb_image, mark_mask, ink_mask = synthetic_page()
    cleaned_image = remove_marks(rgb_image, mark_mask, ink_mask, 2, 16)
    restored_pixels = cleaned_image[mark_mask & ~ink_mask]
    assert numpy.abs(restored_pixels.astype(int) - 220).max() <= 1


def test_empty_mark_mask_changes_nothing() -> None:
    """Without marks the output equals the input."""
    rgb_image, _, ink_mask = synthetic_page()
    cleaned_image = remove_marks(rgb_image, numpy.zeros_like(ink_mask), ink_mask, 2, 16)
    assert numpy.array_equal(cleaned_image, rgb_image)
