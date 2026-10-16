"""Blanking of picture regions before detection."""

import numpy

from readermarks.inference.region_masking import REGION_PADDING, blank_regions


def test_regions_are_filled_with_paper_and_rest_untouched() -> None:
    """Inside a box (plus padding) the page becomes paper colour; outside nothing changes."""
    rgb_image = numpy.full((100, 80, 3), 230, numpy.uint8)
    rgb_image[30:60, 20:50] = 15
    rgb_image[90:95, 5:10] = 40
    blanked_image = blank_regions(rgb_image, [[20, 30, 50, 60]])
    assert (blanked_image[30 - REGION_PADDING:60 + REGION_PADDING, 20:50] == 230).all()
    assert numpy.array_equal(blanked_image[90:95, 5:10], rgb_image[90:95, 5:10])
    assert blank_regions(rgb_image, []) is rgb_image
