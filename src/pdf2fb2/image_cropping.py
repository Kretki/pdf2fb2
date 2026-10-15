"""Cropping of page images for OCR input."""

from PIL import Image

CROP_PADDING = 8
MIN_CROP_SIDE = 64


def crop_segment_for_ocr(page_image: Image.Image, bounding_box: list[float],
                         padding_pixels: int = CROP_PADDING) -> Image.Image:
    """Cut a padded segment out of a page and upscale tiny crops.

    Usage:
        Call for every segment sent to the OCR model.

    Args:
        page_image: Full RGB page image.
        bounding_box: [x1, y1, x2, y2] in page pixels.
        padding_pixels: Margin added on every side, in pixels (>= 0).

    Returns:
        RGB crop; if its shorter side is under 64 px it is upscaled so the VLM can read it.
    """
    image_width, image_height = page_image.size
    left_edge, top_edge, right_edge, bottom_edge = bounding_box
    segment_crop = page_image.crop((max(0, int(left_edge) - padding_pixels), max(0, int(top_edge) - padding_pixels),
                                    min(image_width, int(right_edge) + padding_pixels),
                                    min(image_height, int(bottom_edge) + padding_pixels)))
    shorter_side = min(segment_crop.size)
    if shorter_side >= MIN_CROP_SIDE:
        return segment_crop
    scale_factor = MIN_CROP_SIDE / max(1, shorter_side)
    return segment_crop.resize((max(1, round(segment_crop.width * scale_factor)),
                                max(1, round(segment_crop.height * scale_factor))), Image.LANCZOS)
