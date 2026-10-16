"""Word crops in the TexTAR input format: box + 10% bottom padding (for underlines), 128x96, values 0..1."""

import math
from collections.abc import Sequence

import numpy
import torch
from PIL import Image

CROP_HEIGHT = 128
CROP_WIDTH = 96
BOTTOM_PADDING_RATIO = 0.1

PixelWordBox = Sequence[int]


def padded_box(word_box: PixelWordBox, page_height: int) -> tuple[int, int, int, int]:
    """Extend a word box downwards so underlines below the baseline are included.

    Args:
        word_box: ``(x0, y0, x1, y1)`` pixel box.
        page_height: Page height in pixels (the box is clipped to it).

    Returns:
        ``(x0, y0, x1, y1_padded)``.
    """
    box_height = abs(word_box[3] - word_box[1])
    return (word_box[0], word_box[1], word_box[2],
            min(page_height, word_box[3] + math.ceil(BOTTOM_PADDING_RATIO * box_height)))


def crop_tensor(page_image: Image.Image, word_box: PixelWordBox) -> torch.Tensor:
    """Cut, resize (bilinear, aspect not kept, as in training) and convert one word.

    Args:
        page_image: RGB page.
        word_box: ``(x0, y0, x1, y1)`` pixel box.

    Returns:
        ``(3, 128, 96)`` float32 tensor in 0..1.
    """
    word_image = page_image.crop(padded_box(word_box, page_image.height)).resize((CROP_WIDTH, CROP_HEIGHT),
                                                                                  Image.Resampling.BILINEAR)
    return torch.from_numpy(numpy.asarray(word_image, dtype=numpy.float32) / 255.0).permute(2, 0, 1)


def page_word_crops(page_image: Image.Image, word_boxes: Sequence[PixelWordBox]) -> torch.Tensor:
    """Crops of all words of a page.

    Args:
        page_image: RGB page.
        word_boxes: Word boxes.

    Returns:
        ``(N, 3, 128, 96)`` float32 tensor (``(0, 3, 128, 96)`` for no words).
    """
    if not word_boxes:
        return torch.zeros((0, 3, CROP_HEIGHT, CROP_WIDTH))
    return torch.stack([crop_tensor(page_image, word_box) for word_box in word_boxes])
