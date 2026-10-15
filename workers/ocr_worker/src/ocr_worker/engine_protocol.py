"""Abstraction every OCR backend implements."""

from typing import Protocol

from PIL import Image


class OcrEngine(Protocol):
    """Reads the text of one image crop.

    Usage:
        Implemented by ``TransformersOcrEngine`` and ``HttpOcrEngine``; the job
        runner depends only on this protocol.

    Attributes:
        supports_parallel_requests: True when ``recognize_image`` may be called
            from several threads at once.
    """

    supports_parallel_requests: bool

    def recognize_image(self, crop_image: Image.Image, max_new_tokens: int) -> str:
        """Read the text in an image crop.

        Args:
            crop_image: RGB crop of one layout segment.
            max_new_tokens: Generation budget, 1..32768.

        Returns:
            Raw model output (Markdown / HTML tables).
        """
        ...
