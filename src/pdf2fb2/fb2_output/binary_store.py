"""In-memory store of images embedded into the FB2 as ``<binary>`` elements."""

import io

from PIL import Image

MAX_IMAGE_SIDE = 2000
JPEG_QUALITY = 88
IMAGE_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png"}


class BinaryStore:
    """Encodes pictures and keeps them by FB2 binary id.

    Usage:
        ``add_image`` while building the flow; ``stored_items`` when writing the FB2.
    """

    def __init__(self) -> None:
        """Create an empty store."""
        self._stored_items: dict[str, tuple[str, bytes]] = {}

    @property
    def stored_items(self) -> dict[str, tuple[str, bytes]]:
        """All stored images.

        Returns:
            Mapping ``binary_id -> (content_type, encoded_bytes)`` in insertion order.
        """
        return self._stored_items

    def add_image(self, binary_id: str, picture_image: Image.Image, image_format: str = "JPEG") -> str:
        """Encode a picture (downscaled to 2000 px max) and store it.

        Args:
            binary_id: FB2 id, e.g. ``p0001_1.jpg``.
            picture_image: Image to embed.
            image_format: ``JPEG`` (quality 88) or ``PNG`` (lossless).

        Returns:
            ``binary_id``, for use in ``<image l:href="#...">``.

        Raises:
            KeyError: If ``image_format`` is not supported.
        """
        content_type = IMAGE_FORMATS[image_format]
        rgb_picture = picture_image.convert("RGB")
        if max(rgb_picture.size) > MAX_IMAGE_SIDE:
            rgb_picture.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.LANCZOS)
        image_buffer = io.BytesIO()
        save_options = {"optimize": True} if image_format == "PNG" else {"quality": JPEG_QUALITY, "optimize": True}
        rgb_picture.save(image_buffer, image_format, **save_options)
        self._stored_items[binary_id] = (content_type, image_buffer.getvalue())
        return binary_id
