"""Bounded image file I/O (8-bit RGB / single-channel masks), never binarising the input."""

from pathlib import Path

import cv2
import numpy

MAX_IMAGE_PIXELS = 12000 * 12000
MAX_IMAGE_BYTES = 512 * 1024 * 1024


def read_rgb_image(image_path: Path) -> numpy.ndarray:
    """Read an image file as 8-bit RGB.

    Usage:
        Call for every page or crop that enters the pipeline.

    Args:
        image_path: PNG/JPEG/TIFF file.

    Returns:
        Array (H, W, 3) uint8 in RGB order; grayscale inputs are replicated to 3 channels.

    Raises:
        ValueError: If the file is too large, unreadable or not an image.
    """
    if image_path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError(f"{image_path} is larger than {MAX_IMAGE_BYTES} bytes.")
    encoded_bytes = numpy.frombuffer(image_path.read_bytes(), dtype=numpy.uint8)
    decoded_image = cv2.imdecode(encoded_bytes, cv2.IMREAD_COLOR)
    if decoded_image is None:
        raise ValueError(f"{image_path} is not a readable image.")
    if decoded_image.shape[0] * decoded_image.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError(f"{image_path} has more than {MAX_IMAGE_PIXELS} pixels.")
    return cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)


def write_rgb_image(image_path: Path, rgb_image: numpy.ndarray) -> None:
    """Write an RGB uint8 array as PNG (lossless).

    Args:
        image_path: Destination ``.png`` path; the parent directory is created.
        rgb_image: Array (H, W, 3) uint8 in RGB order.

    Returns:
        None.

    Raises:
        OSError: If encoding or writing fails.
    """
    image_path.parent.mkdir(parents=True, exist_ok=True)
    encode_success, encoded_bytes = cv2.imencode(".png", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    if not encode_success:
        raise OSError(f"Could not encode {image_path}.")
    image_path.write_bytes(encoded_bytes.tobytes())


def write_mask_image(image_path: Path, mask_image: numpy.ndarray) -> None:
    """Write a single-channel (H, W) or multi-channel (H, W, C<=4) uint8 mask as PNG.

    Args:
        image_path: Destination ``.png`` path; the parent directory is created.
        mask_image: uint8 mask; 0 = background, 255 = class present.

    Returns:
        None.

    Raises:
        OSError: If encoding or writing fails.
    """
    image_path.parent.mkdir(parents=True, exist_ok=True)
    encode_success, encoded_bytes = cv2.imencode(".png", mask_image)
    if not encode_success:
        raise OSError(f"Could not encode {image_path}.")
    image_path.write_bytes(encoded_bytes.tobytes())


def read_mask_image(image_path: Path) -> numpy.ndarray:
    """Read a single-channel uint8 mask written by ``write_mask_image``.

    Args:
        image_path: PNG mask file.

    Returns:
        Array (H, W) uint8.

    Raises:
        ValueError: If the file is not a readable image.
    """
    encoded_bytes = numpy.frombuffer(image_path.read_bytes(), dtype=numpy.uint8)
    decoded_mask = cv2.imdecode(encoded_bytes, cv2.IMREAD_GRAYSCALE)
    if decoded_mask is None:
        raise ValueError(f"{image_path} is not a readable mask.")
    return decoded_mask
