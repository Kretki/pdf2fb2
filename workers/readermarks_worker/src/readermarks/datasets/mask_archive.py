"""Compact storage of multi-class binary masks (bit-packed, compressed ``.npz``, no pickle)."""

from pathlib import Path

import numpy

MAX_ARCHIVE_BYTES = 256 * 1024 * 1024


def write_mask_archive(archive_path: Path, class_masks: numpy.ndarray) -> None:
    """Store (C, H, W) masks as packed bits.

    Args:
        archive_path: Destination ``.npz`` file.
        class_masks: (C, H, W) array; non-zero = class present.

    Returns:
        None.
    """
    numpy.savez_compressed(
        archive_path, packed=numpy.packbits(class_masks > 0, axis=-1), shape=numpy.array(class_masks.shape, numpy.int64)
    )


def read_mask_archive(archive_path: Path) -> numpy.ndarray:
    """Load masks written by ``write_mask_archive``.

    Args:
        archive_path: ``.npz`` file.

    Returns:
        (C, H, W) uint8 array with values 0/1.

    Raises:
        ValueError: If the file is too large or malformed.
    """
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError(f"{archive_path} is larger than {MAX_ARCHIVE_BYTES} bytes.")
    with numpy.load(archive_path, allow_pickle=False) as archive_content:
        mask_shape = tuple(int(dimension) for dimension in archive_content["shape"])
        if len(mask_shape) != 3:
            raise ValueError(f"{archive_path} does not hold (C, H, W) masks.")
        return numpy.unpackbits(archive_content["packed"], axis=-1, count=mask_shape[2]).astype(numpy.uint8)
