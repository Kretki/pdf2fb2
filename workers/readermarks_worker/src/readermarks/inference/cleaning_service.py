"""High-level cleaning actions used by the CLI and the pdf2fb2 worker: one page or a folder of pages."""

from collections.abc import Callable, Sequence
from pathlib import Path

from readermarks.datasets.image_files import read_rgb_image
from readermarks.inference.cleaning_outputs import WrittenPage, write_page_outputs
from readermarks.inference.page_cleaner import PageCleaner
from readermarks.inference.region_masking import RegionBox, blank_regions

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff"})
PageCallback = Callable[[int, int], None]


def clean_page_file(
    page_cleaner: PageCleaner,
    input_path: Path,
    output_directory: Path,
    write_debug: bool,
    masked_regions: Sequence[RegionBox] = (),
) -> WrittenPage:
    """Clean one page image into ``output_directory``; masked regions are blanked first.

    Usage:
        ``python -m readermarks clean --input page.png --out out_dir/``.

    Args:
        page_cleaner: Loaded cleaner.
        input_path: Page image.
        output_directory: Directory for ``clean.png``, masks, ``marks.json`` (and ``debug.png``).
        write_debug: Also write the debug overlay.
        masked_regions: Pixel boxes (printed pictures) blanked with paper colour before detection; the
            cleaned page keeps them blank.

    Returns:
        What was written.
    """
    rgb_image = blank_regions(read_rgb_image(input_path), masked_regions)
    page_verdict = page_cleaner.inspect_page(rgb_image)
    return write_page_outputs(
        input_path,
        rgb_image,
        page_verdict,
        output_directory,
        write_debug,
        page_cleaner.cleaning_settings.patch_threshold,
    )


def list_input_images(input_directory: Path) -> list[Path]:
    """Image files of a folder, in name order (no recursion, no symlinks).

    Args:
        input_directory: Folder of page images.

    Returns:
        Sorted image paths.
    """
    return sorted(
        candidate_path
        for candidate_path in input_directory.iterdir()
        if candidate_path.suffix.lower() in IMAGE_SUFFIXES
        and candidate_path.is_file()
        and not candidate_path.is_symlink()
    )


def clean_page_batch(
    page_cleaner: PageCleaner,
    input_paths: list[Path],
    output_directory: Path,
    write_debug: bool,
    page_callback: PageCallback | None = None,
    page_regions: Sequence[Sequence[RegionBox]] | None = None,
) -> list[WrittenPage]:
    """Clean many pages; each page gets its own sub-directory named after the image stem.

    Usage:
        ``python -m readermarks clean --input-dir pages/ --out out_dir/ --batch`` and the pdf2fb2 worker.

    Args:
        page_cleaner: Loaded cleaner.
        input_paths: Page images.
        output_directory: Root output directory.
        write_debug: Also write debug overlays.
        page_callback: Called with ``(page_number, page_count)`` before each page (progress).
        page_regions: Masked regions per page (same order as ``input_paths``), or None.

    Returns:
        Written pages in input order.
    """
    regions_by_page = page_regions if page_regions is not None else [()] * len(input_paths)
    written_pages = []
    for page_position, (input_path, masked_regions) in enumerate(zip(input_paths, regions_by_page, strict=True), 1):
        if page_callback is not None:
            page_callback(page_position, len(input_paths))
        written_pages.append(
            clean_page_file(page_cleaner, input_path, output_directory / input_path.stem, write_debug, masked_regions)
        )
    return written_pages
