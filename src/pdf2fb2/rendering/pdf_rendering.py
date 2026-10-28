"""Rendering PDF pages and regions to images with PyMuPDF."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PIL import Image

from pdf2fb2.parallel_work.parallel_mapping import map_in_processes

POINTS_PER_INCH = 72.0
MIN_REGION_POINTS = 2
COVER_HEIGHT_PIXELS = 1200


@dataclass(frozen=True)
class PageRenderTask:
    """One page to render in a worker process.

    Attributes:
        pdf_path: Input PDF.
        page_index: 0-based page index.
        image_path: PNG file to write.
        render_dpi: Resolution in dots per inch, 72..600.
    """

    pdf_path: Path
    page_index: int
    image_path: Path
    render_dpi: int


def render_page_task(render_task: PageRenderTask) -> Path:
    """Render one page to PNG (runs in a worker process, so it opens the PDF itself).

    Args:
        render_task: Page, target file and resolution.

    Returns:
        Path of the written PNG.
    """
    with pymupdf.open(render_task.pdf_path) as pdf_document:
        pdf_document[render_task.page_index].get_pixmap(dpi=render_task.render_dpi,
                                                        alpha=False).save(render_task.image_path)
    return render_task.image_path


def count_pdf_pages(pdf_path: Path) -> int:
    """Number of pages of a PDF.

    Args:
        pdf_path: Input PDF.

    Returns:
        Page count (>= 0).
    """
    with pymupdf.open(pdf_path) as pdf_document:
        return pdf_document.page_count


def render_pdf_pages(pdf_path: Path, output_directory: Path, render_dpi: int,
                     page_callback: Callable[[int, int], None] | None = None) -> list[Path]:
    """Render every page of a PDF to a PNG file, pages in parallel worker processes.

    Usage:
        First pipeline step; the images feed segmentation and OCR.

    Args:
        pdf_path: Input PDF.
        output_directory: Directory for ``page_NNNN.png`` (created if missing).
        render_dpi: Resolution in dots per inch, 72..600.
        page_callback: Optional ``callback(rendered_count, page_count)`` called after each rendered page
            (in completion order).

    Returns:
        Paths of the rendered pages, in page order.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    render_tasks = [PageRenderTask(pdf_path, page_index, output_directory / f"page_{page_index + 1:04d}.png",
                                   render_dpi) for page_index in range(count_pdf_pages(pdf_path))]
    return map_in_processes(render_page_task, render_tasks, page_callback)


def _pixmap_to_image(page_pixmap: pymupdf.Pixmap) -> Image.Image:
    """Convert an RGB PyMuPDF pixmap to a PIL image.

    Args:
        page_pixmap: Pixmap rendered with ``alpha=False``.

    Returns:
        RGB PIL image.
    """
    return Image.frombytes("RGB", (page_pixmap.width, page_pixmap.height), page_pixmap.samples)


def crop_pdf_region(pdf_document: pymupdf.Document, page_record: dict, bounding_box: list[float],
                    segmentation_dpi: int, image_dpi: int) -> Image.Image | None:
    """Re-render a segment straight from the PDF (sharper than cropping the page image).

    Usage:
        Call for pictures, charts, formulas and unparsed tables embedded in the FB2.

    Args:
        pdf_document: Open PDF.
        page_record: Page record (``page`` is 1-based, ``image`` is the rendered page).
        bounding_box: [x1, y1, x2, y2] in pixels of the page rendered at ``segmentation_dpi``.
        segmentation_dpi: DPI the page image was rendered at.
        image_dpi: DPI for the re-rendered region.

    Returns:
        RGB image of the region, or None if the region is empty or too small.
    """
    pdf_page = pdf_document[page_record["page"] - 1]
    if pdf_page.rotation:  # rotated pages use different coordinates: crop the rendered page instead
        with Image.open(page_record["image"]) as page_image:
            return page_image.convert("RGB").crop(tuple(int(coordinate) for coordinate in bounding_box))
    scale_factor = POINTS_PER_INCH / segmentation_dpi
    region_rect = pymupdf.Rect(*(coordinate * scale_factor for coordinate in bounding_box)) & pdf_page.rect
    if region_rect.is_empty or region_rect.width < MIN_REGION_POINTS or region_rect.height < MIN_REGION_POINTS:
        return None
    return _pixmap_to_image(pdf_page.get_pixmap(dpi=image_dpi, clip=region_rect, alpha=False))


def render_cover_image(pdf_document: pymupdf.Document, target_height: int = COVER_HEIGHT_PIXELS) -> Image.Image:
    """Render the first page as the book cover.

    Args:
        pdf_document: Open PDF with at least one page.
        target_height: Cover height in pixels.

    Returns:
        RGB image of the first page.
    """
    first_page = pdf_document[0]
    zoom_factor = target_height / first_page.rect.height
    return _pixmap_to_image(first_page.get_pixmap(matrix=pymupdf.Matrix(zoom_factor, zoom_factor), alpha=False))
