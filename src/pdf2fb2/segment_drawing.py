"""Draws segmentation boxes over page images for visual inspection."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pdf2fb2.label_roles import DEFAULT_LABEL_COLOR, LABEL_COLORS

LABEL_FONT_FILE = "DejaVuSans.ttf"
LABEL_FONT_SIZE = 18
FILL_ALPHA = 40
TAG_ALPHA = 230
OUTLINE_WIDTH = 3


def _load_label_font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load DejaVu Sans, falling back to PIL's default font.

    Returns:
        Font used for box tags.
    """
    try:
        return ImageFont.truetype(LABEL_FONT_FILE, LABEL_FONT_SIZE)
    except OSError:
        return ImageFont.load_default()


def _draw_single_segment(drawing_context: ImageDraw.ImageDraw, page_segment: dict,
                         label_font: ImageFont.ImageFont) -> None:
    """Draw one translucent box with an ``order: label`` tag above it.

    Args:
        drawing_context: Draw object of the RGBA overlay.
        page_segment: Segment with ``bbox``, ``label`` and ``order``.
        label_font: Font for the tag.
    """
    left_edge, top_edge, right_edge, bottom_edge = page_segment["bbox"]
    label_color = LABEL_COLORS.get(page_segment["label"], DEFAULT_LABEL_COLOR)
    drawing_context.rectangle([left_edge, top_edge, right_edge, bottom_edge], fill=(*label_color, FILL_ALPHA),
                              outline=(*label_color, 255), width=OUTLINE_WIDTH)
    tag_text = f"{page_segment['order']}: {page_segment['label']}"
    tag_width, tag_height = drawing_context.textbbox((0, 0), tag_text, font=label_font)[2:]
    tag_top_edge = max(0, top_edge - tag_height - 4)
    drawing_context.rectangle([left_edge, tag_top_edge, left_edge + tag_width + 6, tag_top_edge + tag_height + 4],
                              fill=(*label_color, TAG_ALPHA))
    drawing_context.text((left_edge + 3, tag_top_edge + 2), tag_text, fill=(255, 255, 255, 255), font=label_font)


def draw_segment_overlay(image_path: Path, page_segments: list[dict], output_path: Path) -> None:
    """Save a copy of a page image with its segments drawn on top.

    Args:
        image_path: Rendered page image.
        page_segments: Ordered segments of the page.
        output_path: Destination PNG.

    Returns:
        None. Writes ``output_path``.
    """
    with Image.open(image_path) as page_file:
        page_image = page_file.convert("RGBA")
    overlay_image = Image.new("RGBA", page_image.size, (0, 0, 0, 0))
    drawing_context = ImageDraw.Draw(overlay_image)
    label_font = _load_label_font()
    for page_segment in page_segments:
        _draw_single_segment(drawing_context, page_segment, label_font)
    Image.alpha_composite(page_image, overlay_image).convert("RGB").save(output_path)
