"""Drawing tools (pencil, pen, highlighter), their textures and multiply-blending onto the page."""

from dataclasses import dataclass

import cv2
import numpy

from readermarks.synthesis.stroke_rendering import LocalAlpha

PENCIL_TOOL = "pencil"
PEN_TOOL = "pen"
HIGHLIGHTER_TOOL = "highlighter"
PEN_COLORS = ((25, 30, 40), (30, 45, 135), (20, 60, 170), (165, 30, 35), (35, 35, 35))
HIGHLIGHTER_COLORS = ((255, 238, 70), (160, 240, 120), (250, 150, 205), (255, 190, 90))
LABEL_ALPHA_THRESHOLD = 0.22
HIGHLIGHTER_LABEL_THRESHOLD = 0.12


@dataclass(frozen=True)
class DrawingTool:
    """How a reader drew a mark.

    Attributes:
        tool_kind: ``pencil``, ``pen`` or ``highlighter``.
        tool_color: RGB color of the pigment, 0..255.
        stroke_width: Typical width in page pixels.
        peak_opacity: Opacity of a full-pressure stroke, 0..1.
    """

    tool_kind: str
    tool_color: tuple[int, int, int]
    stroke_width: float
    peak_opacity: float


def choose_tool(random_source: numpy.random.Generator, tool_kind: str | None = None) -> DrawingTool:
    """Draw a random pencil or pen (highlighters are chosen explicitly).

    Args:
        random_source: Random generator.
        tool_kind: Force a kind; None picks pencil (45%) or pen (55%).

    Returns:
        Drawing tool.
    """
    chosen_kind = tool_kind or (PENCIL_TOOL if random_source.random() < 0.45 else PEN_TOOL)
    if chosen_kind == HIGHLIGHTER_TOOL:
        highlighter_color = HIGHLIGHTER_COLORS[random_source.integers(len(HIGHLIGHTER_COLORS))]
        return DrawingTool(
            HIGHLIGHTER_TOOL, highlighter_color, random_source.uniform(10, 18), random_source.uniform(0.35, 0.7)
        )
    if chosen_kind == PENCIL_TOOL:
        graphite_level = int(random_source.uniform(90, 170))
        return DrawingTool(
            PENCIL_TOOL, (graphite_level,) * 3, random_source.uniform(0.9, 2.6), random_source.uniform(0.6, 0.95)
        )
    pen_color = PEN_COLORS[random_source.integers(len(PEN_COLORS))]
    return DrawingTool(PEN_TOOL, pen_color, random_source.uniform(1.0, 3.5), random_source.uniform(0.8, 1.0))


def graphite_texture(alpha_map: numpy.ndarray, random_source: numpy.random.Generator) -> numpy.ndarray:
    """Grainy, streaky pencil coverage with occasional sheen (lighter specular pixels).

    Args:
        alpha_map: (h, w) float32 coverage.
        random_source: Random generator.

    Returns:
        Textured coverage, same shape.
    """
    grain_noise = cv2.GaussianBlur(random_source.normal(0, 1, alpha_map.shape).astype(numpy.float32), (0, 0), 0.7)
    streak_noise = cv2.GaussianBlur(
        random_source.normal(0, 1, alpha_map.shape).astype(numpy.float32), (0, 0), sigmaX=3.0, sigmaY=0.6
    )
    grain_texture = numpy.clip(
        0.72 + 0.22 * grain_noise / (grain_noise.std() + 1e-6) + 0.12 * streak_noise / (streak_noise.std() + 1e-6),
        0.2,
        1.0,
    )
    sheen_pixels = random_source.random(alpha_map.shape) < 0.06
    return alpha_map * numpy.where(sheen_pixels, grain_texture * 0.55, grain_texture)


def textured_alpha(
    local_alpha: LocalAlpha, drawing_tool: DrawingTool, random_source: numpy.random.Generator
) -> LocalAlpha:
    """Apply the tool's opacity and texture to a rasterised mark.

    Args:
        local_alpha: Coverage of the mark.
        drawing_tool: Tool that drew it.
        random_source: Random generator.

    Returns:
        New local alpha with values 0..1.
    """
    alpha_map = local_alpha.alpha_map * drawing_tool.peak_opacity
    if drawing_tool.tool_kind == PENCIL_TOOL:
        alpha_map = graphite_texture(alpha_map, random_source)
    elif drawing_tool.tool_kind == HIGHLIGHTER_TOOL:
        alpha_map = cv2.GaussianBlur(alpha_map, (0, 0), 1.2)
    else:
        ink_density = cv2.GaussianBlur(random_source.normal(0, 1, alpha_map.shape).astype(numpy.float32), (0, 0), 2.0)
        alpha_map = alpha_map * numpy.clip(0.93 + 0.07 * ink_density / (ink_density.std() + 1e-6), 0.7, 1.0)
    return LocalAlpha(numpy.clip(alpha_map, 0.0, 1.0).astype(numpy.float32), local_alpha.x_offset, local_alpha.y_offset)


def clip_window(
    local_alpha: LocalAlpha, page_height: int, page_width: int
) -> tuple[slice, slice, numpy.ndarray] | None:
    """Clip a local alpha window to the page.

    Args:
        local_alpha: Mark coverage.
        page_height: Page height in pixels.
        page_width: Page width in pixels.

    Returns:
        ``(row_slice, column_slice, clipped_alpha)``, or None when fully outside the page.
    """
    alpha_height, alpha_width = local_alpha.alpha_map.shape
    row_start, column_start = max(0, local_alpha.y_offset), max(0, local_alpha.x_offset)
    final_row = min(page_height, local_alpha.y_offset + alpha_height)
    column_end = min(page_width, local_alpha.x_offset + alpha_width)
    if final_row <= row_start or column_end <= column_start:
        return None
    clipped_alpha = local_alpha.alpha_map[
        row_start - local_alpha.y_offset : final_row - local_alpha.y_offset,
        column_start - local_alpha.x_offset : column_end - local_alpha.x_offset,
    ]
    return slice(row_start, final_row), slice(column_start, column_end), clipped_alpha


def composite_mark(
    page_image: numpy.ndarray,
    label_mask: numpy.ndarray,
    local_alpha: LocalAlpha,
    drawing_tool: DrawingTool,
    ink_mask: numpy.ndarray,
    under_print: bool,
) -> None:
    """Multiply-blend one mark onto the page and record its label pixels.

    Usage:
        Called by the sample generator for every mark; both arrays are updated in place.

    Args:
        page_image: (H, W, 3) float32 page, values 0..1, mutated.
        label_mask: (H, W) uint8 mask of the mark's class, mutated (255 where the mark is).
        local_alpha: Textured coverage of the mark.
        drawing_tool: Tool that drew it (pigment color).
        ink_mask: Printed-ink mask; with ``under_print`` the mark is weakened under glyphs.
        under_print: True when the mark goes under the printed glyphs.

    Returns:
        None.
    """
    clipped_window = clip_window(local_alpha, page_image.shape[0], page_image.shape[1])
    if clipped_window is None:
        return
    row_slice, column_slice, clipped_alpha = clipped_window
    label_threshold = (
        HIGHLIGHTER_LABEL_THRESHOLD if drawing_tool.tool_kind == HIGHLIGHTER_TOOL else LABEL_ALPHA_THRESHOLD
    )
    label_window = label_mask[row_slice, column_slice]
    label_window[clipped_alpha >= label_threshold] = 255
    blend_alpha = clipped_alpha
    if under_print:
        blend_alpha = numpy.where(ink_mask[row_slice, column_slice] > 0, clipped_alpha * 0.25, clipped_alpha)
    pigment_color = numpy.array(drawing_tool.tool_color, numpy.float32) / 255.0
    page_window = page_image[row_slice, column_slice]
    page_window *= 1.0 - blend_alpha[..., None] * (1.0 - pigment_color)
