"""Scanner realism applied after compositing (STAGE_CLEAN.md §7.4).

Photometric effects change only the image; geometric effects are applied identically to the
image (bilinear) and to the label masks (nearest neighbour).
"""

import cv2
import numpy


def yellow_and_tone(page_image: numpy.ndarray, random_source: numpy.random.Generator) -> numpy.ndarray:
    """Paper yellowing tint plus brightness, contrast and gamma changes.

    Args:
        page_image: (H, W, 3) float32 page 0..1.
        random_source: Random generator.

    Returns:
        New float32 page 0..1.
    """
    paper_tint = numpy.array([1.0, random_source.uniform(0.97, 1.0), random_source.uniform(0.88, 1.0)], numpy.float32)
    toned_page = page_image * paper_tint
    contrast = random_source.uniform(0.85, 1.12)
    brightness = random_source.uniform(-0.05, 0.05)
    toned_page = (toned_page - 0.5) * contrast + 0.5 + brightness
    gamma_value = random_source.uniform(0.8, 1.25)
    return numpy.clip(toned_page, 0.0, 1.0) ** gamma_value


def bleed_through(
    page_image: numpy.ndarray, verso_image: numpy.ndarray, random_source: numpy.random.Generator
) -> numpy.ndarray:
    """Ghost of the mirrored back page showing through the paper.

    Args:
        page_image: (H, W, 3) float32 page 0..1.
        verso_image: Another page (any size) float32 0..1, used as the back side.
        random_source: Random generator.

    Returns:
        New float32 page 0..1.
    """
    mirrored = cv2.resize(verso_image[:, ::-1], (page_image.shape[1], page_image.shape[0]))
    mirrored = cv2.GaussianBlur(mirrored, (0, 0), random_source.uniform(0.8, 2.0))
    ghost_strength = random_source.uniform(0.04, 0.16)
    return page_image * (1.0 - ghost_strength * (1.0 - mirrored))


def gutter_shading(page_image: numpy.ndarray, random_source: numpy.random.Generator) -> numpy.ndarray:
    """Page-curl darkening that fades out from the gutter side.

    Args:
        page_image: (H, W, 3) float32 page 0..1.
        random_source: Random generator.

    Returns:
        New float32 page 0..1.
    """
    page_width = page_image.shape[1]
    shade_width = random_source.uniform(0.03, 0.12) * page_width
    distance = numpy.arange(page_width, dtype=numpy.float32)
    if random_source.random() < 0.5:
        distance = distance[::-1]
    shade_profile = 1.0 - random_source.uniform(0.08, 0.3) * numpy.exp(-distance / shade_width)
    return page_image * shade_profile[None, :, None]


def apply_photometric(
    page_image: numpy.ndarray, verso_image: numpy.ndarray | None, random_source: numpy.random.Generator
) -> numpy.ndarray:
    """Yellowing/tone, optional bleed-through and optional gutter shading.

    Args:
        page_image: (H, W, 3) float32 page 0..1.
        verso_image: Back page for bleed-through, or None.
        random_source: Random generator.

    Returns:
        New float32 page 0..1.
    """
    result_page = yellow_and_tone(page_image, random_source)
    if verso_image is not None and random_source.random() < 0.35:
        result_page = bleed_through(result_page, verso_image, random_source)
    if random_source.random() < 0.3:
        result_page = gutter_shading(result_page, random_source)
    return result_page


def random_homography(page_height: int, page_width: int, random_source: numpy.random.Generator) -> numpy.ndarray:
    """Small rotation (±2°) combined with a slight perspective warp.

    Args:
        page_height: Page height in pixels.
        page_width: Page width in pixels.
        random_source: Random generator.

    Returns:
        3x3 float64 homography.
    """
    rotation = cv2.getRotationMatrix2D((page_width / 2, page_height / 2), random_source.uniform(-2, 2), 1.0)
    rotation_matrix = numpy.vstack([rotation, [0, 0, 1]])
    corner_points = numpy.float32([[0, 0], [page_width, 0], [page_width, page_height], [0, page_height]])
    corner_jitter = random_source.uniform(-0.008, 0.008, (4, 2)).astype(numpy.float32) * [page_width, page_height]
    perspective = cv2.getPerspectiveTransform(corner_points, corner_points + corner_jitter.astype(numpy.float32))
    return perspective @ rotation_matrix


def apply_geometric(
    page_image: numpy.ndarray, class_masks: numpy.ndarray, random_source: numpy.random.Generator
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Warp image and masks with the same homography.

    Args:
        page_image: (H, W, 3) float32 page 0..1.
        class_masks: (C, H, W) uint8 masks.
        random_source: Random generator.

    Returns:
        ``(warped_page, warped_masks)`` with unchanged shapes.
    """
    page_height, page_width = page_image.shape[:2]
    homography = random_homography(page_height, page_width, random_source)
    warped_page = cv2.warpPerspective(
        page_image, homography, (page_width, page_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    warped_masks = numpy.stack(
        [
            cv2.warpPerspective(
                class_mask, homography, (page_width, page_height), flags=cv2.INTER_NEAREST, borderValue=0
            )
            for class_mask in class_masks
        ]
    )
    return warped_page, warped_masks


def finish_scan(page_image: numpy.ndarray, random_source: numpy.random.Generator) -> numpy.ndarray:
    """Slight blur, sensor noise and JPEG compression (q 60..95), then 8-bit quantisation.

    Args:
        page_image: (H, W, 3) float32 page 0..1.
        random_source: Random generator.

    Returns:
        (H, W, 3) uint8 RGB page.
    """
    blur_sigma = random_source.uniform(0.0, 0.8)
    finished_page = cv2.GaussianBlur(page_image, (0, 0), blur_sigma) if blur_sigma > 0.25 else page_image
    finished_page = finished_page + random_source.normal(0, random_source.uniform(0.0, 0.015), finished_page.shape)
    page_bytes = (numpy.clip(finished_page, 0.0, 1.0) * 255.0 + 0.5).astype(numpy.uint8)
    if random_source.random() < 0.7:
        jpeg_quality = int(random_source.integers(60, 96))
        _, encoded_jpeg = cv2.imencode(
            ".jpg", cv2.cvtColor(page_bytes, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )
        page_bytes = cv2.cvtColor(cv2.imdecode(encoded_jpeg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return page_bytes
