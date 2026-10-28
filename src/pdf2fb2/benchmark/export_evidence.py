"""Page evidence of exported training examples, built with the pipeline's own evidence code.

For every book-text segment of a converted page: the units of the neighbouring segments as context, the
page-evidence columns (OCR confidence, word boxes, second read) and, when requested, word crops written under the
data directory, so training examples carry exactly what pipeline jobs send.
"""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from pdf2fb2.benchmark.page_outputs import segment_text
from pdf2fb2.text_verification.region_mapping import place_units
from pdf2fb2.text_verification.unit_evidence import page_contexts, unit_crops, unit_features
from pdf2fb2.text_verification.verify_units import VerifyUnit, verify_units

UNSAFE_STEM_CHARACTERS = str.maketrans({character_text: "_" for character_text in "/\\:. "})


@dataclass(frozen=True)
class CropOptions:
    """Where word crops of exported examples go.

    Attributes:
        data_directory: Root the crop paths are relative to (``benchmark_data``), or None for no crops.
        stem_prefix: Prefix that makes file names unique (split and book key).
    """

    data_directory: Path | None
    stem_prefix: str


@dataclass(frozen=True)
class SegmentUnits:
    """Units of one book-text segment of a page.

    Attributes:
        page_segment: Segment record.
        unit_text: Unstyled text the units come from.
        segment_units: Units with offsets.
    """

    page_segment: dict
    unit_text: str
    segment_units: list[VerifyUnit]


def page_segment_units(page_segments: list[dict]) -> list[SegmentUnits]:
    """Units of the given segments.

    Args:
        page_segments: Book-text segments of a page in reading order.

    Returns:
        Their units.
    """
    return [SegmentUnits(page_segment, segment_text(page_segment), verify_units(segment_text(page_segment)))
            for page_segment in page_segments]


def segment_crops(page_image: Image.Image | None, placement: object, crop_options: CropOptions,
                  stem_suffix: str) -> list[str | None]:
    """Crops of one segment when crops are requested.

    Args:
        page_image: Opened page image, or None.
        placement: Unit placement or None.
        crop_options: Crop settings.
        stem_suffix: Page and segment part of the file stem.

    Returns:
        Crop path per unit relative to the data directory (empty without crops).
    """
    if page_image is None or crop_options.data_directory is None:
        return []
    crop_stem = f"{crop_options.stem_prefix}_{stem_suffix}".translate(UNSAFE_STEM_CHARACTERS)
    return list(unit_crops(page_image, placement, crop_stem, crop_options.data_directory))


def page_evidence(page_record: dict, segment_entries: list[SegmentUnits], crop_options: CropOptions) -> list[dict]:
    """Evidence keys of every segment's training example.

    Usage:
        Called by ``verifier_export.page_examples`` for every converted page.

    Args:
        page_record: Page record (``image`` for crops).
        segment_entries: Book-text segments of the page with their units.
        crop_options: Crop settings.

    Returns:
        Per segment ``{"context_before", "context_after", "features", "crops"}`` (crops may be empty).
    """
    contexts = page_contexts([[verify_unit.unit_text for verify_unit in segment_entry.segment_units]
                              for segment_entry in segment_entries])
    page_image = None
    if crop_options.data_directory is not None and Path(str(page_record.get("image", ""))).is_file():
        with Image.open(page_record["image"]) as page_file:
            page_image = page_file.convert("RGB")
    evidence_list = []
    for segment_index, (segment_entry, (context_before, context_after)) in enumerate(zip(segment_entries, contexts,
                                                                                         strict=True)):
        placement = place_units(segment_entry.segment_units, segment_entry.page_segment.get("words") or [])
        features = unit_features(segment_entry.page_segment, segment_entry.unit_text, segment_entry.segment_units,
                                 placement)
        evidence_list.append({"context_before": list(context_before), "context_after": list(context_after),
                              "features": {feature_name: list(column_values)
                                           for feature_name, column_values in features.items()},
                              "crops": segment_crops(page_image, placement, crop_options,
                                                     f"p{page_record.get('page', 0)}_s{segment_index}")})
    return evidence_list
