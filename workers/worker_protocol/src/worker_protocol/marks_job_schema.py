"""Reader-mark cleaning job (orchestrator -> readermarks worker): page images to clean.

Each page may carry ``masked_regions`` (picture boxes found by the layout model) that the worker
blanks with paper colour before detection, so printed pictures are never taken for reader marks.

Result: JSON document ``{"pages": [{"image_file": str, "clean_file": str, "marks_file": str,
"has_marks": bool, "mark_count": int}]}``; ``clean_file`` is a byte copy of the page when
``has_marks`` is false. All paths are relative to ``work_directory``.
"""

import math
from dataclasses import dataclass
from pathlib import Path

from worker_protocol.field_validation import (
    require_absolute_directory,
    require_list,
    require_string,
    validate_device_name,
)
from worker_protocol.job_schema import MAX_PAGE_COUNT, checked_relative_path
from worker_protocol.path_safety import resolve_inside_directory

MAX_REGIONS_PER_PAGE = 512
MAX_COORDINATE = 1_000_000.0
RegionBox = tuple[float, float, float, float]


def require_boolean(job_payload: dict, field_name: str) -> bool:
    """Return a mandatory boolean field.

    Args:
        job_payload: Decoded JSON object.
        field_name: Key to read.

    Returns:
        The boolean value.

    Raises:
        ValueError: If the field is missing or not a boolean.
    """
    field_value = job_payload.get(field_name)
    if not isinstance(field_value, bool):
        raise ValueError(f"Field {field_name!r} must be a boolean.")
    return field_value


def validate_region_box(candidate_box: object) -> RegionBox:
    """Validate one ``[x_start, y_start, x_end, y_end]`` pixel box.

    Args:
        candidate_box: Decoded JSON value.

    Returns:
        The box as a float tuple.

    Raises:
        ValueError: If it is not four finite, ordered, bounded numbers.
    """
    if not isinstance(candidate_box, (list, tuple)) or len(candidate_box) != 4:
        raise ValueError("A masked region must be [x_start, y_start, x_end, y_end].")
    if not all(isinstance(coordinate, (int, float)) and not isinstance(coordinate, bool)
               and math.isfinite(coordinate) and 0 <= coordinate <= MAX_COORDINATE for coordinate in candidate_box):
        raise ValueError("Masked region coordinates must be finite numbers within 0..1e6.")
    if candidate_box[2] < candidate_box[0] or candidate_box[3] < candidate_box[1]:
        raise ValueError("Masked region end must not precede its start.")
    return tuple(float(coordinate) for coordinate in candidate_box)


def validate_masked_regions(job_payload: dict, page_count: int) -> tuple[tuple[RegionBox, ...], ...]:
    """Read the per-page masked regions (absent field = no regions).

    Args:
        job_payload: Decoded JSON object.
        page_count: Number of pages in the job.

    Returns:
        One tuple of boxes per page.

    Raises:
        ValueError: If the field does not have one bounded list of valid boxes per page.
    """
    raw_regions = job_payload.get("masked_regions")
    if raw_regions is None:
        return tuple(() for _ in range(page_count))
    if not isinstance(raw_regions, list) or len(raw_regions) != page_count:
        raise ValueError("Field 'masked_regions' must hold one list per page.")
    if not all(isinstance(page_regions, list) and len(page_regions) <= MAX_REGIONS_PER_PAGE
               for page_regions in raw_regions):
        raise ValueError(f"Each page may have at most {MAX_REGIONS_PER_PAGE} masked regions.")
    return tuple(tuple(validate_region_box(candidate_box) for candidate_box in page_regions)
                 for page_regions in raw_regions)


@dataclass(frozen=True)
class MarksJob:
    """Request to detect and remove reader marks on page images.

    Usage:
        Built by the orchestrator, serialised with ``to_payload`` and decoded by the
        readermarks worker with ``from_payload``.

    Attributes:
        work_directory: Absolute directory all relative paths refer to.
        checkpoint_directory: Absolute directory with ``spotter.pt``, ``segmenter.pt``, ``thresholds.json``.
        device_name: Torch device, e.g. ``cuda:0`` / ``gpu:0`` / ``cpu``.
        page_images: Relative page image paths, in page order.
        output_directory: Relative directory for per-page outputs.
        result_file: Relative path of the JSON result document.
        write_debug: Whether debug overlays are written.
        masked_regions: Per page, picture boxes to blank before detection.
    """

    work_directory: Path
    checkpoint_directory: Path
    device_name: str
    page_images: tuple[str, ...]
    output_directory: str
    result_file: str
    write_debug: bool
    masked_regions: tuple[tuple[RegionBox, ...], ...] = ()

    def to_payload(self) -> dict:
        """Serialise the job to a JSON-compatible object.

        Usage:
            Call before writing the job file.

        Returns:
            Dictionary with string paths, ready for ``json.dumps``.
        """
        return {"work_directory": str(self.work_directory), "checkpoint_directory": str(self.checkpoint_directory),
                "device_name": self.device_name, "page_images": list(self.page_images),
                "output_directory": self.output_directory, "result_file": self.result_file,
                "write_debug": self.write_debug,
                "masked_regions": [[list(region_box) for region_box in page_regions]
                                   for page_regions in self.page_regions()]}

    @classmethod
    def from_payload(cls, job_payload: dict) -> "MarksJob":
        """Decode and validate a marks job.

        Usage:
            Call in the readermarks worker right after reading the job file.

        Args:
            job_payload: Decoded JSON object produced by ``to_payload``.

        Returns:
            The validated MarksJob.

        Raises:
            ValueError: If any field is missing, out of range or escapes the working directory.
        """
        work_directory = require_absolute_directory(require_string(job_payload, "work_directory"), "work_directory")
        page_entries = require_list(job_payload, "page_images", MAX_PAGE_COUNT)
        return cls(
            work_directory=work_directory,
            checkpoint_directory=require_absolute_directory(require_string(job_payload, "checkpoint_directory"),
                                                            "checkpoint_directory"),
            device_name=validate_device_name(require_string(job_payload, "device_name", 16)),
            page_images=tuple(checked_relative_path(work_directory, page_entry) for page_entry in page_entries),
            output_directory=checked_relative_path(work_directory, require_string(job_payload, "output_directory")),
            result_file=checked_relative_path(work_directory, require_string(job_payload, "result_file")),
            write_debug=require_boolean(job_payload, "write_debug"),
            masked_regions=validate_masked_regions(job_payload, len(page_entries)),
        )

    def page_regions(self) -> tuple[tuple[RegionBox, ...], ...]:
        """Masked regions of every page (empty tuples when none were given).

        Usage:
            Call instead of reading ``masked_regions`` directly.

        Returns:
            One tuple of boxes per page, in page order.
        """
        return self.masked_regions or tuple(() for _ in self.page_images)

    def resolve_inside(self, relative_path: str) -> Path:
        """Resolve a relative job path against the working directory.

        Usage:
            Call before opening a page image or writing outputs.

        Args:
            relative_path: Path relative to ``work_directory``.

        Returns:
            Absolute path inside ``work_directory``.

        Raises:
            ValueError: If the path escapes ``work_directory``.
        """
        return resolve_inside_directory(self.work_directory, relative_path)
