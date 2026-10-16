"""Text-style job (orchestrator -> textar worker): word attributes inside the text regions of pages.

The worker detects word boxes on every page (docTR DBNet), classifies them with TexTAR using the
page's words as context and returns, per requested region, the words whose centre lies inside it.

Result: JSON document ``{"pages": [{"image_file": str, "regions": [[word, ...], ...]}]}`` with one
word list per requested region, words as ``{"bbox": [x0, y0, x1, y1], "emphasis_class": int,
"emphasis_score": float, "line_class": int, "line_score": float}``. ``emphasis_class`` is
0 normal / 1 bold / 2 italic / 3 bold+italic, ``line_class`` 0 none / 1 underline / 2 strikeout /
3 underline+strikeout. All paths are relative to ``work_directory``.
"""

from dataclasses import dataclass
from pathlib import Path

from worker_protocol.field_validation import (
    require_absolute_directory,
    require_list,
    require_string,
    validate_device_name,
)
from worker_protocol.job_schema import MAX_PAGE_COUNT, checked_relative_path
from worker_protocol.marks_job_schema import MAX_REGIONS_PER_PAGE, RegionBox, validate_region_box
from worker_protocol.path_safety import resolve_inside_directory

EMPHASIS_CLASS_NAMES = ("normal", "bold", "italic", "bold_italic")
LINE_CLASS_NAMES = ("none", "underline", "strikeout", "underline_strikeout")


@dataclass(frozen=True)
class StylePage:
    """One page image plus the text regions whose word styles are wanted.

    Attributes:
        image_file: Page image path relative to the job's working directory.
        text_regions: ``(x_start, y_start, x_end, y_end)`` pixel boxes of the text segments.
    """

    image_file: str
    text_regions: tuple[RegionBox, ...]

    def to_payload(self) -> dict:
        """Serialise the page entry.

        Returns:
            JSON-compatible dictionary.
        """
        return {"image_file": self.image_file, "text_regions": [list(region_box) for region_box in self.text_regions]}


def validate_style_page(work_directory: Path, page_entry: object) -> StylePage:
    """Decode one page entry of a style job.

    Args:
        work_directory: Job working directory (paths must stay inside it).
        page_entry: Decoded JSON value.

    Returns:
        The validated page.

    Raises:
        ValueError: If the entry is not an object with a safe image path and bounded valid regions.
    """
    if not isinstance(page_entry, dict):
        raise ValueError("Each style page must be an object.")
    region_entries = require_list(page_entry, "text_regions", MAX_REGIONS_PER_PAGE)
    return StylePage(image_file=checked_relative_path(work_directory, require_string(page_entry, "image_file")),
                     text_regions=tuple(validate_region_box(region_entry) for region_entry in region_entries))


@dataclass(frozen=True)
class StyleJob:
    """Request to detect word-level text attributes (bold, italic, underline, strikeout).

    Usage:
        Built by the orchestrator, serialised with ``to_payload`` and decoded by the textar worker
        with ``from_payload``.

    Attributes:
        work_directory: Absolute directory all relative paths refer to.
        textar_directory: Absolute directory with ``TexTAR-trained.pt``.
        detector_directory: Absolute directory with the docTR ``db_resnet50`` weights.
        device_name: Torch device, e.g. ``cuda:0`` / ``gpu:0`` / ``cpu``.
        style_pages: Pages with their text regions, in page order.
        result_file: Relative path of the JSON result document.
    """

    work_directory: Path
    textar_directory: Path
    detector_directory: Path
    device_name: str
    style_pages: tuple[StylePage, ...]
    result_file: str

    def to_payload(self) -> dict:
        """Serialise the job to a JSON-compatible object.

        Usage:
            Call before writing the job file.

        Returns:
            Dictionary with string paths, ready for ``json.dumps``.
        """
        return {"work_directory": str(self.work_directory), "textar_directory": str(self.textar_directory),
                "detector_directory": str(self.detector_directory), "device_name": self.device_name,
                "style_pages": [style_page.to_payload() for style_page in self.style_pages],
                "result_file": self.result_file}

    @classmethod
    def from_payload(cls, job_payload: dict) -> "StyleJob":
        """Decode and validate a style job.

        Usage:
            Call in the textar worker right after reading the job file.

        Args:
            job_payload: Decoded JSON object produced by ``to_payload``.

        Returns:
            The validated StyleJob.

        Raises:
            ValueError: If any field is missing, out of range or escapes the working directory.
        """
        work_directory = require_absolute_directory(require_string(job_payload, "work_directory"), "work_directory")
        page_entries = require_list(job_payload, "style_pages", MAX_PAGE_COUNT)
        return cls(
            work_directory=work_directory,
            textar_directory=require_absolute_directory(require_string(job_payload, "textar_directory"),
                                                        "textar_directory"),
            detector_directory=require_absolute_directory(require_string(job_payload, "detector_directory"),
                                                          "detector_directory"),
            device_name=validate_device_name(require_string(job_payload, "device_name", 16)),
            style_pages=tuple(validate_style_page(work_directory, page_entry) for page_entry in page_entries),
            result_file=checked_relative_path(work_directory, require_string(job_payload, "result_file")),
        )

    def resolve_inside(self, relative_path: str) -> Path:
        """Resolve a relative job path against the working directory.

        Usage:
            Call before opening a page image or writing the result.

        Args:
            relative_path: Path relative to ``work_directory``.

        Returns:
            Absolute path inside ``work_directory``.

        Raises:
            ValueError: If the path escapes ``work_directory``.
        """
        return resolve_inside_directory(self.work_directory, relative_path)
