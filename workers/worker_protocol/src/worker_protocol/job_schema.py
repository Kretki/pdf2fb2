"""Layout job description and helpers shared by all job types.

Layout job  (orchestrator -> layout_worker): page images to segment.
    Result: JSON document ``{"pages": [{"image_file": str, "boxes": [box, ...]}]}``,
    box = ``{"label": str, "bbox": [x1, y1, x2, y2], "score": float, "order": int | None}``
    in pixel coordinates of the page image.

All image and result paths inside a job are relative to ``work_directory``.
"""

from dataclasses import dataclass
from pathlib import Path

from worker_protocol.field_validation import (
    optional_string,
    require_absolute_directory,
    require_list,
    require_number,
    require_string,
    validate_device_name,
)
from worker_protocol.path_safety import resolve_inside_directory

MAX_PAGE_COUNT = 20_000


def optional_directory(job_payload: dict, field_name: str) -> Path | None:
    """Read an optional absolute directory field.

    Usage:
        Use for model directories, which some backends do not need.

    Args:
        job_payload: Decoded JSON object.
        field_name: Key to read.

    Returns:
        The directory Path, or None when the field is absent/null.

    Raises:
        ValueError: If the field is present but not an existing absolute directory.
    """
    directory_text = optional_string(job_payload, field_name)
    return None if directory_text is None else require_absolute_directory(directory_text, field_name)


def checked_relative_path(work_directory: Path, relative_path: str) -> str:
    """Validate a relative job path and return it unchanged.

    Usage:
        Use while decoding jobs, so traversal is rejected up front.

    Args:
        work_directory: Working directory of the job.
        relative_path: Path string from the job file.

    Returns:
        ``relative_path`` unchanged.

    Raises:
        ValueError: If the path escapes ``work_directory``.
    """
    resolve_inside_directory(work_directory, relative_path)
    return relative_path


@dataclass(frozen=True)
class LayoutJob:
    """Request to segment page images with the layout model.

    Usage:
        Built by the orchestrator, serialised with ``to_payload`` and decoded by
        the layout worker with ``from_payload``.

    Attributes:
        work_directory: Absolute directory all relative paths refer to.
        model_directory: Absolute directory of the PP-DocLayoutV3 weights.
        device_name: Paddle device, e.g. ``gpu:0`` or ``cpu``.
        score_threshold: Minimum detection score, 0..1.
        page_images: Relative page image paths, in page order.
        result_file: Relative path of the JSON result document.
    """

    work_directory: Path
    model_directory: Path | None
    device_name: str
    score_threshold: float
    page_images: tuple[str, ...]
    result_file: str

    def to_payload(self) -> dict:
        """Serialise the job to a JSON-compatible object.

        Usage:
            Call before writing the job file.

        Returns:
            Dictionary with string paths, ready for ``json.dumps``.
        """
        return {
            "work_directory": str(self.work_directory),
            "model_directory": None if self.model_directory is None else str(self.model_directory),
            "device_name": self.device_name,
            "score_threshold": self.score_threshold,
            "page_images": list(self.page_images),
            "result_file": self.result_file,
        }

    @classmethod
    def from_payload(cls, job_payload: dict) -> "LayoutJob":
        """Decode and validate a layout job.

        Usage:
            Call in the layout worker right after reading the job file.

        Args:
            job_payload: Decoded JSON object produced by ``to_payload``.

        Returns:
            The validated LayoutJob.

        Raises:
            ValueError: If any field is missing, out of range or escapes the working directory.
        """
        work_directory = require_absolute_directory(require_string(job_payload, "work_directory"), "work_directory")
        page_entries = require_list(job_payload, "page_images", MAX_PAGE_COUNT)
        return cls(
            work_directory=work_directory,
            model_directory=optional_directory(job_payload, "model_directory"),
            device_name=validate_device_name(require_string(job_payload, "device_name", 16)),
            score_threshold=require_number(job_payload, "score_threshold", 0.0, 1.0),
            page_images=tuple(checked_relative_path(work_directory, page_entry) for page_entry in page_entries),
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
