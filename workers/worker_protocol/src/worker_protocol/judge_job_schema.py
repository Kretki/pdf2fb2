"""Judge job (orchestrator -> proofread worker): does a printed text line really contain a word as read?

The token verifier is unsure about some units (probability in an uncertainty band). For each such unit the
orchestrator crops the printed line and asks a vision-language model (the local Qwen3.5 checkpoint) whether the
line image shows the word exactly as read. The answer works for any script: the model compares ink and text.

Result: JSON document ``{"items": [{"key": str, "error_probability": float}]}`` in job order, where
``error_probability`` is the model's probability of "no" (the word is not printed as read).
All image and result paths are relative to ``work_directory``.
"""

from dataclasses import dataclass
from pathlib import Path

from worker_protocol.field_validation import (
    require_absolute_directory,
    require_list,
    require_string,
    validate_device_name,
)
from worker_protocol.job_schema import checked_relative_path
from worker_protocol.path_safety import resolve_inside_directory

MAX_JUDGE_ITEMS = 200_000
MAX_LINE_LENGTH = 2_000
MAX_TARGET_LENGTH = 512
MAX_KEY_LENGTH = 160


@dataclass(frozen=True)
class JudgeItem:
    """One unit to judge.

    Attributes:
        item_key: Caller's identifier, echoed in the result.
        image_file: Line crop, relative to the work directory.
        line_text: Text read for the whole line.
        target_text: The unit in question.
    """

    item_key: str
    image_file: str
    line_text: str
    target_text: str

    def to_payload(self) -> dict:
        """Serialise the item.

        Returns:
            JSON-compatible dictionary.
        """
        return {"key": self.item_key, "image_file": self.image_file, "line_text": self.line_text,
                "target_text": self.target_text}


def validate_judge_item(item_entry: object, work_directory: Path) -> JudgeItem:
    """Decode one item.

    Args:
        item_entry: Decoded JSON value.
        work_directory: Directory the image path must stay inside.

    Returns:
        The validated item.

    Raises:
        ValueError: If a field is missing, too long or the image path escapes the work directory.
    """
    if not isinstance(item_entry, dict):
        raise ValueError("Each judge item must be an object.")
    return JudgeItem(require_string(item_entry, "key", MAX_KEY_LENGTH),
                     checked_relative_path(work_directory, require_string(item_entry, "image_file")),
                     require_string(item_entry, "line_text", MAX_LINE_LENGTH),
                     require_string(item_entry, "target_text", MAX_TARGET_LENGTH))


@dataclass(frozen=True)
class JudgeJob:
    """Request to judge uncertain units against their line images.

    Usage:
        Built by the orchestrator, serialised with ``to_payload``, decoded by the proofread worker
        (``--judge-job``) with ``from_payload``.

    Attributes:
        work_directory: Absolute directory relative paths refer to.
        model_directory: Absolute directory of the vision-language model (``models/qwen3.5-9B``).
        device_name: Torch device, e.g. ``gpu:0``.
        judge_items: Units to judge.
        result_file: Relative path of the JSON result document.
    """

    work_directory: Path
    model_directory: Path
    device_name: str
    judge_items: tuple[JudgeItem, ...]
    result_file: str

    def to_payload(self) -> dict:
        """Serialise the job.

        Returns:
            Dictionary ready for ``json.dumps``.
        """
        return {"work_directory": str(self.work_directory), "model_directory": str(self.model_directory),
                "device_name": self.device_name,
                "items": [judge_item.to_payload() for judge_item in self.judge_items],
                "result_file": self.result_file}

    @classmethod
    def from_payload(cls, job_payload: dict) -> "JudgeJob":
        """Decode and validate a judge job.

        Args:
            job_payload: Decoded JSON object produced by ``to_payload``.

        Returns:
            The validated job.

        Raises:
            ValueError: If any field is missing, out of range or escapes the working directory.
        """
        work_directory = require_absolute_directory(require_string(job_payload, "work_directory"), "work_directory")
        return cls(
            work_directory=work_directory,
            model_directory=require_absolute_directory(require_string(job_payload, "model_directory"),
                                                       "model_directory"),
            device_name=validate_device_name(require_string(job_payload, "device_name", 16)),
            judge_items=tuple(validate_judge_item(item_entry, work_directory)
                              for item_entry in require_list(job_payload, "items", MAX_JUDGE_ITEMS)),
            result_file=checked_relative_path(work_directory, require_string(job_payload, "result_file")),
        )

    def resolve_inside(self, relative_path: str) -> Path:
        """Resolve a relative job path against the working directory.

        Args:
            relative_path: Path relative to ``work_directory``.

        Returns:
            Absolute path inside ``work_directory``.

        Raises:
            ValueError: If the path escapes ``work_directory``.
        """
        return resolve_inside_directory(self.work_directory, relative_path)
