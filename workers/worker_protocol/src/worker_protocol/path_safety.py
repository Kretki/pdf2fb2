"""Path helpers that keep job-file paths inside one working directory."""

from pathlib import Path, PurePosixPath


def resolve_inside_directory(base_directory: Path, relative_path: str) -> Path:
    """Resolve a job-file path against a working directory, rejecting traversal.

    Usage:
        Call for every path read from a job file before opening it.

    Args:
        base_directory: Absolute working directory every job path must stay inside.
        relative_path: POSIX-style path relative to ``base_directory``; must not be
            absolute and must not contain ``..`` components.

    Returns:
        The resolved absolute path inside ``base_directory``.

    Raises:
        ValueError: If the path is empty, absolute, contains ``..`` or resolves
            (e.g. through a symlink) outside ``base_directory``.
    """
    if not isinstance(relative_path, str) or not relative_path or "\x00" in relative_path:
        raise ValueError("Job path must be a non-empty string without NUL characters.")
    candidate_path = PurePosixPath(relative_path)
    if candidate_path.is_absolute() or ".." in candidate_path.parts:
        raise ValueError(f"Job path must be relative and must not contain '..': {relative_path!r}")
    resolved_base = base_directory.resolve()
    resolved_path = (resolved_base / candidate_path).resolve()
    if not resolved_path.is_relative_to(resolved_base):
        raise ValueError(f"Job path escapes the working directory: {relative_path!r}")
    return resolved_path


def relative_to_directory(base_directory: Path, target_path: Path) -> str:
    """Express a path inside the working directory as a POSIX relative string.

    Usage:
        Call when writing a job file, so workers receive relative paths only.

    Args:
        base_directory: Working directory the job file refers to.
        target_path: Path that must lie inside ``base_directory``.

    Returns:
        The POSIX-style relative path from ``base_directory`` to ``target_path``.

    Raises:
        ValueError: If ``target_path`` is not inside ``base_directory``.
    """
    return target_path.resolve().relative_to(base_directory.resolve()).as_posix()
