"""Persistent cache of OCR results keyed by page and segment box."""

from pathlib import Path

from worker_protocol.json_files import read_json_document, write_json_document


def build_segment_key(page_number: int, page_segment: dict) -> str:
    """Build the cache key of a segment.

    Args:
        page_number: 1-based page number.
        page_segment: Segment with ``bbox`` in pixels.

    Returns:
        Key like ``"3:120,88,940,410"``.
    """
    return f"{page_number}:" + ",".join(str(int(coordinate)) for coordinate in page_segment["bbox"])


class OcrTextCache:
    """Maps segment keys to cleaned OCR text, stored as JSON in the output directory.

    Usage:
        Load at the start of the OCR stage, ``store`` new texts, then ``save_to_disk``.
        Re-running a conversion skips every segment already in the cache.
    """

    def __init__(self, cache_path: Path) -> None:
        """Load the cache file if it exists.

        Args:
            cache_path: JSON file holding ``{segment_key: text}``.

        Raises:
            ValueError: If the existing file is not a valid JSON object.
        """
        self._cache_path = cache_path
        loaded_entries = read_json_document(cache_path) if cache_path.exists() else {}
        self._cached_texts = {entry_key: entry_text for entry_key, entry_text in loaded_entries.items()
                              if isinstance(entry_text, str)}

    def __contains__(self, segment_key: str) -> bool:
        """Tell whether a segment has a cached text.

        Args:
            segment_key: Key from ``build_segment_key``.

        Returns:
            True if cached.
        """
        return segment_key in self._cached_texts

    @property
    def entry_count(self) -> int:
        """Number of cached segments.

        Returns:
            Count of entries.
        """
        return len(self._cached_texts)

    def lookup_text(self, segment_key: str) -> str:
        """Return the cached text of a segment.

        Args:
            segment_key: Key from ``build_segment_key``.

        Returns:
            Cached text, or an empty string when missing.
        """
        return self._cached_texts.get(segment_key, "")

    def store_text(self, segment_key: str, segment_text: str) -> None:
        """Add or replace a cached text (in memory; call ``save_to_disk`` to persist).

        Args:
            segment_key: Key from ``build_segment_key``.
            segment_text: Cleaned OCR text.

        Returns:
            None.
        """
        self._cached_texts[segment_key] = segment_text

    def save_to_disk(self) -> None:
        """Write the cache to disk atomically.

        Returns:
            None.

        Raises:
            OSError: If the file cannot be written.
        """
        write_json_document(self._cache_path, self._cached_texts)
