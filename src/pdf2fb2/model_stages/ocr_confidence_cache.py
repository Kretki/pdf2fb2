"""Persistent per-character OCR confidences, keyed like the OCR text cache.

The text cache keeps cleaned text only; the confidences belong to the raw model output (before cleaning), so
both are stored here and mapped to the final text's units later by alignment (``text_verification.unit_evidence``).
"""

import math
from pathlib import Path

from worker_protocol.json_files import read_json_document, write_json_document

CONFIDENCE_CACHE_FILE = "ocr_confidence.json"
MAX_RAW_LENGTH = 200_000


def valid_entry(cache_entry: object) -> bool:
    """Tell whether a cache entry is ``{"text": str, "logprobs": [number per character]}``.

    Args:
        cache_entry: Decoded value.

    Returns:
        True for a well-formed entry.
    """
    if not isinstance(cache_entry, dict):
        return False
    raw_text, char_logprobs = cache_entry.get("text"), cache_entry.get("logprobs")
    return isinstance(raw_text, str) and len(raw_text) <= MAX_RAW_LENGTH and isinstance(char_logprobs, list) \
        and len(char_logprobs) == len(raw_text) and all(
            isinstance(char_logprob, int | float) and math.isfinite(char_logprob) for char_logprob in char_logprobs)


class OcrConfidenceCache:
    """Maps segment keys to raw OCR text with per-character log-probabilities.

    Usage:
        Opened by the OCR stage next to ``ocr_cache.json``; filled from worker results; read when segments get
        their ``ocr_confidence``.
    """

    def __init__(self, cache_path: Path) -> None:
        """Load the cache file if it exists (malformed entries are dropped).

        Args:
            cache_path: JSON file.

        Raises:
            ValueError: If the existing file is not a valid JSON object.
        """
        self._cache_path = cache_path
        loaded_entries = read_json_document(cache_path) if cache_path.exists() else {}
        self._entries = {entry_key: cache_entry for entry_key, cache_entry in loaded_entries.items()
                         if valid_entry(cache_entry)}

    def lookup_entry(self, segment_key: str) -> dict | None:
        """Raw text and log-probabilities of a segment.

        Args:
            segment_key: Key from ``build_segment_key``.

        Returns:
            ``{"raw_text": str, "char_logprobs": [float]}`` or None.
        """
        cache_entry = self._entries.get(segment_key)
        if cache_entry is None:
            return None
        return {"raw_text": cache_entry["text"], "char_logprobs": list(cache_entry["logprobs"])}

    def store_entry(self, segment_key: str, raw_text: str, char_logprobs: list[float]) -> None:
        """Add or replace an entry when it is well formed (in memory; call ``save_to_disk``).

        Args:
            segment_key: Key from ``build_segment_key``.
            raw_text: Model output before cleaning.
            char_logprobs: One log-probability per character of ``raw_text``.

        Returns:
            None.
        """
        cache_entry = {"text": raw_text, "logprobs": char_logprobs}
        if valid_entry(cache_entry):
            self._entries[segment_key] = cache_entry

    def save_to_disk(self) -> None:
        """Write the cache file.

        Returns:
            None.
        """
        write_json_document(self._cache_path, self._entries)
