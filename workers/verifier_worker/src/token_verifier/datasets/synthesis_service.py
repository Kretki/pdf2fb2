"""``synth`` command: turn exported reference chunks into synthetic labelled examples.

Reads ``references_<split>.jsonl`` (written by ``pdf2fb2-bench export-references``) and writes
``synthetic_<split>.jsonl`` into the same data directory. Every corrupted copy of a chunk is cut into segments
of log-uniform length (``min_segment_units`` .. ``max_segment_units``), matching the paragraph-sized segments the
pipeline verifies, and every segment becomes one example whose neighbouring reference units are kept as
unlabelled page context (as the pipeline sends the neighbouring segments of a page).
"""

import dataclasses
import math
import random
from collections import defaultdict
from pathlib import Path

from token_verifier.datasets.example_files import TokenExample, read_json_lines, write_examples
from token_verifier.datasets.noise_synthesis import corrupt_chunk
from token_verifier.verifier_settings import SynthesisSettings

CONTEXT_UNITS = 32


def reference_chunks(references_path: Path) -> dict[tuple[str, str], list[list[str]]]:
    """Reference chunks grouped by book.

    Args:
        references_path: ``references_<split>.jsonl``.

    Returns:
        Mapping ``(book_key, language)`` -> list of unit lists.
    """
    chunks_by_book: dict[tuple[str, str], list[list[str]]] = defaultdict(list)
    for line_object in read_json_lines(references_path):
        chunks_by_book[(str(line_object.get("book_key", "")), str(line_object.get("language", "")))].append(
            [str(unit_text) for unit_text in line_object["words"]])
    return chunks_by_book


def segment_ranges(unit_count: int, synthesis_settings: SynthesisSettings,
                   noise_generator: random.Random) -> list[tuple[int, int]]:
    """Consecutive segments of log-uniform length covering a chunk.

    Args:
        unit_count: Units of the chunk.
        synthesis_settings: Segment length limits.
        noise_generator: Random generator.

    Returns:
        ``(start, end)`` pairs in order.
    """
    lower_bound = math.log(synthesis_settings.min_segment_units)
    upper_bound = math.log(synthesis_settings.max_segment_units)
    segment_bounds, segment_start = [], 0
    while segment_start < unit_count:
        segment_length = round(math.exp(noise_generator.uniform(lower_bound, upper_bound)))
        segment_end = min(unit_count, segment_start + max(1, segment_length))
        segment_bounds.append((segment_start, segment_end))
        segment_start = segment_end
    return segment_bounds


def chunk_examples(book_key: str, language_name: str, chunk_units: list[str], donor_units: list[str],
                   synthesis_settings: SynthesisSettings, noise_generator: random.Random) -> list[TokenExample]:
    """Corrupted segment examples of one chunk (``examples_per_chunk`` copies, each segmented anew).

    Args:
        book_key: Source book.
        language_name: Corpus language.
        chunk_units: Units of the chunk.
        donor_units: Units of another chunk of the same book.
        synthesis_settings: Rates, limits and copies.
        noise_generator: Random generator.

    Returns:
        Examples.
    """
    return [dataclasses.replace(
        corrupt_chunk(book_key, language_name, chunk_units[segment_start:segment_end], donor_units,
                      synthesis_settings, noise_generator),
        context_before=tuple(chunk_units[max(0, segment_start - CONTEXT_UNITS):segment_start]),
        context_after=tuple(chunk_units[segment_end:segment_end + CONTEXT_UNITS]))
        for _ in range(synthesis_settings.examples_per_chunk)
        for segment_start, segment_end in segment_ranges(len(chunk_units), synthesis_settings, noise_generator)]


def synthesize_split(data_directory: Path, split_name: str, synthesis_settings: SynthesisSettings) -> Path:
    """Write ``synthetic_<split>.jsonl``.

    Usage:
        ``python -m token_verifier synth --split train`` (and ``valid``).

    Args:
        data_directory: Directory with the reference export.
        split_name: ``train``, ``valid`` or ``test``.
        synthesis_settings: Rates, limits and seed.

    Returns:
        Path of the written file.

    Raises:
        FileNotFoundError: If the reference export is missing.
    """
    references_path = data_directory / f"references_{split_name}.jsonl"
    noise_generator = random.Random(synthesis_settings.random_seed)  # nosec B311  # noqa: S311 - data synthesis, not cryptography
    token_examples: list[TokenExample] = []
    for (book_key, language_name), book_chunks in sorted(reference_chunks(references_path).items()):
        for chunk_index, chunk_units in enumerate(book_chunks):
            donor_units = book_chunks[(chunk_index + 1) % len(book_chunks)] if len(book_chunks) > 1 else []
            token_examples.extend(chunk_examples(book_key, language_name, chunk_units, donor_units,
                                                 synthesis_settings, noise_generator))
    output_path = data_directory / f"synthetic_{split_name}.jsonl"
    write_examples(output_path, token_examples)
    return output_path
