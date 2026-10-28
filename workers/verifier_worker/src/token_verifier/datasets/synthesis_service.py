"""``synth`` command: turn exported reference chunks into synthetic labelled examples.

Reads ``references_<split>.jsonl`` (written by ``pdf2fb2-bench export-references``) and writes
``synthetic_<split>.jsonl`` into the same data directory.
"""

import random
from collections import defaultdict
from pathlib import Path

from token_verifier.datasets.example_files import TokenExample, read_json_lines, write_examples
from token_verifier.datasets.noise_synthesis import corrupt_chunk
from token_verifier.verifier_settings import SynthesisSettings


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
            token_examples.extend(corrupt_chunk(book_key, language_name, chunk_units, donor_units,
                                                synthesis_settings, noise_generator)
                                  for _ in range(synthesis_settings.examples_per_chunk))
    output_path = data_directory / f"synthetic_{split_name}.jsonl"
    write_examples(output_path, token_examples)
    return output_path
