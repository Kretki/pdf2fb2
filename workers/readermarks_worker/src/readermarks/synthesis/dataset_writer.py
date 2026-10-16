"""Write a deterministic synthetic dataset split (images, packed masks, manifest) in parallel.

Layout: ``<data>/synthetic/<split>/sample_NNNNNN.png`` + ``sample_NNNNNN_masks.npz`` and
``<data>/synthetic/<split>/manifest.jsonl`` (one JSON line per sample).
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy

from readermarks.datasets.image_files import read_rgb_image, write_rgb_image
from readermarks.datasets.mask_archive import write_mask_archive
from readermarks.settings.class_catalog import SPOTTER_CLASSES
from readermarks.settings.model_settings import SynthesisSettings
from readermarks.synthesis.page_analysis import PageAnalysis, analyse_page
from readermarks.synthesis.sample_generator import generate_sample

MANIFEST_NAME = "manifest.jsonl"
SPLIT_SEED_OFFSETS = {"train": 0, "validation": 1_000_000, "test": 2_000_000}
WORKER_STATE: dict = {}


@dataclass(frozen=True)
class SplitRequest:
    """What to generate for one split.

    Attributes:
        split_name: ``train``, ``validation`` or ``test``.
        base_paths: Clean page images used as bases.
        sample_count: Number of samples.
        output_directory: Directory of the split.
    """

    split_name: str
    base_paths: tuple[Path, ...]
    sample_count: int
    output_directory: Path


def initialise_worker(base_paths: tuple[Path, ...], synthesis_settings: SynthesisSettings) -> None:
    """Load and analyse the base pages once per worker process.

    Args:
        base_paths: Clean page images.
        synthesis_settings: Generator settings.

    Returns:
        None. Fills the module-level ``WORKER_STATE``.
    """
    base_images = [read_rgb_image(base_path) for base_path in base_paths]
    WORKER_STATE["base_images"] = base_images
    WORKER_STATE["analyses"] = [analyse_page(base_image) for base_image in base_images]
    WORKER_STATE["settings"] = synthesis_settings


def build_one_sample(sample_index: int, sample_seed: int, output_directory: Path) -> dict:
    """Generate and write one sample inside a worker process.

    Args:
        sample_index: Index within the split.
        sample_seed: Seed of this sample.
        output_directory: Directory of the split.

    Returns:
        Manifest record of the sample.
    """
    random_source = numpy.random.default_rng(sample_seed)
    base_images: list[numpy.ndarray] = WORKER_STATE["base_images"]
    analyses: list[PageAnalysis] = WORKER_STATE["analyses"]
    base_index = int(random_source.integers(len(base_images)))
    verso_index = int(random_source.integers(len(base_images)))
    verso_image = base_images[verso_index].astype(numpy.float32) / 255.0 if verso_index != base_index else None
    synthetic_sample = generate_sample(
        base_images[base_index], analyses[base_index], verso_image, WORKER_STATE["settings"], random_source
    )
    sample_name = f"sample_{sample_index:06d}"
    write_rgb_image(output_directory / f"{sample_name}.png", synthetic_sample.page_image)
    write_mask_archive(output_directory / f"{sample_name}_masks.npz", synthetic_sample.class_masks)
    class_pixels = {
        class_name: int((class_mask > 0).sum())
        for class_name, class_mask in zip(SPOTTER_CLASSES, synthetic_sample.class_masks, strict=True)
    }
    return {
        "sample": sample_name,
        "seed": sample_seed,
        "base_page": base_index,
        "has_marks": synthetic_sample.has_marks,
        "rule_page": synthetic_sample.sample_recipe.rule_page,
        "mark_count": synthetic_sample.sample_recipe.mark_count,
        "class_pixels": class_pixels,
    }


def write_split(split_request: SplitRequest, synthesis_settings: SynthesisSettings, worker_count: int) -> Path:
    """Generate a whole split in parallel and write its manifest.

    Usage:
        Called by the ``synth`` command for train/validation/test.

    Args:
        split_request: Split description.
        synthesis_settings: Generator settings (seed, mixing rules).
        worker_count: Number of worker processes (>= 1).

    Returns:
        Path of the written manifest.
    """
    split_request.output_directory.mkdir(parents=True, exist_ok=True)
    seed_offset = synthesis_settings.random_seed + SPLIT_SEED_OFFSETS[split_request.split_name]
    sample_indices = range(split_request.sample_count)
    with ProcessPoolExecutor(
        max_workers=worker_count, initializer=initialise_worker, initargs=(split_request.base_paths, synthesis_settings)
    ) as process_pool:
        manifest_records = list(
            process_pool.map(
                build_one_sample,
                sample_indices,
                [seed_offset + sample_index for sample_index in sample_indices],
                [split_request.output_directory] * len(sample_indices),
                chunksize=8,
            )
        )
    manifest_path = split_request.output_directory / MANIFEST_NAME
    manifest_lines = [json.dumps(manifest_record) for manifest_record in manifest_records]
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return manifest_path


def default_worker_count() -> int:
    """Number of generator processes: all cores but two, at least one.

    Returns:
        Process count.
    """
    return max(1, (os.cpu_count() or 2) - 2)


def read_manifest(split_directory: Path) -> list[dict]:
    """Read the manifest of a generated split.

    Args:
        split_directory: Directory of the split.

    Returns:
        Manifest records in sample order.

    Raises:
        FileNotFoundError: If the split has not been generated.
    """
    manifest_text = (split_directory / MANIFEST_NAME).read_text(encoding="utf-8")
    return [json.loads(manifest_line) for manifest_line in manifest_text.splitlines() if manifest_line.strip()]
