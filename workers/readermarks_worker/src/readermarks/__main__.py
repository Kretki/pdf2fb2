"""Command-line interface of the reader-mark pipeline (wiring only; logic lives in the service modules).

Usage:
    python -m readermarks env-check
    python -m readermarks synth                 # synthetic train/validation/test splits
    python -m readermarks preview               # 50 preview samples with masks
    python -m readermarks cache-features        # frozen DINOv2 features
    python -m readermarks train-spotter
    python -m readermarks train-segmenter
    python -m readermarks tune                  # thresholds on real validation pages
    python -m readermarks eval --split real_val
    python -m readermarks clean --input page.png --out out_dir/ [--debug]
    python -m readermarks clean --input-dir pages/ --out out_dir/ --batch
    python -m readermarks convert-labels --export labels.json --out out_dir/
    python -m readermarks active-learning
"""

import json
import sys
from pathlib import Path, PurePath

import typer

from readermarks.settings.model_settings import ReaderMarkSettings, load_settings

command_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Reader-mark detection and removal.")
CONFIG_OPTION = typer.Option(None, "--config", help="YAML file overriding default_settings.yaml.")
SPLIT_OPTION = typer.Option("train", "--split", help="synthetic split to preview")
EVALUATION_SPLIT_OPTION = typer.Option("real_val", "--split", help="real_val or real_test")
INPUT_FILE_OPTION = typer.Option(None, "--input", help="one page image")
INPUT_DIRECTORY_OPTION = typer.Option(None, "--input-dir", help="folder of page images (with --batch)")
OUTPUT_OPTION = typer.Option(..., "--out", help="output directory")
BATCH_OPTION = typer.Option(False, "--batch", help="clean every image of --input-dir")
DEBUG_OPTION = typer.Option(False, "--debug", help="also write debug.png overlays")
EXPORT_OPTION = typer.Option(..., "--export", help="Label Studio JSON export")
COUNT_OPTION = typer.Option(30, "--count", min=1, max=1000, help="pages to export")


def checked_path(user_path: Path | None, option_name: str) -> Path | None:
    """Reject user-supplied paths that contain ``..`` components.

    Args:
        user_path: Path from the command line, or None.
        option_name: Option name for the error message.

    Returns:
        The resolved path, or None.

    Raises:
        typer.BadParameter: If the path contains ``..``.
    """
    if user_path is None:
        return None
    if ".." in PurePath(user_path).parts:
        raise typer.BadParameter(f"{option_name} must not contain '..'.")
    return user_path.resolve()


def settings_from(config_file: Path | None) -> ReaderMarkSettings:
    """Load settings, optionally overridden by a user YAML file.

    Args:
        config_file: Optional override file.

    Returns:
        Settings.
    """
    return load_settings(checked_path(config_file, "--config"))


def print_json(json_value: object) -> None:
    """Print a compact JSON summary for the user.

    Args:
        json_value: JSON-serialisable value.

    Returns:
        None.
    """
    typer.echo(json.dumps(json_value, indent=1, default=str))


@command_app.command("env-check")
def environment_command() -> None:
    """Print CUDA/sm_120 support, device capability and a bf16 matmul benchmark."""
    from readermarks.environment_check import describe_environment

    print_json(describe_environment())


@command_app.command("synth")
def synthesis_command(config_file: Path | None = CONFIG_OPTION) -> None:
    """Generate the synthetic train/validation/test splits (deterministic seed, manifest)."""
    from readermarks.synthesis.synthesis_service import generate_synthetic_dataset

    print_json(generate_synthetic_dataset(settings_from(config_file), ("train", "validation", "test")))


@command_app.command("preview")
def preview_command(config_file: Path | None = CONFIG_OPTION, split_name: str = SPLIT_OPTION) -> None:
    """Render preview sheets (samples side by side with their colour-coded masks)."""
    from readermarks.synthesis.synthesis_service import render_previews

    print_json(render_previews(settings_from(config_file), split_name))


@command_app.command("cache-features")
def cache_command(config_file: Path | None = CONFIG_OPTION) -> None:
    """Encode synthetic splits and real pages with the frozen backbone (fp16 cache)."""
    from readermarks.training.training_service import cache_all_features

    print_json(cache_all_features(settings_from(config_file)))


@command_app.command("train-spotter")
def spotter_command(config_file: Path | None = CONFIG_OPTION) -> None:
    """Train the Stage 1 spotter head on cached features."""
    from readermarks.training.training_service import run_spotter_training

    print_json(run_spotter_training(settings_from(config_file)).__dict__)


@command_app.command("train-segmenter")
def segmenter_command(config_file: Path | None = CONFIG_OPTION) -> None:
    """Train the Stage 2 U-Net segmenter on synthetic crops."""
    from readermarks.training.training_service import run_segmenter_training

    print_json(run_segmenter_training(settings_from(config_file)).__dict__)


@command_app.command("tune")
def tuning_command(config_file: Path | None = CONFIG_OPTION) -> None:
    """Tune thresholds on the real validation pages and write thresholds.json."""
    from readermarks.evaluation.evaluation_service import run_tuning

    print_json(run_tuning(settings_from(config_file)))


@command_app.command("eval")
def evaluation_command(config_file: Path | None = CONFIG_OPTION, split_key: str = EVALUATION_SPLIT_OPTION) -> None:
    """Write the evaluation report (Markdown + JSON)."""
    from readermarks.evaluation.evaluation_service import REAL_SPLITS, run_evaluation

    if split_key not in REAL_SPLITS:
        raise typer.BadParameter(f"--split must be one of {sorted(REAL_SPLITS)}.")
    json_path, markdown_path, _ = run_evaluation(settings_from(config_file), split_key)
    typer.echo(markdown_path.read_text(encoding="utf-8"))
    typer.echo(f"report: {markdown_path} / {json_path}")


@command_app.command("clean")
def cleaning_command(
    input_file: Path | None = INPUT_FILE_OPTION,
    input_directory: Path | None = INPUT_DIRECTORY_OPTION,
    output_directory: Path = OUTPUT_OPTION,
    batch_mode: bool = BATCH_OPTION,
    write_debug: bool = DEBUG_OPTION,
    config_file: Path | None = CONFIG_OPTION,
) -> None:
    """Clean one page (--input) or a folder of pages (--input-dir --batch)."""
    from readermarks.inference.cleaning_service import clean_page_batch, clean_page_file, list_input_images
    from readermarks.inference.model_loading import load_models
    from readermarks.training.training_service import compute_device

    if (input_file is None) == (input_directory is None) or batch_mode != (input_directory is not None):
        raise typer.BadParameter("use either --input FILE or --input-dir DIR --batch.")
    page_cleaner = load_models(settings_from(config_file), compute_device()).page_cleaner
    output_root = checked_path(output_directory, "--out")
    if input_file is not None:
        written_pages = [clean_page_file(page_cleaner, checked_path(input_file, "--input"), output_root, write_debug)]
    else:
        written_pages = clean_page_batch(
            page_cleaner, list_input_images(checked_path(input_directory, "--input-dir")), output_root, write_debug
        )
    print_json({"pages": len(written_pages), "with_marks": sum(real_page.has_marks for real_page in written_pages)})


@command_app.command("convert-labels")
def conversion_command(
    export_file: Path = EXPORT_OPTION, output_directory: Path = OUTPUT_OPTION, config_file: Path | None = CONFIG_OPTION
) -> None:
    """Convert a Label Studio export into pixel masks and patch labels (.npz per task)."""
    from readermarks.labeling.labeling_service import convert_export

    print_json(
        convert_export(
            settings_from(config_file), checked_path(export_file, "--export"), checked_path(output_directory, "--out")
        )
    )


@command_app.command("active-learning")
def active_learning_command(config_file: Path | None = CONFIG_OPTION, task_count: int = COUNT_OPTION) -> None:
    """Export the most uncertain unlabeled pages as a pre-filled Label Studio task file."""
    from readermarks.labeling.labeling_service import run_active_learning

    print_json(run_active_learning(settings_from(config_file), task_count))


def run_command_line() -> int:
    """Entry point of ``python -m readermarks`` and the ``readermarks`` script.

    Returns:
        Process exit code.
    """
    command_app()
    return 0


if __name__ == "__main__":
    sys.exit(run_command_line())
