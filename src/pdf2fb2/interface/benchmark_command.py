"""``pdf2fb2-bench``: benchmark runs over the test corpus and verifier data export (wiring only).

Subcommands: ``splits`` (create / show the locked book split), ``run`` (convert sampled pages of a split and
score them), ``score`` (score an existing run again), ``export-references`` and ``export-aligned`` (verifier
training data). Only books with ``ocr_state == correct`` in ``catalog.csv`` are used.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from pdf2fb2.benchmark.benchmark_runner import RunSettings, books_to_convert, run_benchmark
from pdf2fb2.benchmark.benchmark_scoring import markdown_report, score_run
from pdf2fb2.benchmark.book_splits import SPLIT_NAMES, locked_splits
from pdf2fb2.benchmark.corpus_catalog import read_correct_books
from pdf2fb2.benchmark.verifier_export import export_aligned, export_references
from pdf2fb2.interface.command_line import ABORTED_EXIT_CODE, build_argument_parser, run_command_line
from pdf2fb2.interface.input_validation import bounded_integer, parse_safe_path
from pdf2fb2.interface.pipeline_wiring import build_source_registry
from pdf2fb2.interface.run_planning import confirm_benchmark_run
from pdf2fb2.project_paths import PROJECT_ROOT

DEFAULT_CORPUS_ROOT = PROJECT_ROOT / "input" / "tests"
DEFAULT_DATA_DIRECTORY = PROJECT_ROOT / "benchmark_data"
DEFAULT_RUNS_DIRECTORY = PROJECT_ROOT / "benchmark_runs"


class CommandLineConverter:
    """Runs the normal ``pdf2fb2`` command line on one PDF.

    Usage:
        Built with the extra pipeline options of the run (e.g. ``--no-proofread``).
    """

    def __init__(self, pipeline_arguments: tuple[str, ...]) -> None:
        """Store the extra options.

        Args:
            pipeline_arguments: Options appended to every conversion.
        """
        self._pipeline_arguments = pipeline_arguments

    def convert_book(self, pdf_path: Path, language_code: str, output_directory: Path) -> int:
        """Convert one PDF.

        Args:
            pdf_path: PDF to convert.
            language_code: Value of ``--lang``.
            output_directory: Output folder.

        Returns:
            Exit code of ``pdf2fb2``.
        """
        return run_command_line([str(pdf_path), "-o", str(output_directory), "--lang", language_code, "--no-vis",
                                 "--yes", *self._pipeline_arguments])


def add_common_arguments(command_parser: argparse.ArgumentParser) -> None:
    """Options shared by every subcommand.

    Args:
        command_parser: Subcommand parser (mutated).

    Returns:
        None.
    """
    command_parser.add_argument("--corpus", dest="corpus_root", type=parse_safe_path, default=DEFAULT_CORPUS_ROOT,
                                help="corpus directory with catalog.csv (default input/tests)")
    command_parser.add_argument("--data-dir", dest="data_directory", type=parse_safe_path,
                                default=DEFAULT_DATA_DIRECTORY, help="splits.json and exported data")


def build_benchmark_parser() -> argparse.ArgumentParser:
    """Argument parser of ``pdf2fb2-bench``.

    Returns:
        Parser with the five subcommands.
    """
    argument_parser = argparse.ArgumentParser(prog="pdf2fb2-bench", description=__doc__)
    subcommand_parsers = argument_parser.add_subparsers(dest="command_name", required=True)
    add_common_arguments(subcommand_parsers.add_parser("splits", help="create or show the locked split"))
    run_parser = subcommand_parsers.add_parser("run", help="convert sampled pages of a split and score them")
    add_common_arguments(run_parser)
    run_parser.add_argument("--split", dest="split_name", choices=SPLIT_NAMES, default="test")
    run_parser.add_argument("--name", dest="run_name", type=parse_safe_path, required=True,
                            help="run folder under benchmark_runs/")
    run_parser.add_argument("--pages", dest="pages_per_book", type=bounded_integer(1, 1000), default=10)
    run_parser.add_argument("--languages", dest="language_names", default="",
                            help="comma-separated corpus languages, e.g. old_russian,english")
    run_parser.add_argument("--reuse", dest="reuse_outputs", action="store_true",
                            help="keep existing conversions of the run folder")
    run_parser.add_argument("--yes", dest="skip_confirmation", action="store_true",
                            help="start without waiting for Enter after the time estimate")
    run_parser.add_argument("pipeline_arguments", nargs=argparse.REMAINDER,
                            help="after '--': options passed to every pdf2fb2 conversion")
    score_parser = subcommand_parsers.add_parser("score", help="score an existing run again")
    add_common_arguments(score_parser)
    score_parser.add_argument("--name", dest="run_name", type=parse_safe_path, required=True)
    references_parser = subcommand_parsers.add_parser("export-references", help="reference chunks for synthesis")
    add_common_arguments(references_parser)
    references_parser.add_argument("--split", dest="split_name", choices=SPLIT_NAMES, default="train")
    references_parser.add_argument("--chunks-per-language", dest="chunks_per_language",
                                   type=bounded_integer(1, 50_000), default=600,
                                   help="reference chunks per language, shared by its books (default 600)")
    aligned_parser = subcommand_parsers.add_parser("export-aligned", help="labelled pipeline output of a run")
    add_common_arguments(aligned_parser)
    aligned_parser.add_argument("--name", dest="run_name", type=parse_safe_path, required=True)
    return argument_parser


def run_split_command(parsed_arguments: argparse.Namespace) -> int:
    """Create (if needed) and summarise the split.

    Args:
        parsed_arguments: Parsed options.

    Returns:
        Exit code 0.
    """
    catalog_books = read_correct_books(parsed_arguments.corpus_root)
    book_splits = locked_splits(catalog_books, parsed_arguments.data_directory)
    split_counts = Counter((catalog_book.language_name, book_splits[catalog_book.book_key])
                           for catalog_book in catalog_books)
    print(json.dumps({f"{language_name}/{split_name}": book_count
                      for (language_name, split_name), book_count in sorted(split_counts.items())}, indent=1))
    return 0


def run_benchmark_split(parsed_arguments: argparse.Namespace) -> int:
    """Convert and score a split.

    Args:
        parsed_arguments: Parsed options.

    Returns:
        Exit code 0, or 130 when the admin aborts after the estimate.
    """
    pipeline_arguments = tuple(argument_text for argument_text in parsed_arguments.pipeline_arguments
                               if argument_text != "--")
    run_settings = RunSettings(
        corpus_root=parsed_arguments.corpus_root, data_directory=parsed_arguments.data_directory,
        run_directory=DEFAULT_RUNS_DIRECTORY / parsed_arguments.run_name, split_name=parsed_arguments.split_name,
        pages_per_book=parsed_arguments.pages_per_book,
        language_names=tuple(language_name for language_name in parsed_arguments.language_names.split(",")
                             if language_name),
        reuse_outputs=parsed_arguments.reuse_outputs)
    if not confirm_benchmark_run(build_argument_parser().parse_args(["--no-vis", *pipeline_arguments]),
                                 books_to_convert(run_settings), run_settings.pages_per_book,
                                 not parsed_arguments.skip_confirmation):
        print("pdf2fb2-bench: aborted")
        return ABORTED_EXIT_CODE
    run_benchmark(run_settings, build_source_registry(), CommandLineConverter(pipeline_arguments))
    print(markdown_report(score_run(run_settings.run_directory, parsed_arguments.corpus_root)))
    return 0


def run_score_command(parsed_arguments: argparse.Namespace) -> int:
    """Score an existing run again.

    Args:
        parsed_arguments: Parsed options.

    Returns:
        Exit code 0.
    """
    print(markdown_report(score_run(DEFAULT_RUNS_DIRECTORY / parsed_arguments.run_name, parsed_arguments.corpus_root)))
    return 0


def run_references_command(parsed_arguments: argparse.Namespace) -> int:
    """Export reference chunks of a split.

    Args:
        parsed_arguments: Parsed options.

    Returns:
        Exit code 0.
    """
    print(export_references(parsed_arguments.corpus_root, parsed_arguments.data_directory,
                            parsed_arguments.split_name, parsed_arguments.chunks_per_language))
    return 0


def run_aligned_command(parsed_arguments: argparse.Namespace) -> int:
    """Export the labelled pipeline output of a run.

    Args:
        parsed_arguments: Parsed options.

    Returns:
        Exit code 0.
    """
    print(export_aligned(DEFAULT_RUNS_DIRECTORY / parsed_arguments.run_name, parsed_arguments.corpus_root,
                         parsed_arguments.data_directory))
    return 0


COMMAND_HANDLERS = {"splits": run_split_command, "run": run_benchmark_split, "score": run_score_command,
                    "export-references": run_references_command, "export-aligned": run_aligned_command}


def run_benchmark_command(argument_list: list[str] | None = None) -> int:
    """Entry point of ``pdf2fb2-bench``.

    Usage:
        ``uv run pdf2fb2-bench <subcommand> [options]``.

    Args:
        argument_list: Arguments without the program name; None means ``sys.argv[1:]``.

    Returns:
        Exit code of the subcommand.
    """
    parsed_arguments = build_benchmark_parser().parse_args(argument_list)
    return COMMAND_HANDLERS[parsed_arguments.command_name](parsed_arguments)
