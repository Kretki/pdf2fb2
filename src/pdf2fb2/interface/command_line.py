"""Command-line entry point: parses arguments and wires the pipeline together.

Usage:
    uv run pdf2fb2 book.pdf --lang en                 # results in output/book/
    uv run pdf2fb2 book.djvu --lang ru -o some/dir    # DjVu (needs ddjvu); results in some/dir/
    uv run pdf2fb2 --batch                            # every book under input/ -> output/<sub-folders>/<stem>/
    uv run pdf2fb2 book.pdf --lang ja,en,la           # several languages, main first (or --lang auto)
    uv run pdf2fb2 --selftest
    uv run pdf2fb2 book.pdf --yes                     # print the time estimate, start without waiting for Enter
    # faster OCR: serve Qianfan-OCR with vLLM and point the OCR worker at it
    uv run pdf2fb2 book.pdf --qianfan-url http://localhost:8000/v1 --workers 8
"""

import argparse
import sys
from functools import partial
from pathlib import Path

from pdf2fb2.fb2_output.fb2_document import BookOptions
from pdf2fb2.interface.batch_conversion import STATUS_ERROR, STATUS_OK, convert_books, convert_one_book
from pdf2fb2.interface.book_jobs import BookConversionJob, find_input_books, single_book_job
from pdf2fb2.interface.input_validation import (
    bounded_integer,
    existing_book_parser,
    parse_device,
    parse_existing_directory,
    parse_language_list,
    parse_probability,
    parse_safe_path,
    parse_server_url,
    parse_short_text,
)
from pdf2fb2.interface.pipeline_wiring import build_pipeline, build_source_registry
from pdf2fb2.interface.run_planning import confirm_book_jobs
from pdf2fb2.interface.selftest_document import create_selftest_pdf
from pdf2fb2.pipeline.conversion_pipeline import ConversionRequest
from pdf2fb2.progress.terminal_progress import create_progress_display
from pdf2fb2.project_paths import (
    DOCTR_DETECTOR_DIRECTORY,
    LAYOUT_MODEL_DIRECTORY,
    PROOFREAD_MODEL_DIRECTORY,
    QIANFAN_MODEL_DIRECTORY,
    TEXTAR_MODEL_DIRECTORY,
    VERIFIER_MODEL_DIRECTORY,
)
from pdf2fb2.text_languages.language_profiles import resolve_book_languages
from worker_protocol.ocr_job_schema import MAX_PARALLEL_REQUESTS

FAILURE_EXIT_CODE = 1
SUCCESS_EXIT_CODE = 0
ABORTED_EXIT_CODE = 130
SELFTEST_PDF_NAME = "selftest.pdf"
SELFTEST_DIRECTORY = "selftest"
DEFAULT_INPUT_DIRECTORY = Path("input")
DEFAULT_OUTPUT_ROOT = Path("output")


def _add_model_arguments(argument_parser: argparse.ArgumentParser) -> None:
    """Register model, device and OCR backend options.

    Args:
        argument_parser: Parser to extend.
    """
    argument_parser.add_argument("--device", dest="device_name", type=parse_device, default="gpu:0")
    argument_parser.add_argument("--threshold", dest="score_threshold", type=parse_probability, default=0.5,
                                 help="layout score threshold, 0..1")
    argument_parser.add_argument("--picture-threshold", dest="picture_threshold", type=parse_probability,
                                 default=0.05, help="layout score threshold of the picture proposals for "
                                 "lettering that does not fit the book's font size, 0..1")
    argument_parser.add_argument("--no-picture-recovery", dest="skip_picture_recovery", action="store_true",
                                 help="keep lettering that does not fit the book's font size as text")
    argument_parser.add_argument("--residual-threshold", dest="residual_threshold", type=parse_probability,
                                 default=0.3, help="layout score threshold when pages with text left outside "
                                 "every segment are segmented again, 0..1")
    argument_parser.add_argument("--no-residual-text", dest="skip_residual_text", action="store_true",
                                 help="do not look for text left outside every segment before OCR")
    argument_parser.add_argument("--no-layout-review", dest="skip_layout_review", action="store_true",
                                 help="keep the layout labels (no running-head, page-number and heading statistics)")
    argument_parser.add_argument("--layout-model-dir", dest="layout_model_directory", type=parse_existing_directory,
                                 default=LAYOUT_MODEL_DIRECTORY)
    argument_parser.add_argument("--qianfan-path", dest="qianfan_directory", type=parse_existing_directory,
                                 default=QIANFAN_MODEL_DIRECTORY)
    argument_parser.add_argument("--qianfan-url", dest="qianfan_url", type=parse_server_url,
                                 help="OpenAI-compatible server, e.g. http://localhost:8000/v1")
    argument_parser.add_argument("--qianfan-model", dest="qianfan_model", type=parse_short_text,
                                 default="Qianfan-OCR", help="model name the server exposes")
    argument_parser.add_argument("--workers", dest="parallel_requests", type=bounded_integer(1, MAX_PARALLEL_REQUESTS),
                                 default=8, help="parallel requests (server mode only)")


def _add_marks_arguments(argument_parser: argparse.ArgumentParser) -> None:
    """Register reader-mark (handwriting / pen mark) removal options.

    Args:
        argument_parser: Parser to extend.
    """
    argument_parser.add_argument("--keep-marks", dest="keep_marks", action="store_true",
                                 help="skip the handwriting check, reader-mark removal and the rescan of marked pages")
    argument_parser.add_argument("--marks-model-dir", dest="marks_model_directory", type=parse_existing_directory,
                                 default=None, help="readermarks checkpoints (default models/readermarks)")
    argument_parser.add_argument("--debug-marks", dest="debug_marks", action="store_true",
                                 help="write marks/<page>/debug.png overlays")
    argument_parser.add_argument("--no-styles", dest="skip_styles", action="store_true",
                                 help="skip TexTAR word styles (bold/italic/strikeout) in the FB2")
    argument_parser.add_argument("--textar-model-dir", dest="textar_model_directory", type=parse_existing_directory,
                                 default=TEXTAR_MODEL_DIRECTORY, help="TexTAR checkpoint directory")
    argument_parser.add_argument("--doctr-model-dir", dest="doctr_model_directory", type=parse_existing_directory,
                                 default=DOCTR_DETECTOR_DIRECTORY, help="docTR db_resnet50 word detector weights")
    argument_parser.add_argument("--no-proofread", dest="skip_proofread", action="store_true",
                                 help="skip the language-model check of the text (Qwen3.5) before the FB2 is written")
    argument_parser.add_argument("--proofread-model-dir", dest="proofread_model_directory",
                                 type=parse_existing_directory, default=PROOFREAD_MODEL_DIRECTORY,
                                 help="local language model used for proofreading")
    argument_parser.add_argument("--no-verify", dest="skip_verify", action="store_true",
                                 help="skip the token verifier and its OCR re-reads")
    argument_parser.add_argument("--verify-model-dir", dest="verify_model_directory", type=parse_safe_path,
                                 default=VERIFIER_MODEL_DIRECTORY,
                                 help="fine-tuned token verifier (skipped when it has no config.json)")
    argument_parser.add_argument("--full-proofread", dest="full_proofread", action="store_true",
                                 help="let Qwen3.5 score the whole text even when the token verifier is available")


def _add_book_arguments(argument_parser: argparse.ArgumentParser) -> None:
    """Register rendering and FB2 options.

    Args:
        argument_parser: Parser to extend.
    """
    argument_parser.add_argument("--dpi", dest="render_dpi", type=bounded_integer(72, 600), default=200,
                                 help="render DPI for segmentation/OCR")
    argument_parser.add_argument("--image-dpi", dest="image_dpi", type=bounded_integer(72, 600), default=300,
                                 help="DPI for pictures embedded in the FB2")
    argument_parser.add_argument("--lang", dest="language_codes", type=parse_language_list, default=("en",),
                                 help="book language code (ru, en, ...); several comma-separated codes, main "
                                 "first (ja,en,la), mark paragraphs and quotations in the other languages; "
                                 "auto picks among all built-in languages")
    argument_parser.add_argument("--title", dest="book_title", type=parse_short_text)
    argument_parser.add_argument("--author", dest="author_name", type=parse_short_text)
    argument_parser.add_argument("--genre", dest="genre_name", type=parse_short_text, default="nonfiction")
    argument_parser.add_argument("--page-markers", dest="page_marker_mode", choices=["anchor", "visible", "none"],
                                 default="anchor", help="anchor: invisible id=page_N on the first paragraph of each "
                                 "page; visible: a '— N —' subtitle at each page start")


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser.

    Returns:
        Parser whose ``dest`` names match the attributes used by the wiring code.
    """
    argument_parser = argparse.ArgumentParser(description="PDF/DjVu -> FB2 with PP-DocLayoutV3 + Qianfan-OCR")
    argument_parser.add_argument("book_path", nargs="?",
                                 type=existing_book_parser(build_source_registry().supported_suffixes))
    argument_parser.add_argument("-o", "--out", dest="output_directory", type=parse_safe_path, default=None,
                                 help="result directory of a single book (default <output root>/<book name>)")
    argument_parser.add_argument("--batch", dest="batch_mode", action="store_true",
                                 help="convert every book under --input-dir, one result folder per book")
    argument_parser.add_argument("--input-dir", dest="input_directory", type=parse_safe_path,
                                 default=DEFAULT_INPUT_DIRECTORY, help="books for --batch (default input)")
    argument_parser.add_argument("--output-root", dest="output_root", type=parse_safe_path,
                                 default=DEFAULT_OUTPUT_ROOT, help="root of the result folders (default output)")
    argument_parser.add_argument("--no-vis", dest="skip_drawings", action="store_true",
                                 help="skip drawing segmentation boxes")
    argument_parser.add_argument("--selftest", dest="run_selftest", action="store_true",
                                 help="generate a test PDF and convert it")
    argument_parser.add_argument("--yes", dest="skip_confirmation", action="store_true",
                                 help="start without waiting for Enter after the time estimate")
    _add_model_arguments(argument_parser)
    _add_marks_arguments(argument_parser)
    _add_book_arguments(argument_parser)
    return argument_parser


def build_request(parsed_arguments: argparse.Namespace, book_job: BookConversionJob) -> ConversionRequest:
    """Collect the conversion request of one book from the command line.

    Args:
        parsed_arguments: Parsed command line.
        book_job: Book and its output directory.

    Returns:
        Conversion request.
    """
    main_language, book_languages = resolve_book_languages(parsed_arguments.language_codes)
    book_options = BookOptions(
        book_title=parsed_arguments.book_title, author_name=parsed_arguments.author_name,
        genre_name=parsed_arguments.genre_name, language_code=main_language,
        page_marker_mode=parsed_arguments.page_marker_mode, segmentation_dpi=parsed_arguments.render_dpi,
        image_dpi=parsed_arguments.image_dpi, book_languages=book_languages)
    return ConversionRequest(book_path=book_job.book_path, output_directory=book_job.output_directory,
                             book_options=book_options, draw_segments=not parsed_arguments.skip_drawings)


def check_batch_arguments(parsed_arguments: argparse.Namespace, argument_parser: argparse.ArgumentParser) -> None:
    """Reject options that name a single book in ``--batch`` mode.

    Args:
        parsed_arguments: Parsed command line.
        argument_parser: Parser, used to report the conflict.

    Raises:
        SystemExit: Via ``argument_parser.error`` for a book path, ``-o``, ``--title``, ``--author`` or
            ``--selftest`` together with ``--batch``.
    """
    single_book_options = {"a book path": parsed_arguments.book_path, "-o": parsed_arguments.output_directory,
                           "--title": parsed_arguments.book_title, "--author": parsed_arguments.author_name,
                           "--selftest": parsed_arguments.run_selftest or None}
    conflicting_options = [option_name for option_name, option_value in single_book_options.items()
                           if option_value is not None]
    if conflicting_options:
        argument_parser.error(f"--batch converts every book; remove {', '.join(conflicting_options)} "
                              "(use --output-root for the result root)")


def resolve_book_jobs(parsed_arguments: argparse.Namespace,
                      argument_parser: argparse.ArgumentParser) -> list[BookConversionJob]:
    """Books to convert: every book of the input directory, the self-test PDF or the given book.

    Args:
        parsed_arguments: Parsed command line.
        argument_parser: Parser, used to report a missing book or input directory.

    Returns:
        Book jobs (exactly one outside ``--batch``).

    Raises:
        SystemExit: Via ``argument_parser.error`` on conflicting or missing arguments.
    """
    output_root = parsed_arguments.output_root
    if parsed_arguments.batch_mode:
        check_batch_arguments(parsed_arguments, argument_parser)
        try:
            return find_input_books(parsed_arguments.input_directory, output_root,
                                    build_source_registry().supported_suffixes)
        except FileNotFoundError as missing_error:
            argument_parser.error(str(missing_error))
    if parsed_arguments.run_selftest:
        selftest_directory = parsed_arguments.output_directory or output_root / SELFTEST_DIRECTORY
        selftest_directory.mkdir(parents=True, exist_ok=True)
        return [BookConversionJob(create_selftest_pdf(selftest_directory / SELFTEST_PDF_NAME), selftest_directory)]
    if parsed_arguments.book_path is None:
        argument_parser.error("give a book path (.pdf, .djvu), --batch or --selftest")
    return [single_book_job(parsed_arguments.book_path, output_root, parsed_arguments.output_directory)]


def convert_book_job(parsed_arguments: argparse.Namespace, book_job: BookConversionJob) -> int:
    """Build the pipeline for one book and run it.

    Args:
        parsed_arguments: Parsed command line.
        book_job: Book and its output directory.

    Returns:
        Exit code of the conversion: 0 when all checks passed, 1 otherwise.

    Raises:
        WorkerFailedError, DocumentConversionError, OSError: On a failed conversion.
    """
    conversion_pipeline = build_pipeline(parsed_arguments, create_progress_display(book_job.book_path.name))
    return conversion_pipeline.convert_document(build_request(parsed_arguments, book_job))


def run_command_line(argument_list: list[str] | None = None) -> int:
    """Parse the command line and run one conversion or a batch.

    Usage:
        Entry point of the ``pdf2fb2`` script and of ``main.py``.

    Args:
        argument_list: Arguments without the program name; None means ``sys.argv[1:]``.

    Returns:
        Exit code: 0 success, 1 failed checks, a failed book or a worker failure, 130 aborted by the admin.
    """
    argument_parser = build_argument_parser()
    parsed_arguments = argument_parser.parse_args(argument_list)
    book_jobs = resolve_book_jobs(parsed_arguments, argument_parser)
    if not confirm_book_jobs(parsed_arguments, book_jobs):
        print("pdf2fb2: aborted", file=sys.stderr)
        return ABORTED_EXIT_CODE
    convert_book = partial(convert_book_job, parsed_arguments)
    if parsed_arguments.batch_mode:
        return convert_books(book_jobs, convert_book)
    book_result = convert_one_book(book_jobs[0], convert_book)
    if book_result.status_text == STATUS_ERROR:
        print(f"pdf2fb2: {book_result.error_text}", file=sys.stderr)
    return SUCCESS_EXIT_CODE if book_result.status_text == STATUS_OK else FAILURE_EXIT_CODE
