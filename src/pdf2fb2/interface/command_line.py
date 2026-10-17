"""Command-line entry point: parses arguments and wires the pipeline together.

Usage:
    uv run pdf2fb2 book.pdf -o out --lang en
    uv run pdf2fb2 --selftest -o out
    # faster OCR: serve Qianfan-OCR with vLLM and point the OCR worker at it
    uv run pdf2fb2 book.pdf --qianfan-url http://localhost:8000/v1 --workers 8
"""

import argparse
import sys
from pathlib import Path

from pdf2fb2.fb2_output.fb2_document import BookOptions
from pdf2fb2.interface.input_validation import (
    bounded_integer,
    parse_device,
    parse_existing_directory,
    parse_existing_pdf,
    parse_language_code,
    parse_probability,
    parse_safe_path,
    parse_server_url,
    parse_short_text,
)
from pdf2fb2.interface.pipeline_wiring import build_pipeline
from pdf2fb2.interface.selftest_document import create_selftest_pdf
from pdf2fb2.pipeline.conversion_pipeline import ConversionRequest
from pdf2fb2.progress.terminal_progress import create_progress_display
from pdf2fb2.project_paths import (
    DOCTR_DETECTOR_DIRECTORY,
    LAYOUT_MODEL_DIRECTORY,
    PROOFREAD_MODEL_DIRECTORY,
    QIANFAN_MODEL_DIRECTORY,
    TEXTAR_MODEL_DIRECTORY,
)
from pdf2fb2.worker_runtime.worker_launcher import WorkerFailedError
from worker_protocol.ocr_job_schema import MAX_PARALLEL_REQUESTS

FAILURE_EXIT_CODE = 1
SELFTEST_PDF_NAME = "selftest.pdf"


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
                                 help="keep the layout labels after OCR (no page-number and heading statistics)")
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


def _add_book_arguments(argument_parser: argparse.ArgumentParser) -> None:
    """Register rendering and FB2 options.

    Args:
        argument_parser: Parser to extend.
    """
    argument_parser.add_argument("--dpi", dest="render_dpi", type=bounded_integer(72, 600), default=200,
                                 help="render DPI for segmentation/OCR")
    argument_parser.add_argument("--image-dpi", dest="image_dpi", type=bounded_integer(72, 600), default=300,
                                 help="DPI for pictures embedded in the FB2")
    argument_parser.add_argument("--lang", dest="language_code", type=parse_language_code, default="en",
                                 help="book language code for FB2 (ru, en, pl, ...)")
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
    argument_parser = argparse.ArgumentParser(description="PDF -> FB2 with PP-DocLayoutV3 + Qianfan-OCR")
    argument_parser.add_argument("pdf_path", nargs="?", type=parse_existing_pdf)
    argument_parser.add_argument("-o", "--out", dest="output_directory", type=parse_safe_path,
                                 default=Path("pdf2fb2_output"))
    argument_parser.add_argument("--no-vis", dest="skip_drawings", action="store_true",
                                 help="skip drawing segmentation boxes")
    argument_parser.add_argument("--selftest", dest="run_selftest", action="store_true",
                                 help="generate a test PDF and convert it")
    _add_model_arguments(argument_parser)
    _add_marks_arguments(argument_parser)
    _add_book_arguments(argument_parser)
    return argument_parser


def build_request(parsed_arguments: argparse.Namespace, pdf_path: Path) -> ConversionRequest:
    """Collect the conversion request from the command line.

    Args:
        parsed_arguments: Parsed command line.
        pdf_path: PDF to convert.

    Returns:
        Conversion request.
    """
    book_options = BookOptions(
        book_title=parsed_arguments.book_title, author_name=parsed_arguments.author_name,
        genre_name=parsed_arguments.genre_name, language_code=parsed_arguments.language_code,
        page_marker_mode=parsed_arguments.page_marker_mode, segmentation_dpi=parsed_arguments.render_dpi,
        image_dpi=parsed_arguments.image_dpi)
    return ConversionRequest(pdf_path=pdf_path, output_directory=parsed_arguments.output_directory,
                             book_options=book_options, draw_segments=not parsed_arguments.skip_drawings)


def resolve_input_pdf(parsed_arguments: argparse.Namespace, argument_parser: argparse.ArgumentParser) -> Path:
    """Return the PDF to convert, generating the self-test PDF when requested.

    Args:
        parsed_arguments: Parsed command line.
        argument_parser: Parser, used to report a missing PDF.

    Returns:
        Path of the PDF.

    Raises:
        SystemExit: Via ``argument_parser.error`` when neither a PDF nor ``--selftest`` is given.
    """
    if parsed_arguments.run_selftest:
        parsed_arguments.output_directory.mkdir(parents=True, exist_ok=True)
        selftest_pdf = create_selftest_pdf(parsed_arguments.output_directory / SELFTEST_PDF_NAME)
        print("Created test PDF:", selftest_pdf)
        return selftest_pdf
    if parsed_arguments.pdf_path is None:
        argument_parser.error("give a PDF path or --selftest")
    return parsed_arguments.pdf_path


def run_command_line(argument_list: list[str] | None = None) -> int:
    """Parse the command line and run one conversion.

    Usage:
        Entry point of the ``pdf2fb2`` script and of ``main.py``.

    Args:
        argument_list: Arguments without the program name; None means ``sys.argv[1:]``.

    Returns:
        Exit code: 0 success, 1 failed checks or a worker failure.
    """
    argument_parser = build_argument_parser()
    parsed_arguments = argument_parser.parse_args(argument_list)
    pdf_path = resolve_input_pdf(parsed_arguments, argument_parser)
    try:
        conversion_pipeline = build_pipeline(parsed_arguments, create_progress_display(pdf_path.name))
        return conversion_pipeline.convert_document(build_request(parsed_arguments, pdf_path))
    except (WorkerFailedError, FileNotFoundError) as pipeline_error:
        print(f"pdf2fb2: {pipeline_error}", file=sys.stderr)
        return FAILURE_EXIT_CODE
