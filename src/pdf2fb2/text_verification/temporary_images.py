"""Removal of the image files the verification loop creates only for its own jobs.

Word crops (``verify_crops/``), judge line crops (``judge_lines/``) and the crop images of the second read
(``second_read/ocr_crops/``) are read by the verifier and judge workers during verification and never again, so
they are deleted once a book has been verified (a 300-page book leaves tens of thousands of crops otherwise).
The second read's text cache is kept, so a re-run does not read the segments again.
"""

import shutil
from pathlib import Path

from pdf2fb2.model_stages.ocr_crop_writer import OCR_CROPS_DIRECTORY
from pdf2fb2.text_verification.second_read import SECOND_READ_DIRECTORY
from pdf2fb2.text_verification.uncertain_units import JUDGE_DIRECTORY
from pdf2fb2.text_verification.unit_evidence import CROP_DIRECTORY

TEMPORARY_DIRECTORIES = (CROP_DIRECTORY, JUDGE_DIRECTORY, f"{SECOND_READ_DIRECTORY}/{OCR_CROPS_DIRECTORY}")


def remove_temporary_images(work_directory: Path) -> list[Path]:
    """Delete the verification-only image directories of a book.

    Usage:
        Called by ``VerificationFeedback.verify_segments`` when verification ends (also after a failure).

    Args:
        work_directory: Book output directory.

    Returns:
        Directories that were removed.
    """
    resolved_root = work_directory.resolve()
    removed_directories = []
    for relative_name in TEMPORARY_DIRECTORIES:
        image_directory = (resolved_root / relative_name).resolve()
        is_inside = image_directory.is_relative_to(resolved_root) and image_directory != resolved_root
        if is_inside and image_directory.is_dir() and not image_directory.is_symlink():
            shutil.rmtree(image_directory)
            removed_directories.append(image_directory)
    return removed_directories
