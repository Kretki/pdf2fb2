"""Runs a style job: per page detect words, predict their attributes and group them by text region."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy
import torch
from PIL import Image

from textar_worker.attribute_inference import AttributePredictor
from textar_worker.region_assignment import group_by_region, reading_order
from textar_worker.textar_network import load_textar_network
from textar_worker.word_crops import page_word_crops
from textar_worker.word_detection import WordDetector, load_word_detector
from worker_protocol.style_job_schema import StyleJob, StylePage

PageCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class StyleModels:
    """Loaded models of the worker.

    Attributes:
        word_detector: docTR word box detector.
        attribute_predictor: TexTAR runner.
    """

    word_detector: WordDetector
    attribute_predictor: AttributePredictor


def torch_device(device_name: str) -> torch.device:
    """Map a pdf2fb2 device name (``gpu:0``, ``cuda:0``, ``cpu``) to a torch device.

    Args:
        device_name: Validated device name.

    Returns:
        Torch device; CUDA names fall back to CPU when CUDA is unavailable.
    """
    if device_name == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name.replace("gpu", "cuda"))


def load_style_models(style_job: StyleJob) -> StyleModels:
    """Load the detector and TexTAR from the job's local model directories.

    Args:
        style_job: Validated job.

    Returns:
        Loaded models.
    """
    selected_device = torch_device(style_job.device_name)
    return StyleModels(word_detector=load_word_detector(style_job.detector_directory, selected_device),
                       attribute_predictor=AttributePredictor(
                           load_textar_network(style_job.textar_directory, selected_device), selected_device))


def style_single_page(style_models: StyleModels, style_job: StyleJob, style_page: StylePage) -> dict:
    """Word attributes of the text regions of one page.

    Args:
        style_models: Loaded models.
        style_job: Job (resolves the image path).
        style_page: Page entry.

    Returns:
        ``{"image_file", "regions"}`` result entry.
    """
    with Image.open(style_job.resolve_inside(style_page.image_file)) as image_file:
        page_image = image_file.convert("RGB")
    detected_boxes = style_models.word_detector.detect_words(numpy.asarray(page_image))
    word_boxes = [detected_boxes[word_index] for word_index in reading_order(detected_boxes)]
    page_words = style_models.attribute_predictor.predict_page(word_boxes, page_word_crops(page_image, word_boxes))
    return {"image_file": style_page.image_file, "regions": group_by_region(page_words, style_page.text_regions)}


def run_style_job(style_job: StyleJob, style_models: StyleModels, page_callback: PageCallback) -> dict:
    """Process every page of the job.

    Args:
        style_job: Validated job.
        style_models: Loaded models.
        page_callback: Called with ``(page_number, page_count)`` before each page.

    Returns:
        Result document (see ``worker_protocol.style_job_schema``).
    """
    page_results = []
    for page_number, style_page in enumerate(style_job.style_pages, start=1):
        page_callback(page_number, len(style_job.style_pages))
        if not style_page.text_regions:
            page_results.append({"image_file": style_page.image_file, "regions": []})
            continue
        page_results.append(style_single_page(style_models, style_job, style_page))
    return {"pages": page_results}
