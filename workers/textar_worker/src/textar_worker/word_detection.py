"""Word boxes from docTR's DBNet (ResNet-50) detector, loaded from local weights only.

TexTAR was trained on word crops cut from docTR ``db_resnet50`` boxes, so the same detector is
used here. Only detection runs; text recognition comes from Qianfan-OCR in the main pipeline.
"""

import os
from pathlib import Path

import numpy
import torch

os.environ.setdefault("USE_TORCH", "1")
from doctr.models import db_resnet50, detection_predictor  # noqa: E402  (docTR reads USE_TORCH at import)

DETECTOR_WEIGHTS_FILE = "db_resnet50-79bd7d70.pt"
PixelWordBox = tuple[int, int, int, int]


class WordDetector:
    """Detects axis-aligned word boxes on page images.

    Usage:
        Build with ``load_word_detector`` and call ``detect_words`` per page.
    """

    def __init__(self, detection_model: object) -> None:
        """Store the docTR detection predictor.

        Args:
            detection_model: ``doctr`` DetectionPredictor.
        """
        self._detection_model = detection_model

    @torch.inference_mode()
    def detect_words(self, page_pixels: numpy.ndarray) -> list[PixelWordBox]:
        """Find the word boxes of one page.

        Args:
            page_pixels: ``(H, W, 3)`` uint8 RGB page.

        Returns:
            ``(x0, y0, x1, y1)`` pixel boxes with positive width and height, in detector order.
        """
        page_height, page_width = page_pixels.shape[:2]
        page_prediction = self._detection_model([page_pixels])[0]
        relative_boxes = page_prediction["words"] if isinstance(page_prediction, dict) else page_prediction
        word_boxes = []
        for relative_box in numpy.asarray(relative_boxes).reshape(-1, 5):
            pixel_box = (round(float(relative_box[0]) * page_width), round(float(relative_box[1]) * page_height),
                         round(float(relative_box[2]) * page_width), round(float(relative_box[3]) * page_height))
            if pixel_box[2] > pixel_box[0] and pixel_box[3] > pixel_box[1]:
                word_boxes.append(pixel_box)
        return word_boxes


def load_word_detector(detector_directory: Path, torch_device: torch.device) -> WordDetector:
    """Build the docTR detector from the local weight file (no download).

    Args:
        detector_directory: Directory containing ``db_resnet50-79bd7d70.pt``.
        torch_device: Device to run on.

    Returns:
        Ready detector.

    Raises:
        FileNotFoundError: If the weight file is missing.
    """
    weights_path = detector_directory / DETECTOR_WEIGHTS_FILE
    if not weights_path.is_file():
        raise FileNotFoundError(f"docTR detector weights not found: {weights_path}")
    detection_network = db_resnet50(pretrained=False, pretrained_backbone=False)
    # weights_only=True: torch's restricted unpickler accepts tensors and plain containers only, no code.
    detector_state = torch.load(  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        weights_path, map_location="cpu", weights_only=True)
    detection_network.load_state_dict(detector_state)
    detection_network = detection_network.eval().to(torch_device)
    return WordDetector(detection_predictor(arch=detection_network, pretrained=False, assume_straight_pages=True))
