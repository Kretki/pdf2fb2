"""PP-DocLayoutV3 detector backed by PaddleOCR's ``LayoutDetection``."""

from pathlib import Path

from paddleocr import LayoutDetection

LAYOUT_MODEL_NAME = "PP-DocLayoutV3"


class PaddleLayoutDetector:
    """Loads PP-DocLayoutV3 once and detects layout boxes on page images.

    Usage:
        Create once per worker process, then call ``detect_boxes`` per page.
        GPU memory is released when the worker process exits.
    """

    def __init__(self, model_directory: Path | None, device_name: str) -> None:
        """Load the layout model.

        Args:
            model_directory: Directory with local PP-DocLayoutV3 weights; None lets
                PaddleOCR resolve the model by name.
            device_name: Paddle device such as ``gpu:0`` or ``cpu``.

        Raises:
            Exception: Whatever PaddleOCR raises when the model cannot be loaded.
        """
        model_options = {"model_name": LAYOUT_MODEL_NAME, "device": device_name}
        if model_directory is not None:
            model_options["model_dir"] = str(model_directory)
        model_source = model_options.get("model_dir", "model hub")
        print(f"layout_worker: loading {LAYOUT_MODEL_NAME} from {model_source}", flush=True)
        self._layout_model = LayoutDetection(**model_options)

    def detect_boxes(self, image_path: Path, score_threshold: float) -> list[dict]:
        """Detect layout boxes on one page image.

        Usage:
            Called by ``run_layout_job`` for every page.

        Args:
            image_path: Absolute path of the page image (PNG).
            score_threshold: Minimum detection score, 0..1.

        Returns:
            List of boxes ``{"label": str, "bbox": [x1, y1, x2, y2], "score": float,
            "order": int | None}`` in pixel coordinates.
        """
        detected_boxes = []
        for prediction in self._layout_model.predict(str(image_path), batch_size=1, threshold=score_threshold):
            raw_boxes = _unwrap_prediction(prediction).get("boxes", [])
            detected_boxes.extend(_convert_box(raw_layout_box) for raw_layout_box in raw_boxes)
        return detected_boxes


def _unwrap_prediction(prediction: object) -> dict:
    """Return the dictionary payload of a PaddleOCR prediction result.

    Args:
        prediction: Result object yielded by ``LayoutDetection.predict``.

    Returns:
        The ``res`` dictionary when present, else the whole JSON payload.
    """
    prediction_data = prediction.json
    return prediction_data.get("res", prediction_data)


def _convert_box(raw_layout_box: dict) -> dict:
    """Convert a PaddleOCR box into the worker-protocol box format.

    Args:
        raw_layout_box: Box with ``label``, ``coordinate`` (>= 4 numbers), optional ``score`` and ``order``.

    Returns:
        Box ``{"label", "bbox", "score", "order"}`` with plain Python numbers.
    """
    raw_order = raw_layout_box.get("order")
    return {
        "label": str(raw_layout_box.get("label", "text")),
        "bbox": [float(coordinate) for coordinate in raw_layout_box["coordinate"][:4]],
        "score": float(raw_layout_box.get("score", 1.0)),
        "order": int(raw_order) if isinstance(raw_order, (int, float)) else None,
    }
