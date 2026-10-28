"""Abstraction every OCR backend implements."""

from typing import Protocol, runtime_checkable

from PIL import Image

from ocr_worker.token_confidence import RecognizedText


class EngineUnavailableError(RuntimeError):
    """Raised by an engine that can no longer read any crop (e.g. its GPU refuses every allocation).

    Usage:
        The job runner stops the job on this error instead of recording a failure for every
        remaining crop; the worker then exits with a code the orchestrator treats as a failure.
    """


class OcrEngine(Protocol):
    """Reads the text of one image crop.

    Usage:
        Implemented by ``TransformersOcrEngine`` and ``HttpOcrEngine``; the job
        runner depends only on this protocol.

    Attributes:
        supports_parallel_requests: True when ``recognize_image`` may be called
            from several threads at once.
    """

    supports_parallel_requests: bool

    def recognize_image(self, crop_image: Image.Image, max_new_tokens: int) -> str:
        """Read the text in an image crop.

        Args:
            crop_image: RGB crop of one layout segment.
            max_new_tokens: Generation budget, 1..32768.

        Returns:
            Raw model output (Markdown / HTML tables).

        Raises:
            EngineUnavailableError: If the engine cannot serve any further crop.
            Exception: Any other error fails only this crop.
        """
        ...


@runtime_checkable
class ConfidenceOcrEngine(Protocol):
    """Reads a crop and reports how sure the model was of every character.

    Usage:
        Optional capability: the job runner uses it when the job asks for confidences and the engine has it.
    """

    def recognize_with_confidence(self, crop_image: Image.Image, max_new_tokens: int) -> RecognizedText:
        """Read the text in an image crop with per-character log-probabilities.

        Args:
            crop_image: RGB crop of one layout segment.
            max_new_tokens: Generation budget, 1..32768.

        Returns:
            Text and one log-probability per character.

        Raises:
            EngineUnavailableError: If the engine cannot serve any further crop.
        """
        ...
