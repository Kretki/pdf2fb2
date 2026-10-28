"""Qianfan-OCR served by an OpenAI-compatible server (vLLM / SGLang)."""

import base64
import io
import json
import urllib.request

from PIL import Image

from ocr_worker.token_confidence import RecognizedText, joined_char_logprobs
from worker_protocol.field_validation import validate_http_url

MAX_RESPONSE_BYTES = 16 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 900


class HttpOcrEngine:
    """Sends crops to ``<server_url>/chat/completions``; safe to call from many threads.

    Usage:
        Use when Qianfan-OCR runs in vLLM, e.g.
        ``vllm serve models/Qianfan-OCR --served-model-name Qianfan-OCR``.
    """

    supports_parallel_requests = True

    def __init__(self, server_url: str, server_model: str, prompt_text: str,
                 timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        """Store the endpoint configuration.

        Args:
            server_url: Base URL such as ``http://localhost:8000/v1`` (http/https only).
            server_model: Model name exposed by the server.
            prompt_text: Instruction sent with every crop.
            timeout_seconds: Per-request timeout in seconds.

        Raises:
            ValueError: If ``server_url`` is not an http(s) URL.
        """
        self._endpoint_url = validate_http_url(server_url).rstrip("/") + "/chat/completions"
        self._server_model = server_model
        self._prompt_text = prompt_text
        self._timeout_seconds = timeout_seconds

    def recognize_image(self, crop_image: Image.Image, max_new_tokens: int) -> str:
        """Send one crop to the server and return its answer.

        Usage:
            Called by the job runner, possibly from several threads.

        Args:
            crop_image: RGB crop of one layout segment.
            max_new_tokens: Generation budget, 1..32768.

        Returns:
            Message content of the first completion choice.

        Raises:
            urllib.error.URLError: On connection or HTTP errors.
            ValueError: If the response is not the expected JSON shape.
        """
        return _extract_message_text(self._post_request(self._build_request_payload(crop_image, max_new_tokens)))

    def recognize_with_confidence(self, crop_image: Image.Image, max_new_tokens: int) -> RecognizedText:
        """Send one crop asking for token log-probabilities.

        Usage:
            Called by the job runner when the job asks for confidences (the server must support ``logprobs``).

        Args:
            crop_image: RGB crop of one layout segment.
            max_new_tokens: Generation budget, 1..32768.

        Returns:
            Message content and per-character log-probabilities (0 where the server's tokens do not spell it).

        Raises:
            urllib.error.URLError: On connection or HTTP errors.
            ValueError: If the response is not the expected JSON shape.
        """
        request_payload = {**self._build_request_payload(crop_image, max_new_tokens), "logprobs": True}
        response_document = self._post_request(request_payload)
        message_text = _extract_message_text(response_document)
        token_texts, token_logprobs = _extract_token_logprobs(response_document)
        return RecognizedText(message_text, tuple(joined_char_logprobs(token_texts, token_logprobs, message_text)))

    def _post_request(self, request_payload: dict) -> dict:
        """POST a chat-completions request and decode the JSON answer.

        Args:
            request_payload: Request body.

        Returns:
            Decoded response.

        Raises:
            urllib.error.URLError: On connection or HTTP errors.
        """
        request_body = json.dumps(request_payload).encode("utf-8")
        # B310/S310/dynamic-urllib-use-detected: the URL scheme is restricted to http/https in __init__
        # (validate_http_url), so file:// and other schemes cannot reach urlopen.
        http_request = urllib.request.Request(  # nosec B310  # noqa: S310
            self._endpoint_url, data=request_body, headers={"Content-Type": "application/json"}, method="POST")
        request_timeout = self._timeout_seconds
        # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
        with urllib.request.urlopen(http_request, timeout=request_timeout) as http_response:  # nosec B310  # noqa: S310
            return json.loads(http_response.read(MAX_RESPONSE_BYTES))

    def _build_request_payload(self, crop_image: Image.Image, max_new_tokens: int) -> dict:
        """Build the chat-completions request body with the crop as a data URL.

        Args:
            crop_image: RGB crop to embed as base64 PNG.
            max_new_tokens: Generation budget.

        Returns:
            JSON-serialisable request payload.
        """
        image_buffer = io.BytesIO()
        crop_image.save(image_buffer, "PNG")
        encoded_image = base64.b64encode(image_buffer.getvalue()).decode("ascii")
        return {
            "model": self._server_model, "temperature": 0, "max_tokens": max_new_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}},
                {"type": "text", "text": self._prompt_text}]}],
        }


def _extract_message_text(response_document: object) -> str:
    """Pull the first choice's message content out of a chat-completions response.

    Args:
        response_document: Decoded JSON response.

    Returns:
        The message content string.

    Raises:
        ValueError: If the response does not have the expected structure.
    """
    try:
        message_text = response_document["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as shape_error:
        raise ValueError("Unexpected chat-completions response shape.") from shape_error
    if not isinstance(message_text, str):
        raise ValueError("Chat-completions message content is not a string.")
    return message_text


def _extract_token_logprobs(response_document: dict) -> tuple[list[str], list[float]]:
    """Token strings and log-probabilities of the first choice (empty when the server sent none).

    Args:
        response_document: Decoded JSON response.

    Returns:
        ``(token texts, log-probabilities)``.
    """
    try:
        token_entries = response_document["choices"][0]["logprobs"]["content"] or []
    except (KeyError, IndexError, TypeError):
        return [], []
    token_texts = [str(token_entry.get("token", "")) for token_entry in token_entries if isinstance(token_entry, dict)]
    token_logprobs = [float(token_entry.get("logprob", 0.0)) for token_entry in token_entries
                      if isinstance(token_entry, dict)]
    return token_texts, token_logprobs
