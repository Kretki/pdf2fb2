"""Qianfan-OCR running in-process with transformers on local weights."""

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


class TransformersOcrEngine:
    """Qianfan-OCR loaded from a local directory (no network, no remote code).

    Usage:
        Create once per worker process; call ``recognize_image`` sequentially
        (the model is not thread-safe, so ``supports_parallel_requests`` is False).
    """

    supports_parallel_requests = False

    def __init__(self, model_directory: Path, device_name: str, prompt_text: str) -> None:
        """Load the model and processor.

        Args:
            model_directory: Directory with the Qianfan-OCR weights (``models/Qianfan-OCR``).
            device_name: ``gpu:N``/``cuda:N`` or ``cpu``; ``gpu`` is mapped to ``cuda``.
            prompt_text: Instruction sent with every crop.

        Raises:
            OSError: If the weights are missing or incomplete.
            ValueError: If the installed transformers does not know the model type.
        """
        self._prompt_text = prompt_text
        torch_device = device_name.replace("gpu", "cuda")
        print(f"ocr_worker: loading Qianfan-OCR from {model_directory} on {torch_device}", flush=True)
        # B615: weights are loaded from a local directory only (local_files_only=True), never downloaded.
        loaded_model = AutoModelForImageTextToText.from_pretrained(  # nosec B615
            str(model_directory), dtype=torch.bfloat16, device_map=torch_device, local_files_only=True
        )
        self._ocr_model = loaded_model.eval()
        self._processor = AutoProcessor.from_pretrained(  # nosec B615
            str(model_directory), local_files_only=True
        )

    def recognize_image(self, crop_image: Image.Image, max_new_tokens: int) -> str:
        """Read the text in an image crop.

        Usage:
            Called by the job runner for every crop.

        Args:
            crop_image: RGB crop of one layout segment.
            max_new_tokens: Generation budget, 1..32768.

        Returns:
            Decoded model output without the prompt tokens.
        """
        chat_messages = [{"role": "user", "content": [
            {"type": "image", "image": crop_image}, {"type": "text", "text": self._prompt_text}]}]
        model_inputs = self._processor.apply_chat_template(
            chat_messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self._ocr_model.device)
        with torch.no_grad():
            output_ids = self._ocr_model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_length = model_inputs["input_ids"].shape[1]
        return self._processor.batch_decode(output_ids[:, prompt_length:], skip_special_tokens=True)[0]
