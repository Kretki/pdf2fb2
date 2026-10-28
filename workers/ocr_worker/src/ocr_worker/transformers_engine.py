"""Qianfan-OCR running in-process with transformers on local weights."""

from functools import partial
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from ocr_worker.cuda_memory_guard import CudaMemoryGuard
from ocr_worker.token_confidence import RecognizedText, incremental_char_logprobs


class TransformersOcrEngine:
    """Qianfan-OCR loaded from a local directory (no network, no remote code).

    Usage:
        Create once per worker process; call ``recognize_image`` sequentially
        (the model is not thread-safe, so ``supports_parallel_requests`` is False).
    """

    supports_parallel_requests = False

    def __init__(self, model_directory: Path, device_name: str, prompt_text: str,
                 memory_guard: CudaMemoryGuard | None = None) -> None:
        """Load the model and processor.

        Args:
            model_directory: Directory with the Qianfan-OCR weights (``models/Qianfan-OCR``).
            device_name: ``gpu:N``/``cuda:N`` or ``cpu``; ``gpu`` is mapped to ``cuda``.
            prompt_text: Instruction sent with every crop.
            memory_guard: Out-of-memory retry policy; None uses the default ``CudaMemoryGuard``.

        Raises:
            OSError: If the weights are missing or incomplete.
            ValueError: If the installed transformers does not know the model type.
        """
        self._prompt_text = prompt_text
        self._memory_guard = memory_guard or CudaMemoryGuard()
        torch_device = device_name.replace("gpu", "cuda")
        print(f"ocr_worker: loading Qianfan-OCR from {model_directory} on {torch_device}", flush=True)
        # B615: weights are loaded from a local directory only (local_files_only=True), never downloaded.
        loaded_model = AutoModelForImageTextToText.from_pretrained(  # nosec B615
            str(model_directory), dtype=torch.bfloat16, device_map=torch_device, local_files_only=True
        )
        self._ocr_model = loaded_model.eval()
        # The config has no transformers_version, so transformers suspects a Mistral tokenizer and warns.
        # Qianfan's Qwen2 tokenizer is correct as saved: True would split mixed-case words (McDonald -> Mc Donald)
        # unlike in training, so the flag is set to False, which only silences the false warning.
        self._processor = AutoProcessor.from_pretrained(  # nosec B615
            str(model_directory), local_files_only=True, fix_mistral_regex=False
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

        Raises:
            torch.cuda.OutOfMemoryError: If the crop still runs out of memory after a cache flush.
            EngineUnavailableError: If several crops in a row ran out of memory (the GPU is unusable).
        """
        return self._memory_guard.run_guarded(partial(self._generate_text, crop_image, max_new_tokens))

    def recognize_with_confidence(self, crop_image: Image.Image, max_new_tokens: int) -> RecognizedText:
        """Read the text in an image crop with per-character log-probabilities.

        Usage:
            Called by the job runner when the job asks for confidences.

        Args:
            crop_image: RGB crop of one layout segment.
            max_new_tokens: Generation budget, 1..32768.

        Returns:
            Text and one log-probability per character.

        Raises:
            torch.cuda.OutOfMemoryError: If the crop still runs out of memory after a cache flush.
            EngineUnavailableError: If several crops in a row ran out of memory.
        """
        return self._memory_guard.run_guarded(partial(self._generate_scored, crop_image, max_new_tokens))

    def _model_inputs(self, crop_image: Image.Image) -> dict:
        """Chat-template inputs of one crop on the model device.

        Args:
            crop_image: RGB crop.

        Returns:
            Model inputs.
        """
        chat_messages = [{"role": "user", "content": [
            {"type": "image", "image": crop_image}, {"type": "text", "text": self._prompt_text}]}]
        return self._processor.apply_chat_template(
            chat_messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self._ocr_model.device)

    def _generate_scored(self, crop_image: Image.Image, max_new_tokens: int) -> RecognizedText:
        """Run one greedy generation that also returns the log-probability of every generated token.

        Args:
            crop_image: RGB crop of one layout segment.
            max_new_tokens: Generation budget, 1..32768.

        Returns:
            Decoded text and per-character log-probabilities.

        Raises:
            torch.cuda.OutOfMemoryError: If the device cannot hold the request.
        """
        model_inputs = self._model_inputs(crop_image)
        with torch.no_grad():
            generation = self._ocr_model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                                  output_scores=True, return_dict_in_generate=True)
            token_logprobs = self._ocr_model.compute_transition_scores(
                generation.sequences, generation.scores, normalize_logits=True)[0].float().cpu().tolist()
        generated_ids = generation.sequences[0, model_inputs["input_ids"].shape[1]:].tolist()
        decoded_text, char_logprobs = incremental_char_logprobs(
            len(generated_ids), lambda prefix_length: self._processor.tokenizer.decode(
                generated_ids[:prefix_length], skip_special_tokens=True), token_logprobs)
        return RecognizedText(decoded_text, tuple(char_logprobs))

    def _generate_text(self, crop_image: Image.Image, max_new_tokens: int) -> str:
        """Run one greedy generation for a crop.

        Args:
            crop_image: RGB crop of one layout segment.
            max_new_tokens: Generation budget, 1..32768.

        Returns:
            Decoded model output without the prompt tokens.

        Raises:
            torch.cuda.OutOfMemoryError: If the device cannot hold the request.
        """
        model_inputs = self._model_inputs(crop_image)
        with torch.no_grad():
            output_ids = self._ocr_model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_length = model_inputs["input_ids"].shape[1]
        return self._processor.batch_decode(output_ids[:, prompt_length:], skip_special_tokens=True)[0]
