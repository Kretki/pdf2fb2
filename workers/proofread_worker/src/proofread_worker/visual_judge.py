"""Vision-language judge: does a line image show a word exactly as it was read?

The local Qwen3.5 checkpoint is loaded with its vision tower (4-bit on CUDA). For every item the model sees the
line crop and a short question and the probability of answering "no" (vs "yes") is read from the next-token
distribution: no text is generated, so the answer is cheap and deterministic.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from proofread_worker.model_loading import torch_device

QUESTION_TEMPLATE = ("This image shows one printed text line. It was read as:\n{line_text}\n\n"
                     "Is the word \"{target_text}\" printed in the image exactly like this (same letters)? "
                     "Answer yes or no.")
MAX_IMAGE_SIDE = 1600


@dataclass
class LoadedJudge:
    """Vision-language model ready to judge.

    Attributes:
        vision_model: Image-text-to-text model.
        model_processor: Processor (chat template, image preprocessing, tokenizer).
        answer_ids: Token ids of " yes"/"yes" and " no"/"no" style answers.
    """

    vision_model: torch.nn.Module
    model_processor: object
    answer_ids: tuple[tuple[int, ...], tuple[int, ...]]


def answer_token_ids(model_processor: object) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """First token ids of the "yes" and "no" answer spellings.

    Args:
        model_processor: Processor with a tokenizer.

    Returns:
        ``(yes ids, no ids)``.
    """
    text_tokenizer = model_processor.tokenizer
    return tuple({text_tokenizer.encode(answer_text, add_special_tokens=False)[0]
                  for answer_text in ("yes", "Yes", " yes", " Yes")}), \
        tuple({text_tokenizer.encode(answer_text, add_special_tokens=False)[0]
               for answer_text in ("no", "No", " no", " No")})


def load_judge(model_directory: Path, device_name: str) -> LoadedJudge:
    """Load the vision-language model from local files.

    Args:
        model_directory: ``models/qwen3.5-9B``.
        device_name: ``gpu:N`` / ``cuda:N`` / ``cpu``.

    Returns:
        The judge.

    Raises:
        OSError: If the weights are missing.
    """
    selected_device = torch_device(device_name)
    model_options: dict = {"dtype": torch.bfloat16, "local_files_only": True}
    if selected_device.type == "cuda":
        model_options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["visual", "lm_head"])
        model_options["device_map"] = {"": selected_device.index or 0}
    # B615: local directory only (local_files_only=True), never downloaded.
    vision_model = AutoModelForImageTextToText.from_pretrained(model_directory, **model_options).eval()  # nosec B615
    if selected_device.type == "cpu":
        vision_model = vision_model.to(selected_device)
    model_processor = AutoProcessor.from_pretrained(model_directory, local_files_only=True)  # nosec B615
    return LoadedJudge(vision_model, model_processor, answer_token_ids(model_processor))


@torch.inference_mode()
def error_probability(loaded_judge: LoadedJudge, line_image: Image.Image, line_text: str, target_text: str) -> float:
    """Probability that the word is not printed as read.

    Usage:
        Called once per judge item.

    Args:
        loaded_judge: Model and processor.
        line_image: RGB line crop.
        line_text: Text read for the line.
        target_text: Unit in question.

    Returns:
        ``P(no) / (P(yes) + P(no))`` of the next token, 0..1.
    """
    line_image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    chat_messages = [{"role": "user", "content": [
        {"type": "image", "image": line_image},
        {"type": "text", "text": QUESTION_TEMPLATE.format(line_text=line_text, target_text=target_text)}]}]
    model_inputs = loaded_judge.model_processor.apply_chat_template(
        chat_messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        enable_thinking=False).to(loaded_judge.vision_model.device)
    next_logits = loaded_judge.vision_model(**model_inputs).logits[0, -1].float()
    consent_ids, refusal_ids = loaded_judge.answer_ids
    consent_mass = torch.logsumexp(next_logits[list(consent_ids)], dim=0)
    refusal_mass = torch.logsumexp(next_logits[list(refusal_ids)], dim=0)
    return float(torch.sigmoid(refusal_mass - consent_mass))
