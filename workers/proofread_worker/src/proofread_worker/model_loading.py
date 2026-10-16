"""Load the local Qwen3.5 language model (text part only) for scoring, 4-bit on CUDA."""

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerBase


@dataclass(frozen=True)
class LoadedLanguageModel:
    """Tokenizer plus the decoder and its output projection.

    Attributes:
        tokenizer: Fast tokenizer (offset mapping is required).
        decoder_model: Transformer that returns hidden states.
        output_head: Projection of hidden states to vocabulary logits.
        torch_device: Device of the inputs.
    """

    tokenizer: PreTrainedTokenizerBase
    decoder_model: torch.nn.Module
    output_head: torch.nn.Module
    torch_device: torch.device


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


def load_language_model(model_directory: Path, device_name: str) -> LoadedLanguageModel:
    """Load the tokenizer and the causal language model from local files only.

    Usage:
        Called once per job. On CUDA the weights are quantised to 4-bit NF4 while loading
        (about 7.7 GB of VRAM for the 9B model); on CPU they stay in bfloat16.

    Args:
        model_directory: Directory with ``config.json``, tokenizer files and safetensors shards.
        device_name: pdf2fb2 device name.

    Returns:
        The loaded model in eval mode.

    Raises:
        OSError: If the files are missing.
    """
    selected_device = torch_device(device_name)
    # B615: a local directory with local_files_only=True; nothing is downloaded, so no revision to pin.
    tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True)  # nosec B615
    model_options: dict = {"dtype": torch.bfloat16, "local_files_only": True}
    if selected_device.type == "cuda":
        model_options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
        model_options["device_map"] = {"": selected_device.index or 0}
    causal_model = AutoModelForCausalLM.from_pretrained(  # nosec B615 - local directory, local_files_only=True
        model_directory, **model_options).eval()
    if selected_device.type == "cpu":
        causal_model = causal_model.to(selected_device)
    return LoadedLanguageModel(tokenizer=tokenizer, decoder_model=causal_model.model,
                               output_head=causal_model.lm_head, torch_device=selected_device)
