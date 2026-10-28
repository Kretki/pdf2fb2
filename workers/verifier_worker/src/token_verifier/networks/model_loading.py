"""Create, save and load the verifier from local directories (never from the network).

A fine-tuned verifier directory (``models/token_verifier``) holds:

* ``backbone/`` — the fine-tuned encoder (and tokenizer for sub-word backbones);
* ``verifier_head.safetensors`` — pooling/evidence/classifier weights;
* ``verifier_config.json`` — backbone kind, evidence widths, evidence model directories, window settings;
* ``thresholds.json`` — decision thresholds tuned on the validation split.

An ensemble directory instead holds ``ensemble.json`` listing member sub-directories in that format; their
probabilities are averaged. Evidence providers (language-model scorer, crop embedder) are loaded once per bundle.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from token_verifier.features.feature_assembly import FeatureAssembler
from token_verifier.features.language_model_scores import LanguageModelScorer
from token_verifier.features.visual_embeddings import VisualEmbedder
from token_verifier.networks.backbone_adapters import BackboneAdapter, load_backbone
from token_verifier.networks.verifier_network import VerifierNetwork

THRESHOLDS_FILE = "thresholds.json"
CONFIG_FILE = "verifier_config.json"
HEAD_FILE = "verifier_head.safetensors"
ENSEMBLE_FILE = "ensemble.json"
BACKBONE_DIRECTORY = "backbone"
DEFAULT_THRESHOLDS = {"misread": 0.5, "inserted": 0.5}


@dataclass(frozen=True)
class EvidenceSources:
    """Where the evidence providers come from.

    Attributes:
        language_model_directory: Masked-LM checkpoint for plausibility features, or None.
        visual_model_directory: Vision checkpoint for crop embeddings, or None.
    """

    language_model_directory: Path | None
    visual_model_directory: Path | None


@dataclass(frozen=True)
class WindowShape:
    """Window settings a verifier is trained and used with.

    Attributes:
        window_units: Units per window.
        max_tokens: Piece limit per window.
        context_units: Unlabelled context units per side.
    """

    window_units: int
    max_tokens: int
    context_units: int


@dataclass
class LoadedVerifier:
    """One network ready for training or prediction.

    Attributes:
        backbone_adapter: Encoder and its input encoding.
        verifier_network: Head (with the encoder as submodule).
        torch_device: Device of the network.
        backbone_name: ``subword`` or ``byte``.
    """

    backbone_adapter: BackboneAdapter
    verifier_network: VerifierNetwork
    torch_device: torch.device
    backbone_name: str


@dataclass
class VerifierBundle:
    """Members, shared evidence providers and window settings.

    Attributes:
        verifier_members: One or more networks (probabilities are averaged).
        feature_assembler: Evidence providers.
        window_shape: Window settings.
    """

    verifier_members: list[LoadedVerifier]
    feature_assembler: FeatureAssembler
    window_shape: WindowShape


def torch_device_of(device_name: str) -> torch.device:
    """Map ``gpu:N`` / ``cuda:N`` / ``cpu`` to a torch device (CPU when CUDA is unavailable).

    Args:
        device_name: Device text.

    Returns:
        Torch device.
    """
    cuda_name = device_name.replace("gpu", "cuda")
    return torch.device(cuda_name if cuda_name.startswith("cuda") and torch.cuda.is_available() else "cpu")


def build_assembler(evidence_sources: EvidenceSources, torch_device: torch.device, evidence_dropout: float = 0.0,
                    random_seed: int = 0) -> FeatureAssembler:
    """Load the configured evidence providers.

    Args:
        evidence_sources: Provider directories.
        torch_device: Device.
        evidence_dropout: Training-time evidence dropout.
        random_seed: Seed of the dropout.

    Returns:
        The assembler.
    """
    language_scorer = None if evidence_sources.language_model_directory is None else LanguageModelScorer(
        evidence_sources.language_model_directory, torch_device)
    visual_embedder = None if evidence_sources.visual_model_directory is None else VisualEmbedder(
        evidence_sources.visual_model_directory, torch_device)
    return FeatureAssembler(language_scorer, visual_embedder, evidence_dropout, random_seed)


def create_verifier(base_model: Path, torch_device: torch.device, feature_assembler: FeatureAssembler,
                    backbone_name: str = "auto") -> LoadedVerifier:
    """Pretrained backbone with a fresh head (start of fine-tuning).

    Args:
        base_model: Local pretrained encoder (``models/xlm-roberta-base``, ``models/mdeberta-v3-base``,
            ``models/byt5-small``).
        torch_device: Device.
        feature_assembler: Defines the evidence widths.
        backbone_name: ``auto``, ``subword`` or ``byte``.

    Returns:
        The verifier in eval mode.
    """
    backbone_adapter, resolved_name = load_backbone(base_model, backbone_name)
    verifier_network = VerifierNetwork(backbone_adapter.encoder_module, backbone_adapter.hidden_size,
                                       feature_assembler.feature_width, feature_assembler.visual_width)
    return LoadedVerifier(backbone_adapter, verifier_network.to(torch_device).eval(), torch_device, resolved_name)


def stored_path(path_value: Path | None) -> str | None:
    """Path text for the config file.

    Args:
        path_value: Directory or None.

    Returns:
        Absolute path text or None.
    """
    return None if path_value is None else str(path_value.resolve())


def save_verifier(loaded_verifier: LoadedVerifier, output_directory: Path, evidence_sources: EvidenceSources,
                  window_shape: WindowShape) -> None:
    """Write one fine-tuned verifier (thresholds are written separately).

    Args:
        loaded_verifier: Network to save.
        output_directory: Target directory.
        evidence_sources: Provider directories used in training.
        window_shape: Window settings used in training.

    Returns:
        None.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    loaded_verifier.backbone_adapter.save_backbone(output_directory / BACKBONE_DIRECTORY)
    head_state = {parameter_name: parameter_value.detach().cpu().contiguous()
                  for parameter_name, parameter_value in loaded_verifier.verifier_network.head_state().items()}
    save_file(head_state, str(output_directory / HEAD_FILE))
    config_values = {"backbone_kind": loaded_verifier.backbone_name,
                     "feature_width": loaded_verifier.verifier_network.feature_width,
                     "visual_width": loaded_verifier.verifier_network.visual_width,
                     "language_model_directory": stored_path(evidence_sources.language_model_directory),
                     "visual_model_directory": stored_path(evidence_sources.visual_model_directory),
                     "window_units": window_shape.window_units, "max_tokens": window_shape.max_tokens,
                     "context_units": window_shape.context_units}
    (output_directory / CONFIG_FILE).write_text(json.dumps(config_values, indent=1), encoding="utf-8")


def read_config(model_directory: Path) -> dict:
    """Configuration of a saved verifier.

    Args:
        model_directory: Fine-tuned verifier directory.

    Returns:
        Decoded ``verifier_config.json``.

    Raises:
        OSError: If the directory holds no verifier in this format (retrain with ``token_verifier train``).
    """
    config_path = model_directory / CONFIG_FILE
    if not config_path.exists():
        raise OSError(f"{model_directory} holds no {CONFIG_FILE}; train the verifier with this version.")
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_member(model_directory: Path, torch_device: torch.device) -> LoadedVerifier:
    """Load one saved verifier network.

    Args:
        model_directory: Member directory.
        torch_device: Device.

    Returns:
        The verifier in eval mode.

    Raises:
        OSError: If files are missing.
    """
    config_values = read_config(model_directory)
    backbone_adapter, backbone_name = load_backbone(model_directory / BACKBONE_DIRECTORY,
                                                    config_values["backbone_kind"])
    verifier_network = VerifierNetwork(backbone_adapter.encoder_module, backbone_adapter.hidden_size,
                                       int(config_values["feature_width"]), int(config_values["visual_width"]))
    verifier_network.load_state_dict(load_file(str(model_directory / HEAD_FILE)), strict=False)
    return LoadedVerifier(backbone_adapter, verifier_network.to(torch_device).eval(), torch_device, backbone_name)


def member_directories(model_directory: Path) -> list[Path]:
    """Directories of the members of a verifier (itself when it is not an ensemble).

    Args:
        model_directory: Verifier directory.

    Returns:
        Member directories.

    Raises:
        ValueError: If ``ensemble.json`` names a directory outside the verifier directory.
    """
    ensemble_path = model_directory / ENSEMBLE_FILE
    if not ensemble_path.exists():
        return [model_directory]
    member_names = json.loads(ensemble_path.read_text(encoding="utf-8"))["members"]
    if any("/" in member_name or member_name.startswith(".") for member_name in member_names):
        raise ValueError(f"{ensemble_path} must list sub-directory names only.")
    return [model_directory / member_name for member_name in member_names]


def load_bundle(model_directory: Path, device_name: str) -> VerifierBundle:
    """Load a fine-tuned verifier (or ensemble) with its evidence providers.

    Usage:
        Called by pipeline jobs, ``tune`` and ``eval``.

    Args:
        model_directory: ``models/token_verifier``.
        device_name: Device text.

    Returns:
        The bundle.

    Raises:
        OSError: If a member is missing or in an old format.
    """
    torch_device = torch_device_of(device_name)
    member_paths = member_directories(model_directory)
    config_values = read_config(member_paths[0])
    evidence_sources = EvidenceSources(
        *(None if config_values.get(key_name) is None else Path(config_values[key_name])
          for key_name in ("language_model_directory", "visual_model_directory")))
    window_shape = WindowShape(int(config_values["window_units"]), int(config_values["max_tokens"]),
                               int(config_values.get("context_units", 0)))
    return VerifierBundle([load_member(member_path, torch_device) for member_path in member_paths],
                          build_assembler(evidence_sources, torch_device), window_shape)


def write_ensemble(model_directory: Path, member_names: list[str]) -> None:
    """Record the members of an ensemble.

    Args:
        model_directory: Verifier directory.
        member_names: Sub-directory names.

    Returns:
        None.
    """
    (model_directory / ENSEMBLE_FILE).write_text(json.dumps({"members": member_names}, indent=1), encoding="utf-8")


def write_thresholds(model_directory: Path, class_thresholds: dict[str, float]) -> None:
    """Store decision thresholds.

    Args:
        model_directory: Verifier directory.
        class_thresholds: ``{"misread": float, "inserted": float}``.

    Returns:
        None.
    """
    (model_directory / THRESHOLDS_FILE).write_text(json.dumps(class_thresholds, indent=1), encoding="utf-8")


def read_thresholds(model_directory: Path) -> dict[str, float]:
    """Decision thresholds tuned on the validation split.

    Args:
        model_directory: Fine-tuned verifier directory.

    Returns:
        ``{"misread": float, "inserted": float}`` (0.5 each when the file is missing).
    """
    thresholds_path = model_directory / THRESHOLDS_FILE
    if not thresholds_path.exists():
        return dict(DEFAULT_THRESHOLDS)
    stored_values = json.loads(thresholds_path.read_text(encoding="utf-8"))
    return {class_name: float(stored_values.get(class_name, default_value))
            for class_name, default_value in DEFAULT_THRESHOLDS.items()}
