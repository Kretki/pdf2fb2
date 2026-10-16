"""Pydantic configuration of the reader-mark pipeline, loaded from ``default_settings.yaml``.

Usage:
    settings = load_settings()               # defaults
    settings = load_settings(Path("my.yaml"))  # defaults overridden by a user file
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_SETTINGS_FILE = Path(__file__).with_name("default_settings.yaml")
MAX_SETTINGS_BYTES = 1024 * 1024


class StrictModel(BaseModel):
    """Base for all setting groups: unknown keys are rejected, values are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PathSettings(StrictModel):
    """Filesystem locations; relative paths are resolved against the repository root."""

    backbone_directory: Path = Path("models/dinov2-small")
    encoder_directory: Path = Path("models/timm/convnext_tiny.fb_in22k_ft_in1k")
    checkpoint_directory: Path = Path("models/readermarks")
    data_directory: Path = Path("readermarks_data")
    real_pages_directory: Path = Path("test_res/pages")
    layout_segments_file: Path | None = Path("test_res/segments.json")

    def resolved(self, relative_path: Path) -> Path:
        """Resolve a configured path against the repository root.

        Usage:
            Call for every path read from this group before using it.

        Args:
            relative_path: One of the fields of this group.

        Returns:
            Absolute path.
        """
        return relative_path if relative_path.is_absolute() else REPOSITORY_ROOT / relative_path


class SpotterSettings(StrictModel):
    """Stage 1: frozen ViT backbone + small convolutional head over the patch grid."""

    input_long_side: int = Field(1344, ge=224, le=2688)
    patch_size: int = Field(14, ge=8, le=32)
    hidden_channels: int = Field(256, ge=32, le=1024)
    convolution_layers: int = Field(3, ge=1, le=6)
    epoch_count: int = Field(40, ge=1, le=500)
    batch_size: int = Field(16, ge=1, le=256)
    learning_rate: float = Field(1e-3, gt=0, lt=1)
    weight_decay: float = Field(1e-4, ge=0, lt=1)
    focal_gamma: float = Field(2.0, ge=0, le=5)
    focal_weight: float = Field(1.0, ge=0, le=10)
    page_loss_weight: float = Field(0.5, ge=0, le=10)
    patch_label_fraction: float = Field(0.02, gt=0, lt=1)
    maximum_positive_weight: float = Field(20.0, ge=1, le=1000)


class SegmenterSettings(StrictModel):
    """Stage 2: U-Net with an ImageNet ConvNeXt-Tiny encoder on native-resolution crops."""

    encoder_name: str = "tu-convnext_tiny"
    crop_size: int = Field(768, ge=128, le=2048)
    step_count: int = Field(6000, ge=1, le=200000)
    batch_size: int = Field(6, ge=1, le=64)
    learning_rate: float = Field(3e-4, gt=0, lt=1)
    weight_decay: float = Field(1e-4, ge=0, lt=1)
    mark_crop_fraction: float = Field(0.8, ge=0, le=1)
    validation_interval: int = Field(500, ge=1, le=100000)
    loader_workers: int = Field(8, ge=0, le=64)


class SynthesisSettings(StrictModel):
    """Synthetic page generator (STAGE_CLEAN.md §7)."""

    random_seed: int = 20261016
    train_count: int = Field(2400, ge=0, le=100000)
    validation_count: int = Field(200, ge=0, le=100000)
    test_count: int = Field(200, ge=0, le=100000)
    unmarked_fraction: float = Field(0.25, ge=0, le=1)
    rule_page_fraction: float = Field(0.30, ge=0, le=1)
    minimum_marks: int = Field(1, ge=1, le=50)
    maximum_marks: int = Field(9, ge=1, le=80)
    synthetic_rule_probability: float = Field(0.5, ge=0, le=1)
    picture_page_fraction: float = Field(0.2, ge=0, le=1)
    text_copy_page_fraction: float = Field(0.3, ge=0, le=1)
    preview_count: int = Field(50, ge=1, le=500)


class CleaningSettings(StrictModel):
    """Inference thresholds and deterministic post-processing (overridden by ``thresholds.json``)."""

    page_threshold: float = Field(0.5, ge=0, le=1)
    patch_threshold: float = Field(0.5, ge=0, le=1)
    minimum_flagged_patches: int = Field(2, ge=1, le=10000)
    region_dilation_patches: int = Field(2, ge=0, le=10)
    pixel_threshold: float = Field(0.5, ge=0, le=1)
    ink_threshold: float = Field(0.5, ge=0, le=1)
    minimum_mark_pixels: int = Field(120, ge=1, le=1000000)
    minimum_component_pixels: int = Field(25, ge=1, le=100000)
    mark_dilation_pixels: int = Field(2, ge=0, le=15)
    background_tile_pixels: int = Field(24, ge=8, le=256)
    rule_maximum_residual: float = Field(0.8, ge=0, le=10)
    rule_maximum_thickness_variation: float = Field(0.18, ge=0, le=5)
    rule_maximum_slope_degrees: float = Field(0.6, ge=0, le=45)
    rule_minimum_length: int = Field(40, ge=1, le=10000)
    hand_minimum_overshoot: int = Field(6, ge=0, le=500)


class ReaderMarkSettings(StrictModel):
    """All configuration groups of the reader-mark pipeline."""

    path_settings: PathSettings = PathSettings()
    spotter_settings: SpotterSettings = SpotterSettings()
    segmenter: SegmenterSettings = SegmenterSettings()
    synthesis: SynthesisSettings = SynthesisSettings()
    cleaning: CleaningSettings = CleaningSettings()


def read_yaml_mapping(settings_file: Path) -> dict:
    """Read a bounded YAML file that must contain a mapping.

    Usage:
        Called by ``load_settings`` for the defaults and the optional user file.

    Args:
        settings_file: YAML file path.

    Returns:
        The decoded mapping (empty for an empty file).

    Raises:
        ValueError: If the file is too large or is not a mapping.
        OSError: If the file cannot be read.
    """
    if settings_file.stat().st_size > MAX_SETTINGS_BYTES:
        raise ValueError(f"{settings_file} is larger than {MAX_SETTINGS_BYTES} bytes.")
    decoded_mapping = yaml.safe_load(settings_file.read_text(encoding="utf-8")) or {}
    if not isinstance(decoded_mapping, dict):
        raise ValueError(f"{settings_file} must contain a YAML mapping.")
    return decoded_mapping


def merge_mappings(base_mapping: dict, override_mapping: dict) -> dict:
    """Recursively merge two setting mappings, the override winning.

    Args:
        base_mapping: Default values.
        override_mapping: Values that replace defaults (nested groups are merged).

    Returns:
        A new merged mapping; the inputs are not mutated.
    """
    merged_mapping = dict(base_mapping)
    for setting_key, setting_value in override_mapping.items():
        if isinstance(setting_value, dict) and isinstance(merged_mapping.get(setting_key), dict):
            merged_mapping[setting_key] = merge_mappings(merged_mapping[setting_key], setting_value)
        else:
            merged_mapping[setting_key] = setting_value
    return merged_mapping


def load_settings(override_file: Path | None = None) -> ReaderMarkSettings:
    """Load the default settings, optionally overridden by a user YAML file.

    Usage:
        Call once in every command and pass the result down (dependency injection).

    Args:
        override_file: Optional YAML file with a subset of the groups/keys.

    Returns:
        Validated, immutable settings.

    Raises:
        pydantic.ValidationError: If a value is out of range or a key is unknown.
        ValueError, OSError: If a file cannot be read.
    """
    settings_mapping = read_yaml_mapping(DEFAULT_SETTINGS_FILE)
    if override_file is not None:
        settings_mapping = merge_mappings(settings_mapping, read_yaml_mapping(override_file))
    return ReaderMarkSettings.model_validate(settings_mapping)
