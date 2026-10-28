"""Image embedding of every unit's word crop (image-and-text verification).

A small self-supervised vision encoder (DINOv2, ``models/dinov2-small``) embeds the crop of the word a unit was
read from; the verifier learns whether the text fits the ink (a plausible word read from a different shape is a
misreading no text model can see). Units without a crop get zeros and a cleared presence flag.
"""

from pathlib import Path

import numpy
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

EMBED_BATCH = 64
MAX_CROP_PIXELS = 4_000_000


class VisualEmbedder:
    """Vision encoder for word crops.

    Usage:
        Built by ``feature_assembly`` when ``visual_model_directory`` is set.
    """

    def __init__(self, model_directory: Path, torch_device: torch.device) -> None:
        """Load processor and encoder from local files.

        Args:
            model_directory: Local vision checkpoint.
            torch_device: Device.

        Raises:
            OSError: If the directory does not hold a model.
        """
        # B615: local directory only (local_files_only=True), never downloaded.
        self.image_processor = AutoImageProcessor.from_pretrained(str(model_directory),  # nosec B615
                                                                  local_files_only=True)
        self.vision_model = AutoModel.from_pretrained(str(model_directory),  # nosec B615
                                                      local_files_only=True).to(torch_device).eval()
        self.torch_device = torch_device
        self.embedding_width = int(self.vision_model.config.hidden_size)

    @property
    def visual_width(self) -> int:
        """Width of a unit's visual row (embedding plus presence flag).

        Returns:
            ``embedding_width + 1``.
        """
        return self.embedding_width + 1

    @torch.inference_mode()
    def _embed_batch(self, crop_paths: list[Path]) -> numpy.ndarray:
        """CLS embeddings of a batch of crops.

        Args:
            crop_paths: Existing crop images.

        Returns:
            ``(crops, embedding_width)`` float32 matrix.
        """
        crop_images = []
        for crop_path in crop_paths:
            with Image.open(crop_path) as crop_file:
                if crop_file.width * crop_file.height > MAX_CROP_PIXELS:
                    crop_file.thumbnail((2000, 2000))
                crop_images.append(crop_file.convert("RGB"))
        model_inputs = self.image_processor(images=crop_images, return_tensors="pt").to(self.torch_device)
        return self.vision_model(**model_inputs).last_hidden_state[:, 0].float().cpu().numpy()

    def crop_rows(self, crop_paths: tuple[Path | None, ...], unit_count: int) -> numpy.ndarray:
        """Visual rows of the units of one text.

        Usage:
            Called once per text by ``feature_assembly``.

        Args:
            crop_paths: Crop per unit or None (may be empty).
            unit_count: Units of the text.

        Returns:
            ``(units, visual_width)`` float32 matrix.
        """
        visual_matrix = numpy.zeros((unit_count, self.visual_width), dtype=numpy.float32)
        present_units = [unit_index for unit_index, crop_path in enumerate(crop_paths)
                         if crop_path is not None and crop_path.is_file()]
        for batch_start in range(0, len(present_units), EMBED_BATCH):
            batch_units = present_units[batch_start:batch_start + EMBED_BATCH]
            embeddings = self._embed_batch([crop_paths[unit_index] for unit_index in batch_units])
            visual_matrix[batch_units, :-1] = embeddings
            visual_matrix[batch_units, -1] = 1.0
        return visual_matrix
