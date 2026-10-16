"""Torch dataset over cached patch features + labels, with padding collation for mixed grid sizes."""

from pathlib import Path

import numpy
import torch
from torch.utils.data import Dataset

from readermarks.networks.feature_cache import FEATURES_SUFFIX, LABELS_SUFFIX
from readermarks.settings.class_catalog import MARK_CLASS_COUNT


class CachedSpotterDataset(Dataset):
    """Cached (features, patch labels, page label) triples of one split.

    Usage:
        ``CachedSpotterDataset(cache_directory)`` with ``collate_padded`` as ``collate_fn``.
    """

    def __init__(self, cache_directory: Path, horizontal_flip: bool = False) -> None:
        """Index the cached samples that have labels.

        Args:
            cache_directory: Directory written by ``cache_items``.
            horizontal_flip: Randomly mirror the grid (training augmentation).
        """
        self._feature_paths = sorted(
            feature_path
            for feature_path in cache_directory.glob(f"*{FEATURES_SUFFIX}")
            if feature_path.with_name(feature_path.name.replace(FEATURES_SUFFIX, LABELS_SUFFIX)).is_file()
        )
        self._horizontal_flip = horizontal_flip

    def __len__(self) -> int:
        """Return the number of samples.

        Returns:
            Sample count.
        """
        return len(self._feature_paths)

    def __getitem__(self, sample_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load one sample.

        Args:
            sample_index: 0..len-1.

        Returns:
            ``(features (D, r, c) float32, labels (C, r, c) float32, page_label () float32)``.
        """
        feature_path = self._feature_paths[sample_index]
        patch_features = numpy.load(feature_path, allow_pickle=False).astype(numpy.float32)
        patch_labels = numpy.load(
            feature_path.with_name(feature_path.name.replace(FEATURES_SUFFIX, LABELS_SUFFIX)), allow_pickle=False
        ).astype(numpy.float32)
        if self._horizontal_flip and numpy.random.random() < 0.5:  # nosec B311 - augmentation only
            patch_features, patch_labels = patch_features[:, :, ::-1].copy(), patch_labels[:, :, ::-1].copy()
        page_label = numpy.float32(patch_labels[:MARK_CLASS_COUNT].any())
        return torch.from_numpy(patch_features), torch.from_numpy(patch_labels), torch.tensor(page_label)


def collate_padded(sample_batch: list[tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, ...]:
    """Pad grids of different sizes to the largest one in the batch.

    Args:
        sample_batch: Items from ``CachedSpotterDataset``.

    Returns:
        ``(features (B, D, R, C), labels (B, K, R, C), valid_mask (B, 1, R, C), page_labels (B,))``.
    """
    maximum_rows = max(features.shape[1] for features, _, _ in sample_batch)
    maximum_columns = max(features.shape[2] for features, _, _ in sample_batch)
    batch_size = len(sample_batch)
    feature_batch = torch.zeros(batch_size, sample_batch[0][0].shape[0], maximum_rows, maximum_columns)
    label_batch = torch.zeros(batch_size, sample_batch[0][1].shape[0], maximum_rows, maximum_columns)
    valid_mask = torch.zeros(batch_size, 1, maximum_rows, maximum_columns)
    for batch_index, (patch_features, patch_labels, _) in enumerate(sample_batch):
        grid_rows, grid_columns = patch_features.shape[1:]
        feature_batch[batch_index, :, :grid_rows, :grid_columns] = patch_features
        label_batch[batch_index, :, :grid_rows, :grid_columns] = patch_labels
        valid_mask[batch_index, :, :grid_rows, :grid_columns] = 1.0
    page_labels = torch.stack([page_label for _, _, page_label in sample_batch])
    return feature_batch, label_batch, valid_mask, page_labels
