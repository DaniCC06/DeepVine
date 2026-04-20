from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader

from .datasets import VineLeafDataset


def create_dataloader(
    root_dir: str | Path,
    transform: Callable[[Image.Image], Tensor],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> DataLoader[tuple[Tensor, int]]:
    """Create optimized DataLoader for vine leaf images.

    Args:
        root_dir: Dataset root path.
        transform: Transform pipeline to apply in dataset.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle samples every epoch.
        num_workers: Number of subprocesses for loading.
        pin_memory: Whether to pin host memory for faster GPU transfers.
        persistent_workers: Keep workers alive between epochs.

    Returns:
        Configured PyTorch DataLoader.

    Raises:
        ValueError: If batch_size or num_workers are invalid.
    """
    if batch_size <= 0:
        msg = "batch_size must be > 0."
        raise ValueError(msg)

    if num_workers < 0:
        msg = "num_workers must be >= 0."
        raise ValueError(msg)

    dataset = VineLeafDataset(root_dir=root_dir, transform=transform)
    use_persistent_workers = persistent_workers and num_workers > 0

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent_workers,
    )
