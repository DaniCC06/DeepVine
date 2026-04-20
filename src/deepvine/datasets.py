from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def build_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Build aggressive train-time augmentation transforms.

    This pipeline is designed for large source images (e.g., 4000x3000) and
    scarce classes, improving generalization by adding geometric and photometric
    perturbations while preserving ImageNet normalization.

    Args:
        image_size: Final square size expected by the backbone.

    Returns:
        A torchvision Compose object for training.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=image_size,
                scale=(0.55, 1.0),
                ratio=(0.75, 1.33),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=35, interpolation=InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Build deterministic evaluation transforms.

    Args:
        image_size: Final square size expected by the backbone.

    Returns:
        A torchvision Compose object for validation and inference.
    """
    resize_size = int(image_size * 1.14)
    return transforms.Compose(
        [
            transforms.Resize(
                size=resize_size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.CenterCrop(size=image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class VineLeafDataset(Dataset[tuple[Tensor, int]]):
    """Dataset for ampelographic grapevine leaf classification.

    Expects the canonical image-folder structure:

    dataset/
        class_a/
            image_1.jpg
            ...
        class_b/
            image_2.jpg
            ...

    Args:
        root_dir: Root dataset directory containing one subfolder per class.
        transform: Callable transform applied to each PIL image.

    Raises:
        FileNotFoundError: If root directory does not exist.
        ValueError: If no class folders or image files are found.
    """

    def __init__(self, root_dir: str | Path, transform: Callable[[Image.Image], Tensor] | None = None) -> None:
        self.root_dir: Path = Path(root_dir)
        if not self.root_dir.exists():
            msg = f"Dataset directory does not exist: {self.root_dir}"
            raise FileNotFoundError(msg)

        class_dirs = sorted(path for path in self.root_dir.iterdir() if path.is_dir())
        if not class_dirs:
            msg = f"No class folders found in: {self.root_dir}"
            raise ValueError(msg)

        self.class_to_idx: dict[str, int] = {class_dir.name: idx for idx, class_dir in enumerate(class_dirs)}
        self.idx_to_class: dict[int, str] = {idx: name for name, idx in self.class_to_idx.items()}
        self.samples: list[tuple[Path, int]] = []

        for class_name, class_idx in self.class_to_idx.items():
            class_path = self.root_dir / class_name
            for image_path in sorted(class_path.rglob("*")):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS and image_path.is_file():
                    self.samples.append((image_path, class_idx))

        if not self.samples:
            msg = f"No images found in dataset directory: {self.root_dir}"
            raise ValueError(msg)

        self.transform: Callable[[Image.Image], Tensor] = transform or build_train_transforms()

    def __len__(self) -> int:
        """Return number of samples available in the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """Return transformed image tensor and class index.

        Args:
            index: Dataset sample index.

        Returns:
            A tuple containing image tensor and integer class label.
        """
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
        tensor = self.transform(image_rgb)
        return tensor, label
