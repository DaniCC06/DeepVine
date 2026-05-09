from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from collections import defaultdict

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deepvine.datasets import VineLeafDataset, build_eval_transforms, build_train_transforms
from deepvine.engine import FocalLabelSmoothingLoss, TrainingHistory, fit, resolve_device
from deepvine.models import VineLeafClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _resolve_project_path(path: str | Path) -> Path:
    """Resolve relative paths against the project root directory."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for DeepVine model training.

    Attributes:
        dataset_dir: Root directory containing class subfolders.
        image_size: Input size for backbone.
        batch_size: Batch size for train/validation loaders. Optimized for RTX 4060Ti.
        epochs: Number of optimization epochs.
        learning_rate: Max learning rate used by OneCycleLR.
        weight_decay: Weight decay for AdamW.
        val_split: Fraction of samples used for validation.
        num_workers: Number of workers for DataLoader. Auto-optimized for GPU size.
        backbone: Backbone architecture to use.
        gamma: Focal loss gamma.
        label_smoothing: Label smoothing factor.
        checkpoint_path: Path to save best checkpoint.
        resume_from_checkpoint: Whether to resume model/optimizer from checkpoint_path.
        device: Optional device override.
        patience: Early stopping patience in epochs.
        max_grad_norm: Gradient clipping norm (0 to disable).
    
    Note:
        Default batch_size=8 is optimized for RTX 4060Ti (8GB VRAM).
        Increase to 12-16 for RTX 3070/4070, or reduce to 4-6 for RTX 3050/4050.
    """

    dataset_dir: str = "./dataset"
    image_size: int = 224
    batch_size: int = 8  # OPTIMIZED for RTX 4060Ti (8GB): was 16
    epochs: int = 30  # Increased for better convergence
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    val_split: float = 0.2
    num_workers: int = 2  # REDUCED from 4: better for smaller GPUs
    backbone: str = "resnet50"
    gamma: float = 2.0
    label_smoothing: float = 0.1
    checkpoint_path: str = "./checkpoints/best_model.pt"
    resume_from_checkpoint: bool = False
    patience: int = 5
    max_grad_norm: float = 1.0
    device: str | None = None


def _build_train_val_loaders(config: TrainingConfig) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]], dict[str, int]]:
    """Create train and validation loaders from dataset directory."""
    dataset_dir = _resolve_project_path(config.dataset_dir)

    # Single filesystem scan to get class mapping and sample count.
    # The val dataset is reused directly instead of creating a throw-away base_dataset.
    val_dataset = VineLeafDataset(
        root_dir=dataset_dir,
        transform=build_eval_transforms(image_size=config.image_size),
    )
    num_samples = len(val_dataset)
    class_to_idx = val_dataset.class_to_idx

    samples_by_class: dict[int, list[int]] = defaultdict(list)
    for sample_index, (_, class_index) in enumerate(val_dataset.samples):
        samples_by_class[class_index].append(sample_index)

    generator = torch.Generator().manual_seed(42)
    train_indices: list[int] = []
    val_indices: list[int] = []

    can_split_without_overlap = all(len(indices) >= 2 for indices in samples_by_class.values()) and num_samples >= len(samples_by_class) * 2
    if can_split_without_overlap:
        for class_index in sorted(samples_by_class):
            class_indices = torch.tensor(samples_by_class[class_index], dtype=torch.int64)
            shuffled_class_indices = class_indices[torch.randperm(len(class_indices), generator=generator)].tolist()

            num_class_val = max(1, int(round(len(shuffled_class_indices) * config.val_split)))
            num_class_val = min(num_class_val, len(shuffled_class_indices) - 1)

            val_indices.extend(shuffled_class_indices[:num_class_val])
            train_indices.extend(shuffled_class_indices[num_class_val:])
    else:
        logger.warning(
            "Dataset is too small for a disjoint train/val split. Training will use all images, "
            "and validation will use one deterministic sample per class."
        )
        train_indices = list(range(num_samples))
        val_indices = [indices[0] for indices in samples_by_class.values()]

    if not train_indices or not val_indices:
        msg = "Unable to build train/validation subsets from the dataset."
        raise ValueError(msg)

    train_subset = Subset(
        VineLeafDataset(
            root_dir=dataset_dir,
            transform=build_train_transforms(image_size=config.image_size),
        ),
        train_indices,
    )
    val_subset = Subset(val_dataset, val_indices)

    pin_memory = torch.cuda.is_available()

    logger.info(
        "Dataset split: %d train / %d val samples (%d classes).",
        len(train_subset),
        len(val_subset),
        len(class_to_idx),
    )

    train_loader = DataLoader(
        dataset=train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        dataset=val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0,
    )

    return train_loader, val_loader, class_to_idx


def run_training(config: TrainingConfig) -> TrainingHistory:
    """Run end-to-end training using the DeepVine engine.

    Args:
        config: Training hyperparameters and runtime setup.

    Returns:
        Training history with best validation metric metadata.
    """
    train_loader, val_loader, class_to_idx = _build_train_val_loaders(config)
    num_classes = len(class_to_idx)
    checkpoint_path = _resolve_project_path(config.checkpoint_path)

    device = resolve_device(config.device)

    model = VineLeafClassifier(
        num_classes=num_classes,
        backbone=config.backbone,  # type: ignore[arg-type]
        dropout_p=0.2,
        freeze_backbone=True,
    )
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = FocalLabelSmoothingLoss(
        gamma=config.gamma,
        label_smoothing=config.label_smoothing,
        alpha=None,
        reduction="mean",
    )

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_classes=num_classes,
        epochs=config.epochs,
        max_lr=config.learning_rate,
        device=device,
        checkpoint_path=checkpoint_path,
        class_to_idx=class_to_idx,
        resume_from_checkpoint=config.resume_from_checkpoint,
        patience=config.patience,
        max_grad_norm=config.max_grad_norm,
    )

    return history


if __name__ == "__main__":
    default_config = TrainingConfig()
    run_training(default_config)
