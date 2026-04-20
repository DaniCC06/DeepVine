from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deepvine.datasets import VineLeafDataset, build_eval_transforms, build_train_transforms
from deepvine.engine import FocalLabelSmoothingLoss, TrainingHistory, fit, resolve_device
from deepvine.models import VineLeafClassifier


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for DeepVine model training.

    Attributes:
        dataset_dir: Root directory containing class subfolders.
        image_size: Input size for backbone.
        batch_size: Batch size for train/validation loaders.
        epochs: Number of optimization epochs.
        learning_rate: Max learning rate used by OneCycleLR.
        weight_decay: Weight decay for AdamW.
        val_split: Fraction of samples used for validation.
        num_workers: Number of workers for DataLoader.
        backbone: Backbone architecture to use.
        gamma: Focal loss gamma.
        label_smoothing: Label smoothing factor.
        checkpoint_path: Path to save best checkpoint.
        device: Optional device override.
    """

    dataset_dir: str = "./dataset"
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 25
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    val_split: float = 0.2
    num_workers: int = 4
    backbone: str = "resnet50"
    gamma: float = 2.0
    label_smoothing: float = 0.1
    checkpoint_path: str = "./checkpoints/best_model.pt"
    device: str | None = None


def _build_train_val_loaders(config: TrainingConfig) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]], dict[str, int]]:
    """Create train and validation loaders from dataset directory."""
    full_dataset = VineLeafDataset(root_dir=config.dataset_dir, transform=None)

    num_samples = len(full_dataset)
    num_val = int(num_samples * config.val_split)
    num_train = num_samples - num_val
    if num_train <= 0 or num_val <= 0:
        msg = "Invalid train/val split. Adjust val_split to keep both subsets non-empty."
        raise ValueError(msg)

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(full_dataset, [num_train, num_val], generator=generator)

    train_subset.dataset.transform = build_train_transforms(image_size=config.image_size)
    val_subset.dataset.transform = build_eval_transforms(image_size=config.image_size)

    pin_memory = torch.cuda.is_available()

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

    return train_loader, val_loader, full_dataset.class_to_idx


def run_training(config: TrainingConfig) -> TrainingHistory:
    """Run end-to-end training using the DeepVine engine.

    Args:
        config: Training hyperparameters and runtime setup.

    Returns:
        Training history with best validation metric metadata.
    """
    train_loader, val_loader, class_to_idx = _build_train_val_loaders(config)
    num_classes = len(class_to_idx)

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
        checkpoint_path=config.checkpoint_path,
        class_to_idx=class_to_idx,
    )

    return history


if __name__ == "__main__":
    default_config = TrainingConfig()
    run_training(default_config)
