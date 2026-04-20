from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass(slots=True)
class TrainEpochMetrics:
    """Metrics produced by a training epoch.

    Attributes:
        loss: Mean loss across the epoch.
        accuracy: Mean top-1 accuracy across the epoch.
    """

    loss: float
    accuracy: float


@dataclass(slots=True)
class EvalMetrics:
    """Evaluation metrics for validation/inference.

    Attributes:
        loss: Mean loss across the validation set.
        accuracy: Mean top-1 accuracy.
        macro_f1: Macro-averaged F1 score.
        weighted_f1: Class-frequency weighted F1 score.
        f1_per_class: One F1 score per class index.
        confusion_matrix: Confusion matrix with shape (C, C).
    """

    loss: float
    accuracy: float
    macro_f1: float
    weighted_f1: float
    f1_per_class: list[float]
    confusion_matrix: Tensor


@dataclass(slots=True)
class TrainingHistory:
    """Complete training history and best checkpoint metadata."""

    train_loss: list[float]
    train_accuracy: list[float]
    val_loss: list[float]
    val_accuracy: list[float]
    val_macro_f1: list[float]
    best_macro_f1: float
    best_checkpoint_path: str


class FocalLabelSmoothingLoss(nn.Module):
    """Multiclass focal loss with optional label smoothing.

    This loss combines focal modulation with label smoothing to improve learning
    in class-imbalanced datasets where minor varieties are under-represented.

    Args:
        gamma: Focal exponent controlling hard-sample emphasis.
        label_smoothing: Smoothing factor in [0, 1).
        alpha: Optional class-wise weighting tensor of shape (num_classes,).
        reduction: Reduction strategy: "mean", "sum", or "none".
    """

    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        alpha: Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0.0:
            msg = "gamma must be >= 0.0"
            raise ValueError(msg)
        if not 0.0 <= label_smoothing < 1.0:
            msg = "label_smoothing must be in [0, 1)."
            raise ValueError(msg)
        if reduction not in {"mean", "sum", "none"}:
            msg = "reduction must be one of: 'mean', 'sum', 'none'."
            raise ValueError(msg)

        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute focal loss with smoothed targets.

        Args:
            logits: Raw logits with shape (N, C).
            targets: Integer labels with shape (N,).

        Returns:
            Scalar loss (or per-sample vector if reduction='none').
        """
        if logits.ndim != 2:
            msg = "logits must have shape (N, C)."
            raise ValueError(msg)
        if targets.ndim != 1:
            msg = "targets must have shape (N,)."
            raise ValueError(msg)

        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        with torch.no_grad():
            smooth_pos = 1.0 - self.label_smoothing
            smooth_neg = self.label_smoothing / float(max(num_classes - 1, 1))
            smoothed_targets = torch.full_like(log_probs, smooth_neg)
            smoothed_targets.scatter_(1, targets.unsqueeze(1), smooth_pos)

        ce_per_sample = -(smoothed_targets * log_probs).sum(dim=1)
        pt = (smoothed_targets * probs).sum(dim=1).clamp(min=1e-8, max=1.0)
        focal_factor = (1.0 - pt).pow(self.gamma)
        loss = focal_factor * ce_per_sample

        if self.alpha is not None:
            alpha_weights = self.alpha.gather(0, targets)
            loss = loss * alpha_weights

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def resolve_device(preferred: str | None = None) -> torch.device:
    """Resolve torch device transparently for CPU/CUDA.

    Args:
        preferred: Optional explicit device string (e.g., 'cuda', 'cpu').

    Returns:
        A valid torch.device object.
    """
    if preferred is not None:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _update_confusion_matrix(confusion: Tensor, preds: Tensor, targets: Tensor, num_classes: int) -> Tensor:
    """Accumulate confusion matrix values efficiently with bincount."""
    indices = (targets * num_classes + preds).to(torch.int64)
    batch_confusion = torch.bincount(indices, minlength=num_classes * num_classes)
    return confusion + batch_confusion.reshape(num_classes, num_classes)


def compute_f1_scores(confusion_matrix: Tensor) -> tuple[list[float], float, float]:
    """Compute class-wise, macro and weighted F1 from confusion matrix."""
    cm = confusion_matrix.float()
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)

    support = cm.sum(dim=1)
    macro_f1 = f1.mean().item()
    weighted_f1 = (f1 * support).sum().item() / (support.sum().item() + 1e-12)

    return [float(v) for v in f1.tolist()], float(macro_f1), float(weighted_f1)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: OneCycleLR,
    device: torch.device,
    scaler: GradScaler,
) -> TrainEpochMetrics:
    """Train model for one epoch using automatic mixed precision.

    Args:
        model: Classification model.
        dataloader: Training data loader.
        criterion: Training loss.
        optimizer: Optimizer instance.
        scheduler: OneCycleLR scheduler stepped per batch.
        device: Target torch device.
        scaler: Gradient scaler for AMP.

    Returns:
        Aggregated loss and accuracy for the epoch.
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = device.type == "cuda"

    progress = tqdm(dataloader, desc="Train", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += batch_size

        progress.set_postfix(loss=f"{running_loss / max(total, 1):.4f}", acc=f"{correct / max(total, 1):.4f}")

    return TrainEpochMetrics(
        loss=running_loss / max(total, 1),
        accuracy=correct / max(total, 1),
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    num_classes: int,
    device: torch.device,
) -> EvalMetrics:
    """Evaluate model with confusion matrix and per-class F1.

    Args:
        model: Classification model.
        dataloader: Validation data loader.
        criterion: Validation loss.
        num_classes: Number of classes.
        device: Target torch device.

    Returns:
        Detailed validation metrics including confusion matrix.
    """
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    progress = tqdm(dataloader, desc="Eval", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size

        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += batch_size

        confusion = _update_confusion_matrix(
            confusion=confusion,
            preds=preds.detach().cpu(),
            targets=targets.detach().cpu(),
            num_classes=num_classes,
        )

        progress.set_postfix(loss=f"{running_loss / max(total, 1):.4f}", acc=f"{correct / max(total, 1):.4f}")

    f1_per_class, macro_f1, weighted_f1 = compute_f1_scores(confusion)

    return EvalMetrics(
        loss=running_loss / max(total, 1),
        accuracy=correct / max(total, 1),
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        f1_per_class=f1_per_class,
        confusion_matrix=confusion,
    )


def save_best_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: OneCycleLR,
    epoch: int,
    metrics: EvalMetrics,
    class_to_idx: dict[str, int],
) -> None:
    """Persist best model checkpoint based on validation macro F1."""
    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": metrics.loss,
        "val_accuracy": metrics.accuracy,
        "val_macro_f1": metrics.macro_f1,
        "val_weighted_f1": metrics.weighted_f1,
        "f1_per_class": metrics.f1_per_class,
        "confusion_matrix": metrics.confusion_matrix,
        "class_to_idx": class_to_idx,
    }
    torch.save(payload, ckpt_path)


def fit(
    model: nn.Module,
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    val_loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    criterion: nn.Module,
    num_classes: int,
    epochs: int,
    max_lr: float,
    device: torch.device,
    checkpoint_path: str | Path,
    class_to_idx: dict[str, int],
) -> TrainingHistory:
    """Run full training process with AMP, OneCycleLR and checkpointing.

    Args:
        model: Classification model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        num_classes: Number of classes.
        epochs: Number of epochs.
        max_lr: Maximum LR for OneCycle policy.
        device: Target torch device.
        checkpoint_path: Path where best model is stored.
        class_to_idx: Mapping from class name to index.

    Returns:
        Full training history and best-score metadata.
    """
    if epochs <= 0:
        msg = "epochs must be > 0."
        raise ValueError(msg)

    model.to(device)
    scaler = GradScaler(enabled=device.type == "cuda")

    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=10_000.0,
    )

    history = TrainingHistory(
        train_loss=[],
        train_accuracy=[],
        val_loss=[],
        val_accuracy=[],
        val_macro_f1=[],
        best_macro_f1=float("-inf"),
        best_checkpoint_path=str(checkpoint_path),
    )

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            num_classes=num_classes,
            device=device,
        )

        history.train_loss.append(train_metrics.loss)
        history.train_accuracy.append(train_metrics.accuracy)
        history.val_loss.append(val_metrics.loss)
        history.val_accuracy.append(val_metrics.accuracy)
        history.val_macro_f1.append(val_metrics.macro_f1)

        if val_metrics.macro_f1 > history.best_macro_f1:
            history.best_macro_f1 = val_metrics.macro_f1
            save_best_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=val_metrics,
                class_to_idx=class_to_idx,
            )

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} | "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f} "
            f"val_macro_f1={val_metrics.macro_f1:.4f}"
        )

    return history
