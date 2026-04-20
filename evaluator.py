from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deepvine.datasets import VineLeafDataset, build_eval_transforms
from deepvine.engine import EvalMetrics, evaluate, resolve_device
from deepvine.models import BackboneName, VineLeafClassifier


@dataclass(slots=True)
class EvaluationConfig:
    """Configuration for final scientific hold-out evaluation.

    Args:
        checkpoint_path: Path to best model checkpoint.
        test_dir: Directory of independent hold-out test images.
        output_dir: Directory where reports and plots are exported.
        backbone: Model backbone used during training.
        image_size: Input size used during training/inference.
        batch_size: Batch size for test loader.
        num_workers: Number of data loading workers.
        top_k_errors: Number of most frequent confusions to report.
        device: Optional runtime device override.
    """

    checkpoint_path: str = "./checkpoints/best_model.pt"
    test_dir: str = "./dataset_test"
    output_dir: str = "./reports"
    backbone: BackboneName = "resnet50"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    top_k_errors: int = 5
    device: str | None = None


def _load_model_from_checkpoint(
    checkpoint_path: str | Path,
    backbone: BackboneName,
    device: torch.device,
) -> tuple[VineLeafClassifier, dict[str, int]]:
    """Load model and class mapping from a training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_to_idx = ckpt.get("class_to_idx")
    if not isinstance(class_to_idx, dict) or not class_to_idx:
        msg = "Checkpoint does not contain a valid class_to_idx mapping."
        raise ValueError(msg)

    model = VineLeafClassifier(
        num_classes=len(class_to_idx),
        backbone=backbone,
        dropout_p=0.2,
        freeze_backbone=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, class_to_idx


def _build_test_loader(config: EvaluationConfig) -> tuple[DataLoader[tuple[Tensor, int]], dict[str, int]]:
    """Build deterministic hold-out test DataLoader."""
    dataset = VineLeafDataset(
        root_dir=config.test_dir,
        transform=build_eval_transforms(image_size=config.image_size),
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )
    return loader, dataset.class_to_idx


def _top_misclassifications(
    confusion_matrix: Tensor,
    idx_to_class: dict[int, str],
    top_k: int,
) -> list[dict[str, Any]]:
    """Extract the most frequent off-diagonal confusion pairs."""
    cm = confusion_matrix.clone().to(torch.int64)
    num_classes = cm.size(0)
    errors: list[dict[str, Any]] = []

    for true_idx in range(num_classes):
        row_total = int(cm[true_idx].sum().item())
        if row_total == 0:
            continue
        for pred_idx in range(num_classes):
            if true_idx == pred_idx:
                continue
            count = int(cm[true_idx, pred_idx].item())
            if count <= 0:
                continue
            errors.append(
                {
                    "true_class_index": true_idx,
                    "predicted_class_index": pred_idx,
                    "true_class": idx_to_class[true_idx],
                    "predicted_class": idx_to_class[pred_idx],
                    "count": count,
                    "rate_within_true_class": count / row_total,
                }
            )

    errors.sort(key=lambda item: item["count"], reverse=True)
    return errors[:top_k]


def _save_confusion_matrix_figure(
    confusion_matrix: Tensor,
    idx_to_class: dict[int, str],
    output_file: str | Path,
) -> None:
    """Render and export a publication-ready confusion matrix image."""
    labels = [idx_to_class[idx] for idx in range(len(idx_to_class))]
    fig_size = max(10, int(len(labels) * 0.6))

    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        confusion_matrix.numpy(),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="white",
    )
    plt.title("DeepVine Hold-out Test Confusion Matrix", fontsize=14, pad=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _build_markdown_report(report: dict[str, Any]) -> str:
    """Build a polished Markdown summary from evaluation data."""
    lines: list[str] = []
    lines.append("# DeepVine - Hold-out Test Final Report")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(f"- Timestamp (UTC): {report['timestamp_utc']}")
    lines.append(f"- Checkpoint: {report['checkpoint_path']}")
    lines.append(f"- Test set: {report['test_dir']}")
    lines.append(f"- Accuracy: {report['metrics']['accuracy']:.4f}")
    lines.append(f"- Macro F1: {report['metrics']['macro_f1']:.4f}")
    lines.append(f"- Weighted F1: {report['metrics']['weighted_f1']:.4f}")
    lines.append("")

    lines.append("## Class-wise F1")
    lines.append("| Class | Index | F1 |")
    lines.append("|---|---:|---:|")
    for class_row in report["class_f1_table"]:
        lines.append(
            f"| {class_row['class_name']} | {class_row['class_index']} | {class_row['f1_score']:.4f} |"
        )
    lines.append("")

    lines.append("## Top-5 Misclassifications")
    lines.append("| Rank | True Class | Predicted Class | Count | Error Rate in True Class |")
    lines.append("|---:|---|---|---:|---:|")
    for idx, row in enumerate(report["top_misclassifications"], start=1):
        lines.append(
            f"| {idx} | {row['true_class']} | {row['predicted_class']} | {row['count']} | {row['rate_within_true_class']:.2%} |"
        )
    lines.append("")

    lines.append("## Artifacts")
    lines.append(f"- JSON report: {report['artifacts']['json_report_path']}")
    lines.append(f"- Markdown report: {report['artifacts']['markdown_report_path']}")
    lines.append(f"- Confusion matrix image: {report['artifacts']['confusion_matrix_image_path']}")
    lines.append("")

    return "\n".join(lines)


def run_holdout_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    """Execute hold-out scientific evaluation and export all artifacts.

    Args:
        config: Evaluation runtime and data configuration.

    Returns:
        Dictionary with computed metrics and generated artifact paths.
    """
    device = resolve_device(config.device)
    model, class_to_idx_ckpt = _load_model_from_checkpoint(
        checkpoint_path=config.checkpoint_path,
        backbone=config.backbone,
        device=device,
    )

    test_loader, class_to_idx_test = _build_test_loader(config)
    if class_to_idx_test != class_to_idx_ckpt:
        msg = "Class mapping mismatch between checkpoint and hold-out test directory."
        raise ValueError(msg)

    criterion = nn.CrossEntropyLoss()
    metrics: EvalMetrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        num_classes=len(class_to_idx_ckpt),
        device=device,
    )

    idx_to_class = {idx: name for name, idx in class_to_idx_ckpt.items()}
    top_errors = _top_misclassifications(
        confusion_matrix=metrics.confusion_matrix,
        idx_to_class=idx_to_class,
        top_k=config.top_k_errors,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    confusion_image = output_dir / "confusion_matrix_holdout.png"
    report_json = output_dir / "holdout_report.json"
    report_md = output_dir / "holdout_report.md"

    _save_confusion_matrix_figure(
        confusion_matrix=metrics.confusion_matrix,
        idx_to_class=idx_to_class,
        output_file=confusion_image,
    )

    class_f1_table = [
        {
            "class_index": idx,
            "class_name": idx_to_class[idx],
            "f1_score": metrics.f1_per_class[idx],
        }
        for idx in range(len(metrics.f1_per_class))
    ]

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(Path(config.checkpoint_path).resolve()),
        "test_dir": str(Path(config.test_dir).resolve()),
        "config": asdict(config),
        "metrics": {
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            "weighted_f1": metrics.weighted_f1,
            "loss": metrics.loss,
        },
        "class_f1_table": class_f1_table,
        "top_misclassifications": top_errors,
        "artifacts": {
            "json_report_path": str(report_json.resolve()),
            "markdown_report_path": str(report_md.resolve()),
            "confusion_matrix_image_path": str(confusion_image.resolve()),
        },
    }

    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    report_md.write_text(_build_markdown_report(report), encoding="utf-8")

    return report


if __name__ == "__main__":
    default_config = EvaluationConfig()
    output = run_holdout_evaluation(default_config)
    print(json.dumps(output["metrics"], indent=2))
