from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deepvine.datasets import VineLeafDataset, build_eval_transforms
from deepvine.models import BackboneName, VineLeafClassifier


@dataclass(slots=True)
class ExportConfig:
    """Configuration for production model export.

    Args:
        checkpoint_path: Path to trained best checkpoint.
        output_dir: Directory for exported model artifacts.
        backbone: Backbone architecture used by the checkpoint.
        image_size: Input tensor side size.
        onnx_opset: ONNX opset version.
        calibration_dir: Directory used for static quantization calibration.
        calibration_batch_size: Batch size for calibration pass.
        calibration_max_samples: Maximum images used in calibration.
    """

    checkpoint_path: str = "./checkpoints/best_model.pt"
    output_dir: str = "./exports"
    backbone: BackboneName = "resnet50"
    image_size: int = 224
    onnx_opset: int = 17
    calibration_dir: str = "./dataset_test"
    calibration_batch_size: int = 16
    calibration_max_samples: int = 256


def _load_model(checkpoint_path: str | Path, backbone: BackboneName) -> tuple[VineLeafClassifier, dict[str, int]]:
    """Load trained model checkpoint on CPU for export tasks."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_to_idx = checkpoint.get("class_to_idx")
    if not isinstance(class_to_idx, dict) or not class_to_idx:
        msg = "Checkpoint does not contain a valid class_to_idx mapping."
        raise ValueError(msg)

    model = VineLeafClassifier(
        num_classes=len(class_to_idx),
        backbone=backbone,
        dropout_p=0.2,
        freeze_backbone=False,
    ).cpu()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, class_to_idx


def _export_torchscript(
    model: VineLeafClassifier,
    output_dir: Path,
    image_size: int,
) -> dict[str, str]:
    """Export TorchScript using scripting and tracing variants."""
    output_dir.mkdir(parents=True, exist_ok=True)

    scripted_path = output_dir / "deepvine_scripted.pt"
    traced_path = output_dir / "deepvine_traced.pt"

    scripted_model = torch.jit.script(model)
    scripted_model.save(str(scripted_path))

    example = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    traced_model = torch.jit.trace(model, example_inputs=example, strict=False)
    traced_model.save(str(traced_path))

    return {
        "scripted": str(scripted_path.resolve()),
        "traced": str(traced_path.resolve()),
    }


def _export_onnx(
    model: VineLeafClassifier,
    output_dir: Path,
    image_size: int,
    opset: int,
) -> str:
    """Export model to ONNX for cross-framework/mobile interoperability."""
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "deepvine.onnx"

    example = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    return str(onnx_path.resolve())


def _build_calibration_loader(config: ExportConfig) -> DataLoader[tuple[Tensor, int]] | None:
    """Create calibration DataLoader for static quantization.

    Returns None when calibration dataset is unavailable.
    """
    calib_root = Path(config.calibration_dir)
    if not calib_root.exists():
        return None

    dataset = VineLeafDataset(
        root_dir=calib_root,
        transform=build_eval_transforms(image_size=config.image_size),
    )
    return DataLoader(
        dataset=dataset,
        batch_size=config.calibration_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )


def _export_quantized_int8_fx(
    model: VineLeafClassifier,
    output_dir: Path,
    image_size: int,
    calibration_loader: DataLoader[tuple[Tensor, int]] | None,
    calibration_max_samples: int,
) -> dict[str, str]:
    """Apply post-training static quantization (INT8) with FX graph mode.

    This export is CPU/mobile oriented and uses qnnpack backend.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.quantized.engine = "qnnpack"

    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    fp32_model = model.cpu().eval()
    example_inputs = (torch.randn(1, 3, image_size, image_size),)
    qconfig_mapping = get_default_qconfig_mapping("qnnpack")

    prepared = prepare_fx(fp32_model, qconfig_mapping, example_inputs)

    seen = 0
    if calibration_loader is not None:
        for images, _ in calibration_loader:
            prepared(images)
            seen += images.size(0)
            if seen >= calibration_max_samples:
                break
    else:
        synthetic_batches = max(calibration_max_samples // 16, 1)
        for _ in range(synthetic_batches):
            prepared(torch.randn(16, 3, image_size, image_size))

    quantized = convert_fx(prepared)

    state_dict_path = output_dir / "deepvine_int8_state_dict.pth"
    scripted_path = output_dir / "deepvine_int8_scripted.pt"

    torch.save(quantized.state_dict(), state_dict_path)
    scripted_quantized = torch.jit.script(quantized)
    scripted_quantized.save(str(scripted_path))

    return {
        "int8_state_dict": str(state_dict_path.resolve()),
        "int8_scripted": str(scripted_path.resolve()),
    }


def _file_size_mb(path: str | Path) -> float:
    """Return file size in megabytes."""
    file_path = Path(path)
    return file_path.stat().st_size / (1024.0 * 1024.0)


def run_export_pipeline(config: ExportConfig) -> dict[str, Any]:
    """Run complete production export: TorchScript, ONNX and INT8 quantized."""
    output_dir = Path(config.output_dir)
    model, class_to_idx = _load_model(config.checkpoint_path, config.backbone)

    ts_paths = _export_torchscript(model=model, output_dir=output_dir, image_size=config.image_size)
    onnx_path = _export_onnx(
        model=model,
        output_dir=output_dir,
        image_size=config.image_size,
        opset=config.onnx_opset,
    )

    calibration_loader = _build_calibration_loader(config)
    quant_paths = _export_quantized_int8_fx(
        model=model,
        output_dir=output_dir,
        image_size=config.image_size,
        calibration_loader=calibration_loader,
        calibration_max_samples=config.calibration_max_samples,
    )

    metadata: dict[str, Any] = {
        "config": asdict(config),
        "num_classes": len(class_to_idx),
        "class_to_idx": class_to_idx,
        "artifacts": {
            "torchscript_scripted": ts_paths["scripted"],
            "torchscript_traced": ts_paths["traced"],
            "onnx": onnx_path,
            "int8_state_dict": quant_paths["int8_state_dict"],
            "int8_scripted": quant_paths["int8_scripted"],
        },
        "sizes_mb": {
            "torchscript_scripted": _file_size_mb(ts_paths["scripted"]),
            "torchscript_traced": _file_size_mb(ts_paths["traced"]),
            "onnx": _file_size_mb(onnx_path),
            "int8_state_dict": _file_size_mb(quant_paths["int8_state_dict"]),
            "int8_scripted": _file_size_mb(quant_paths["int8_scripted"]),
        },
    }

    metadata_path = output_dir / "export_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")

    return metadata


if __name__ == "__main__":
    default_config = ExportConfig()
    result = run_export_pipeline(default_config)
    print(json.dumps(result["sizes_mb"], indent=2))
