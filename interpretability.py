from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deepvine.datasets import build_eval_transforms
from deepvine.engine import resolve_device


class GradCAM:
    """Gradient-weighted Class Activation Mapping helper.

    Args:
        model: Trained classification model in eval mode.
        target_layer: Convolutional layer used to compute CAM maps.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Tensor | None = None
        self.gradients: Tensor | None = None
        self._forward_handle = self.target_layer.register_forward_hook(self._save_activations)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module: nn.Module, _input: tuple[Tensor, ...], output: Tensor) -> None:
        self.activations = output.detach()

    def _save_gradients(
        self,
        _module: nn.Module,
        _grad_input: tuple[Tensor | None, ...],
        grad_output: tuple[Tensor | None, ...],
    ) -> None:
        if grad_output[0] is not None:
            self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        """Remove hooks to prevent memory leaks."""
        self._forward_handle.remove()
        self._backward_handle.remove()

    def generate(self, input_tensor: Tensor, class_idx: int | None = None) -> Tensor:
        """Generate Grad-CAM heatmap for a single image tensor.

        Args:
            input_tensor: Input with shape (1, 3, H, W).
            class_idx: Optional target class index. If None, predicted class is used.

        Returns:
            Heatmap tensor normalized to [0, 1] with shape (H, W).
        """
        if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
            msg = "input_tensor must have shape (1, 3, H, W)."
            raise ValueError(msg)

        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[:, class_idx]
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            msg = "Failed to capture activations/gradients. Check target layer selection."
            raise RuntimeError(msg)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)

        cam = cam.squeeze(0).squeeze(0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam


def overlay_heatmap_on_image(image: Image.Image, heatmap: Tensor, alpha: float = 0.45) -> Image.Image:
    """Overlay a red-scale heatmap over the original image.

    Args:
        image: Base RGB image.
        heatmap: Heatmap tensor in [0, 1] with shape (H, W).
        alpha: Heatmap blending factor.

    Returns:
        PIL image with heatmap overlay.
    """
    image_np = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    heatmap_np = heatmap.detach().cpu().numpy().astype(np.float32)

    if heatmap_np.shape != image_np.shape[:2]:
        msg = "Heatmap and image spatial dimensions must match."
        raise ValueError(msg)

    red_map = np.zeros_like(image_np)
    red_map[..., 0] = heatmap_np
    red_map[..., 1] = np.clip(heatmap_np * 0.35, 0.0, 1.0)

    blended = np.clip((1.0 - alpha) * image_np + alpha * red_map, 0.0, 1.0)
    return Image.fromarray((blended * 255.0).astype(np.uint8))


def get_target_layer(model: nn.Module, backbone: str) -> nn.Module:
    """Resolve the last convolutional block for Grad-CAM.

    Args:
        model: Model used for inference.
        backbone: Backbone name: 'resnet50' or 'efficientnet_b0'.

    Returns:
        Target layer module.
    """
    if hasattr(model, "model"):
        base_model = getattr(model, "model")
    else:
        base_model = model

    if backbone == "resnet50":
        return base_model.layer4[-1].conv3
    if backbone == "efficientnet_b0":
        return base_model.features[-1][0]

    msg = f"Unsupported backbone for Grad-CAM: {backbone}"
    raise ValueError(msg)


def generate_gradcam_artifacts(
    model: nn.Module,
    image_path: str | Path,
    output_dir: str | Path,
    backbone: str,
    image_size: int = 224,
    class_idx: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Generate and save Grad-CAM artifacts for one vine leaf image.

    Args:
        model: Trained model.
        image_path: Path to source image.
        output_dir: Directory where artifacts are saved.
        backbone: Backbone identifier.
        image_size: Model input size.
        class_idx: Optional class index to explain.
        device: Optional explicit runtime device.

    Returns:
        Metadata with predicted class, confidence and output paths.
    """
    resolved_device = resolve_device(device)
    model = model.to(resolved_device)
    model.eval()

    source_path = Path(image_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    original = Image.open(source_path).convert("RGB")
    preprocess = build_eval_transforms(image_size=image_size)
    input_tensor = preprocess(original).unsqueeze(0).to(resolved_device)

    target_layer = get_target_layer(model=model, backbone=backbone)
    gradcam = GradCAM(model=model, target_layer=target_layer)

    try:
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(probs.argmax(dim=1).item())
            confidence = float(probs.max(dim=1).values.item())

        cam = gradcam.generate(input_tensor=input_tensor, class_idx=class_idx)
        cam_cpu = cam.detach().cpu()

        resized_original = original.resize((image_size, image_size), resample=Image.BILINEAR)
        overlay = overlay_heatmap_on_image(image=resized_original, heatmap=cam_cpu)

        heatmap_img = Image.fromarray((cam_cpu.numpy() * 255.0).astype(np.uint8), mode="L")

        overlay_file = output_path / f"{source_path.stem}_gradcam_overlay.png"
        heatmap_file = output_path / f"{source_path.stem}_gradcam_heatmap.png"

        overlay.save(overlay_file)
        heatmap_img.save(heatmap_file)

        return {
            "predicted_class_index": pred_idx,
            "confidence": confidence,
            "overlay_path": str(overlay_file),
            "heatmap_path": str(heatmap_file),
            "explained_class_index": class_idx if class_idx is not None else pred_idx,
        }
    finally:
        gradcam.remove_hooks()
