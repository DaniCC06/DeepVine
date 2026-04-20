from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import sys
from typing import Any
import base64

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from PIL import Image
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deepvine.datasets import build_eval_transforms
from deepvine.engine import resolve_device
from deepvine.models import BackboneName, VineLeafClassifier
from interpretability import GradCAM, get_target_layer, overlay_heatmap_on_image

UNCERTAIN_LABEL = "Variedad Incierta/Requiere Experto"


@dataclass(slots=True)
class InferenceConfig:
    """Runtime configuration for FastAPI inference service.

    Args:
        checkpoint_path: Trained best model checkpoint.
        backbone: Backbone used during training.
        image_size: Deterministic inference image size.
        confidence_threshold: Scientific uncertainty threshold.
        device: Optional explicit runtime device.
    """

    checkpoint_path: str = "./checkpoints/best_model.pt"
    backbone: BackboneName = "resnet50"
    image_size: int = 224
    confidence_threshold: float = 0.70
    device: str | None = None


class PredictionResponse(BaseModel):
    """Schema returned by /predict endpoint."""

    predicted_class: str = Field(..., description="Final class after uncertainty thresholding.")
    raw_predicted_class: str = Field(..., description="Top-1 class predicted by the model.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Top-1 softmax confidence.")
    is_uncertain: bool = Field(..., description="True when confidence is below threshold.")
    threshold: float = Field(..., ge=0.0, le=1.0)
    gradcam_overlay_base64: str | None = Field(default=None)


class InferenceService:
    """Encapsulates model loading, preprocessing and prediction logic."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.preprocess = build_eval_transforms(image_size=config.image_size)
        self.model, self.class_to_idx = self._load_model_and_mapping(
            checkpoint_path=config.checkpoint_path,
            backbone=config.backbone,
            device=self.device,
        )
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

    @staticmethod
    def _load_model_and_mapping(
        checkpoint_path: str | Path,
        backbone: BackboneName,
        device: torch.device,
    ) -> tuple[VineLeafClassifier, dict[str, int]]:
        """Load model and class mapping from checkpoint."""
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

    def _gradcam_overlay_base64(self, image: Image.Image, class_idx: int) -> str:
        """Generate Grad-CAM overlay and return PNG bytes as base64."""
        resized_original = image.resize((self.config.image_size, self.config.image_size), resample=Image.BILINEAR)
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        target_layer = get_target_layer(model=self.model, backbone=self.config.backbone)
        gradcam = GradCAM(model=self.model, target_layer=target_layer)
        try:
            cam = gradcam.generate(input_tensor=input_tensor, class_idx=class_idx).detach().cpu()
            overlay = overlay_heatmap_on_image(image=resized_original, heatmap=cam)
        finally:
            gradcam.remove_hooks()

        buffer = io.BytesIO()
        overlay.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def predict(self, image: Image.Image, include_gradcam: bool) -> PredictionResponse:
        """Predict class, apply uncertainty threshold and optionally Grad-CAM."""
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx_tensor = torch.max(probs, dim=1)

        pred_idx = int(pred_idx_tensor.item())
        conf = float(confidence.item())
        raw_label = self.idx_to_class[pred_idx]
        is_uncertain = conf < self.config.confidence_threshold
        final_label = UNCERTAIN_LABEL if is_uncertain else raw_label

        gradcam_b64: str | None = None
        if include_gradcam:
            gradcam_b64 = self._gradcam_overlay_base64(image=image, class_idx=pred_idx)

        return PredictionResponse(
            predicted_class=final_label,
            raw_predicted_class=raw_label,
            confidence=conf,
            is_uncertain=is_uncertain,
            threshold=self.config.confidence_threshold,
            gradcam_overlay_base64=gradcam_b64,
        )


app = FastAPI(
    title="DeepVine Inference API",
    description="Production-ready backend for grapevine leaf variety classification.",
    version="1.0.0",
)

service: InferenceService | None = None


@app.on_event("startup")
def on_startup() -> None:
    """Load model at service startup for low-latency inference."""
    global service
    config = InferenceConfig()
    service = InferenceService(config=config)


@app.get("/health")
def health() -> dict[str, Any]:
    """Simple liveness endpoint."""
    return {"status": "ok", "service": "deepvine-inference"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Vine leaf image file."),
    include_gradcam: bool = Query(default=False, description="Include Grad-CAM overlay as base64 PNG."),
) -> PredictionResponse:
    """Predict vine variety from image.

    Applies deterministic preprocessing and a scientific confidence threshold:
    if confidence < 0.70, the response label is set to
    'Variedad Incierta/Requiere Experto'.
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        raw_bytes = await file.read()
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {exc}") from exc

    try:
        return service.predict(image=image, include_gradcam=include_gradcam)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
