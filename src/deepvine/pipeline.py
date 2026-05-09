from typing import Any, Dict
from PIL import Image
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

class ObjectDetectionClassificationPipeline:
    """Pipeline that first detects leaves/bunches using YOLO, crops them,
    and then classifies each crop using the main classifier. Combines results using Soft Voting."""
    
    def __init__(self, yolo_model_path: str, classifier_service: Any, yolo_conf: float = 0.40):
        self.yolo_model_path = yolo_model_path
        self.classifier_service = classifier_service
        self.device = classifier_service.device
        self.yolo_conf = yolo_conf
        
        try:
            self.detector = YOLO(self.yolo_model_path)
            logger.info(f"Loaded YOLO detector from {yolo_model_path}")
        except Exception as e:
            logger.warning(f"Could not load YOLO detector. Falling back to full image classification. Error: {e}")
            self.detector = None

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Predicts the plant class by combining YOLO detections and classification."""
        if self.detector is None:
            return self._fallback_predict(image)
            
        results = self.detector(image, conf=self.yolo_conf, verbose=False)
        detections = results[0].boxes
        
        if len(detections) == 0:
            logger.info("No leaves detected. Falling back to full image classification.")
            return self._fallback_predict(image)
            
        rois = []
        for box in detections.xyxy:
            x1, y1, x2, y2 = map(int, box.tolist())
            crop = image.crop((x1, y1, x2, y2))
            rois.append(crop)
            
        # Process all crops as a batch
        tensor_rois = torch.stack([self.classifier_service.preprocess(crop) for crop in rois])
        tensor_rois = tensor_rois.to(self.device)
        
        with torch.no_grad():
            logits = self.classifier_service.model(tensor_rois)
            probs = F.softmax(logits, dim=1)
            
        # Soft Voting: Aggregate probabilities across all detected leaves
        plant_probs = probs.mean(dim=0)
        confidence, predicted_idx = torch.max(plant_probs, dim=0)
        
        idx = int(predicted_idx.item())
        conf = float(confidence.item())
        predicted_class_name = self.classifier_service.idx_to_class[idx]
        
        return {
            "predicted_idx": idx,
            "raw_predicted_class": predicted_class_name,
            "confidence": conf,
            "leaves_detected": len(rois),
            "boxes": detections.xyxy.tolist()
        }
        
    def _fallback_predict(self, image: Image.Image) -> Dict[str, Any]:
        input_tensor = self.classifier_service.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.classifier_service.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx_tensor = torch.max(probs, dim=1)

        idx = int(pred_idx_tensor.item())
        conf = float(confidence.item())
        predicted_class_name = self.classifier_service.idx_to_class[idx]
        
        return {
            "predicted_idx": idx,
            "raw_predicted_class": predicted_class_name,
            "confidence": conf,
            "leaves_detected": 0,
            "boxes": []
        }
