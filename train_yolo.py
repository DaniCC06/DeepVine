import argparse
import logging
from pathlib import Path
import torch
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def train_detector(
    data_yaml: str,
    epochs: int = 30,
    img_size: int = 640,
    batch_size: int = 16,
    model_version: str = "yolov8n.pt",
    project: str = "runs/detect",
    name: str = "vine_leaf_detector",
):
    """
    Trains a YOLO model for leaf/bunch detection.
    
    Optimized for RTX 4060Ti (8GB VRAM).
    
    Pre-requisite: A valid data.yaml pointing to COCO/YOLO formatted annotations.
    
    Args:
        data_yaml: Path to dataset configuration (COCO/YOLO format).
        epochs: Number of training epochs (default: 30).
        img_size: Input image size (default: 640).
        batch_size: Batch size (default: 16 for small models, reduce for RTX 4060Ti).
        model_version: YOLO model variant (yolov8n.pt recommended for 8GB GPU).
        project: Project directory for runs.
        name: Run name.
    """
    logger.info("=" * 80)
    logger.info("YOLO Object Detector Training Pipeline")
    logger.info("=" * 80)
    
    # Validate GPU
    if not torch.cuda.is_available():
        logger.error("CUDA not available. GPU training required.")
        raise RuntimeError("CUDA not available")
    
    device_name = torch.cuda.get_device_name(0)
    logger.info(f"GPU Device: {device_name}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Auto-adjust batch size for small GPUs
    if "4060" in device_name.lower() and batch_size > 16:
        logger.warning(
            f"RTX 4060Ti detected. Reducing batch_size from {batch_size} to 16."
        )
        batch_size = 16
    
    logger.info(f"Model: {model_version} | Batch Size: {batch_size} | Epochs: {epochs}")
    logger.info(f"Input Size: {img_size}x{img_size} | Config: {data_yaml}")
    
    logger.info("\nLoading YOLO model...")
    model = YOLO(model_version)
    
    logger.info("Starting detector training...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=project,
        name=name,
        device=0,  # GPU 0
        patience=10,  # Early stopping patience
        save=True,
        save_period=5,  # Save checkpoint every 5 epochs
        # Mixed precision for RTX 4060Ti
        amp=True,  # Automatic Mixed Precision enabled
        # Optimization flags
        close_mosaic=15,  # Close mosaic augmentation in last 15 epochs
        mosaic=1.0,  # Mosaic augmentation
        # Validation
        val=True,
        verbose=True,
    )
    
    logger.info("Detector training complete.")
    
    # Copy best model to checkpoints directory
    best_model_path = Path(project) / name / "weights" / "best.pt"
    target_path = Path("checkpoints") / "yolo_leaf_detector.pt"
    
    if best_model_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(best_model_path, target_path)
        logger.info(f"✓ Best detector model saved to {target_path}")
        logger.info(f"  Model size: {target_path.stat().st_size / 1e6:.2f} MB")
    else:
        logger.warning(f"Could not find best model at {best_model_path}")
    
    return str(target_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLO detector for DeepVine (RTX 4060Ti optimized)",
        epilog="""
Example usage:
  python train_yolo.py --data data.yaml --epochs 30 --batch 16 --model yolov8n.pt
        """,
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml dataset config (COCO/YOLO format)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16, reduce to 8-12 if OOM on RTX 4060Ti)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model variant (yolov8n.pt, yolov8s.pt, etc.) - nano recommended for small GPUs",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Project directory for output",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="vine_leaf_detector",
        help="Run name",
    )
    
    args = parser.parse_args()
    
    train_detector(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        model_version=args.model,
        project=args.project,
        name=args.name,
    )
