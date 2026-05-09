"""
Integrated training pipeline for DeepVine.

This module orchestrates the complete training workflow:
1. YOLO detector training (for leaf/bunch detection)
2. Classification model training (for variety identification)

Optimized for RTX 4060Ti (8GB VRAM).
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from train import TrainingConfig, run_training
from train_yolo import train_detector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IntegratedTrainingConfig:
    """Configuration for integrated DeepVine training pipeline.
    
    Optimized for RTX 4060Ti with 8GB VRAM.
    """
    # Dataset paths
    dataset_dir: str = "./dataset"
    data_yaml: Optional[str] = None  # Path to COCO/YOLO format data.yaml for detection
    
    # YOLO Detector Training
    train_detector: bool = True
    detector_model: str = "yolov8n.pt"  # Use nano for smaller VRAM footprint
    detector_epochs: int = 30
    detector_batch_size: int = 16
    detector_patience: int = 10
    
    # Classifier Training (RTX 4060Ti optimized)
    train_classifier: bool = True
    classifier_batch_size: int = 8  # REDUCED from default 16 (RTX 4060Ti: 8GB)
    classifier_epochs: int = 30
    classifier_learning_rate: float = 3e-4
    classifier_weight_decay: float = 1e-4
    classifier_backbone: str = "resnet50"
    classifier_image_size: int = 224
    
    # GPU Optimization
    num_workers: int = 2  # REDUCED from 4 (better for smaller GPUs)
    mixed_precision: bool = True  # Automatic for CUDA
    gradient_accumulation_steps: int = 1  # Can increase if memory issues
    max_grad_norm: float = 1.0
    
    # Checkpointing
    detector_checkpoint: str = "./checkpoints/yolo_leaf_detector.pt"
    classifier_checkpoint: str = "./checkpoints/best_model.pt"
    resume_from_checkpoint: bool = False
    
    # Early stopping
    patience: int = 5


def _get_optimal_workers() -> int:
    """Determine optimal number of workers based on GPU availability."""
    if not torch.cuda.is_available():
        return 2
    
    # For small GPUs (RTX 4060 Ti), use fewer workers to reduce memory overhead
    device_name = torch.cuda.get_device_name(0).lower() if torch.cuda.device_count() > 0 else ""
    
    if "4060" in device_name or "3050" in device_name:
        return 2
    elif "3060" in device_name or "3070" in device_name:
        return 4
    else:
        return 4


def _validate_configuration(config: IntegratedTrainingConfig) -> None:
    """Validate configuration before training."""
    if config.train_detector and config.data_yaml is None:
        logger.warning(
            "YOLO detector training requested but no data.yaml provided. "
            "Skipping detector training. Provide --data-yaml to enable."
        )
        config.train_detector = False
    
    if config.train_detector and not Path(config.data_yaml).exists():
        msg = f"data.yaml not found: {config.data_yaml}"
        raise FileNotFoundError(msg)
    
    if not Path(config.dataset_dir).exists():
        msg = f"Dataset directory not found: {config.dataset_dir}"
        raise FileNotFoundError(msg)
    
    # Validate batch size for GPU memory
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        if config.classifier_batch_size > 8 and "4060" in device_name.lower():
            logger.warning(
                f"Batch size {config.classifier_batch_size} may cause OOM on {device_name}. "
                "Recommended: 8. Attempting training anyway..."
            )


def _log_gpu_info() -> None:
    """Log GPU information for debugging."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Using CPU (training will be VERY slow).")
        return
    
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    
    # Log GPU memory
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"Total GPU Memory: {total_memory:.2f} GB")
    
    # Empty cache before training
    torch.cuda.empty_cache()
    available = torch.cuda.mem_get_info()[0] / 1e9
    logger.info(f"Available GPU Memory: {available:.2f} GB")


def run_integrated_training(config: IntegratedTrainingConfig) -> dict[str, str]:
    """Run complete integrated training pipeline.
    
    Args:
        config: Training configuration.
        
    Returns:
        Dictionary with paths to best models.
    """
    logger.info("=" * 80)
    logger.info("DeepVine Integrated Training Pipeline")
    logger.info("=" * 80)
    
    _log_gpu_info()
    _validate_configuration(config)
    
    results = {}
    
    # ============================================================================
    # STEP 1: Train YOLO Detector (if requested)
    # ============================================================================
    if config.train_detector:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Training YOLO Object Detector")
        logger.info("Purpose: Detect grape leaves/bunches in images")
        logger.info("=" * 80)
        
        try:
            train_detector(
                data_yaml=config.data_yaml,
                epochs=config.detector_epochs,
                img_size=config.classifier_image_size,  # Use same size for consistency
                batch_size=config.detector_batch_size,
                model_version=config.detector_model,
                project="runs/detect",
                name="vine_leaf_detector",
            )
            results["detector"] = config.detector_checkpoint
            logger.info(f"✓ YOLO Detector training complete: {config.detector_checkpoint}")
        except Exception as e:
            logger.error(f"✗ YOLO Detector training failed: {e}")
            if not config.train_classifier:
                raise
            logger.warning("Continuing with classifier training (detector unavailable)...")
    else:
        logger.info("\nStep 1 SKIPPED: YOLO Detector training disabled")
    
    # ============================================================================
    # STEP 2: Train Classifier Model (if requested)
    # ============================================================================
    if config.train_classifier:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Training Leaf Variety Classifier")
        logger.info("Purpose: Classify detected leaves into grapevine varieties")
        logger.info("=" * 80)
        logger.info(f"Batch Size: {config.classifier_batch_size} (optimized for RTX 4060Ti)")
        logger.info(f"Workers: {config.num_workers}")
        logger.info(f"Mixed Precision: {config.mixed_precision}")
        
        # Configure classifier training
        classifier_config = TrainingConfig(
            dataset_dir=config.dataset_dir,
            image_size=config.classifier_image_size,
            batch_size=config.classifier_batch_size,
            epochs=config.classifier_epochs,
            learning_rate=config.classifier_learning_rate,
            weight_decay=config.classifier_weight_decay,
            num_workers=config.num_workers,
            backbone=config.classifier_backbone,
            checkpoint_path=config.classifier_checkpoint,
            resume_from_checkpoint=config.resume_from_checkpoint,
            patience=config.patience,
            max_grad_norm=config.max_grad_norm,
        )
        
        try:
            history = run_training(classifier_config)
            results["classifier"] = config.classifier_checkpoint
            logger.info(f"✓ Classifier training complete")
            logger.info(f"  Best Macro F1: {history.best_macro_f1:.4f}")
            logger.info(f"  Checkpoint: {history.best_checkpoint_path}")
        except Exception as e:
            logger.error(f"✗ Classifier training failed: {e}")
            raise
    else:
        logger.info("\nStep 2 SKIPPED: Classifier training disabled")
    
    # ============================================================================
    # Summary
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 80)
    for model_type, checkpoint in results.items():
        logger.info(f"  {model_type.upper()}: {checkpoint}")
    
    return results


def main() -> None:
    """Main entry point with default RTX 4060Ti configuration."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DeepVine Integrated Training Pipeline (RTX 4060Ti Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train both detector and classifier
  python train_integrated.py --data-yaml data.yaml
  
  # Train only classifier
  python train_integrated.py --skip-detector
  
  # Custom batch size
  python train_integrated.py --data-yaml data.yaml --batch-size 6
        """,
    )
    
    parser.add_argument(
        "--data-yaml",
        type=str,
        default=None,
        help="Path to COCO/YOLO format data.yaml for detector training",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./dataset",
        help="Root directory of classification dataset",
    )
    parser.add_argument(
        "--skip-detector",
        action="store_true",
        help="Skip YOLO detector training",
    )
    parser.add_argument(
        "--skip-classifier",
        action="store_true",
        help="Skip classifier training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for classifier (default: 8 for RTX 4060Ti)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of data loading workers (auto-detected if not specified)",
    )
    
    args = parser.parse_args()
    
    # Create config with CLI overrides
    config = IntegratedTrainingConfig(
        data_yaml=args.data_yaml,
        dataset_dir=args.dataset_dir,
        train_detector=not args.skip_detector,
        train_classifier=not args.skip_classifier,
        classifier_batch_size=args.batch_size,
        classifier_epochs=args.epochs,
        detector_epochs=args.epochs,
        num_workers=args.workers if args.workers is not None else _get_optimal_workers(),
        resume_from_checkpoint=args.resume,
    )
    
    run_integrated_training(config)


if __name__ == "__main__":
    main()
