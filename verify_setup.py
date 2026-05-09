"""
Pre-training verification script for DeepVine.

Validates:
- GPU compatibility and memory
- Dataset structure  
- Python dependencies
- Configuration consistency

Run this BEFORE training to catch issues early.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent


def check_gpu() -> bool:
    """Verify GPU availability and memory."""
    logger.info("\n" + "=" * 80)
    logger.info("GPU VERIFICATION")
    logger.info("=" * 80)
    
    if not torch.cuda.is_available():
        logger.error("✗ CUDA not available. CPU training is EXTREMELY slow.")
        logger.error("  Please ensure NVIDIA GPU drivers are installed.")
        return False
    
    device_name = torch.cuda.get_device_name(0)
    device_count = torch.cuda.device_count()
    
    logger.info(f"✓ CUDA Available (Device: {device_name})")
    logger.info(f"✓ GPU Count: {device_count}")
    
    # Memory check
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    available_memory = torch.cuda.mem_get_info()[0] / 1e9
    
    logger.info(f"✓ Total GPU Memory: {total_memory:.2f} GB")
    logger.info(f"✓ Available GPU Memory: {available_memory:.2f} GB")
    
    # GPU compatibility check
    if "4060" in device_name.lower():
        logger.info("✓ RTX 4060Ti detected - using optimized settings (batch_size=8)")
        if available_memory < 7.5:
            logger.warning("⚠ Low available memory. Close other applications.")
    elif "3050" in device_name.lower() or "4050" in device_name.lower():
        logger.warning("⚠ Detected small GPU (4GB). Reduce batch size to 4.")
    elif "4070" in device_name.lower() or "3070" in device_name.lower():
        logger.info("✓ Large GPU detected - can use batch_size=16-24")
    
    return True


def check_pytorch() -> bool:
    """Verify PyTorch and dependencies."""
    logger.info("\n" + "=" * 80)
    logger.info("DEPENDENCY VERIFICATION")
    logger.info("=" * 80)
    
    logger.info(f"✓ PyTorch: {torch.__version__}")
    logger.info(f"✓ CUDA: {torch.version.cuda}")
    
    try:
        import torchvision
        logger.info(f"✓ TorchVision: {torchvision.__version__}")
    except ImportError:
        logger.error("✗ TorchVision not installed: pip install torchvision")
        return False
    
    try:
        from ultralytics import YOLO
        logger.info("✓ Ultralytics (YOLO): Installed")
    except ImportError:
        logger.error("✗ Ultralytics not installed: pip install ultralytics")
        return False
    
    try:
        from PIL import Image
        logger.info("✓ Pillow: Installed")
    except ImportError:
        logger.error("✗ Pillow not installed: pip install Pillow")
        return False
    
    try:
        import fastapi
        logger.info("✓ FastAPI: Installed")
    except ImportError:
        logger.warning("⚠ FastAPI not installed (only needed for serving)")
    
    return True


def check_dataset(dataset_dir: str = "./dataset") -> bool:
    """Verify dataset structure."""
    logger.info("\n" + "=" * 80)
    logger.info("DATASET VERIFICATION")
    logger.info("=" * 80)
    
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        logger.error(f"✗ Dataset directory not found: {dataset_path}")
        logger.info(f"  Create it at: {dataset_path.resolve()}")
        return False
    
    logger.info(f"✓ Dataset directory exists: {dataset_path.resolve()}")
    
    # Check for class folders
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if not class_dirs:
        logger.error("✗ No class folders found in dataset directory")
        logger.info("  Expected structure:")
        logger.info("  dataset/")
        logger.info("    ├── variety_1/")
        logger.info("    │   ├── image1.jpg")
        logger.info("    │   └── image2.jpg")
        logger.info("    ├── variety_2/")
        logger.info("    │   ├── image1.jpg")
        logger.info("    │   └── ...")
        return False
    
    logger.info(f"✓ Found {len(class_dirs)} variety classes:")
    
    # Check images in each class
    total_images = 0
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    
    for class_dir in sorted(class_dirs):
        images = [
            f for f in class_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        total_images += len(images)
        
        status = "✓" if len(images) >= 20 else "⚠"
        logger.info(f"  {status} {class_dir.name}: {len(images)} images")
        
        if len(images) < 5:
            logger.warning(f"    ⚠ Very few images for {class_dir.name}")
    
    if total_images == 0:
        logger.error("✗ No images found in any class")
        return False
    
    logger.info(f"✓ Total images: {total_images}")
    
    if total_images < 100:
        logger.warning("⚠ Small dataset (<100 images). Transfer learning recommended.")
    
    return True


def check_detection_dataset(data_yaml: str = "annotations/data.yaml") -> bool:
    """Verify YOLO detection dataset (optional)."""
    logger.info("\n" + "=" * 80)
    logger.info("YOLO DETECTION DATASET VERIFICATION (Optional)")
    logger.info("=" * 80)
    
    data_yaml_path = Path(data_yaml)
    
    if not data_yaml_path.exists():
        logger.info(f"ℹ  No detection dataset found ({data_yaml})")
        logger.info("  This is optional. Classifier training will proceed.")
        return True  # Not required
    
    logger.info(f"✓ Found detection config: {data_yaml_path}")
    
    try:
        import yaml
        with open(data_yaml_path) as f:
            config = yaml.safe_load(f)
        
        if not isinstance(config, dict):
            logger.error("✗ Invalid YAML format")
            return False
        
        required_keys = {"path", "train", "val", "nc", "names"}
        missing_keys = required_keys - set(config.keys())
        
        if missing_keys:
            logger.error(f"✗ Missing required keys: {missing_keys}")
            return False
        
        logger.info(f"✓ Valid YAML structure")
        logger.info(f"  - Classes: {config['nc']}")
        logger.info(f"  - Class names: {config['names']}")
        
        # Check directories
        data_root = Path(config["path"])
        if not data_root.exists():
            logger.error(f"✗ Data path not found: {data_root}")
            return False
        
        train_dir = data_root / config["train"]
        val_dir = data_root / config["val"]
        
        if not train_dir.exists():
            logger.error(f"✗ Training images not found: {train_dir}")
            return False
        
        if not val_dir.exists():
            logger.warning(f"⚠ Validation images not found: {val_dir}")
        
        logger.info("✓ Detection dataset structure valid")
        return True
        
    except ImportError:
        logger.warning("⚠ PyYAML not installed. Can't validate YAML.")
        return False
    except Exception as e:
        logger.error(f"✗ Error parsing YAML: {e}")
        return False


def check_checkpoint_dirs() -> bool:
    """Verify checkpoint directories exist."""
    logger.info("\n" + "=" * 80)
    logger.info("CHECKPOINT DIRECTORY VERIFICATION")
    logger.info("=" * 80)
    
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    
    if not checkpoint_dir.exists():
        logger.info(f"Creating checkpoint directory: {checkpoint_dir}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"✓ Checkpoint directory ready: {checkpoint_dir}")
    
    return True


def print_recommendations() -> None:
    """Print training recommendations."""
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDED COMMANDS")
    logger.info("=" * 80)
    
    logger.info("\n1. Train Classifier Only (Quick):")
    logger.info("   python run_train.py")
    
    logger.info("\n2. Train Full Pipeline (Recommended) ⭐:")
    logger.info("   python train_integrated.py --data-yaml annotations/data.yaml")
    
    logger.info("\n3. Train YOLO Detector Only:")
    logger.info("   python train_yolo.py --data annotations/data.yaml")
    
    logger.info("\n4. Resume from Checkpoint:")
    logger.info("   python train_integrated.py --data-yaml annotations/data.yaml --resume")
    
    logger.info("\n5. Monitor GPU during training:")
    logger.info("   nvidia-smi -l 1  # Refresh every 1 second")


def main() -> int:
    """Run all checks and report status."""
    logger.info("\n")
    logger.info("🍇 DeepVine Pre-Training Verification")
    logger.info("=" * 80)
    
    checks = [
        ("GPU & CUDA", check_gpu),
        ("PyTorch Dependencies", check_pytorch),
        ("Classification Dataset", lambda: check_dataset("./dataset")),
        ("Detection Dataset (Optional)", lambda: check_detection_dataset("annotations/data.yaml")),
        ("Checkpoint Directories", check_checkpoint_dirs),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            logger.error(f"✗ {check_name} check failed: {e}")
            results[check_name] = False
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {check_name}")
    
    logger.info(f"\nResult: {passed}/{total} checks passed")
    
    if passed == total:
        logger.info("\n✅ All checks passed! Ready to train.")
        print_recommendations()
        return 0
    else:
        logger.error("\n❌ Some checks failed. Fix issues above before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
