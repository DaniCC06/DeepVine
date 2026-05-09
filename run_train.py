"""
Quick classifier training script.

⚠️  RECOMMENDED: For best results with both detection and classification,
   use train_integrated.py instead:
   
   python train_integrated.py --data-yaml annotations/data.yaml

This script trains ONLY the classifier on your variety dataset.
"""
from __future__ import annotations

import logging
from train import TrainingConfig, run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=" * 80)
    logger.info("DeepVine Classifier Training (Quick Start)")
    logger.info("=" * 80)
    logger.info("\nℹ️  Tip: For full detection + classification pipeline, use:")
    logger.info("    python train_integrated.py --data-yaml annotations/data.yaml\n")
    
    # RTX 4060Ti optimized configuration
    cfg = TrainingConfig(
        dataset_dir=r".\dataset",
        batch_size=32,      # Optimized for RTX 4060Ti (16GB)
        epochs=30,
        num_workers=2,
    )
    
    history = run_training(cfg)

    print("\n" + "=" * 80)
    print("Training finished.")
    print(f"Best macro F1: {history.best_macro_f1:.4f}")
    print(f"Best checkpoint: {history.best_checkpoint_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
