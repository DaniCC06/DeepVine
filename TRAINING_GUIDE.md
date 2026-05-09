# 🍇 DeepVine Training Guide - RTX 4060Ti Optimized

## Overview

This guide explains **exactly how** to train the DeepVine model on an RTX 4060Ti GPU. The project consists of two main components:

1. **YOLO Object Detector** - Detects individual grape leaves/bunches in images
2. **Leaf Variety Classifier** - Classifies detected leaves into grapevine varieties

Both have been **optimized for RTX 4060Ti** (8GB VRAM).

---

## Prerequisites

### 1. Python Environment
```bash
# Create virtual environment
python -m venv deepvine_env

# Activate it
# On Windows:
deepvine_env\Scripts\activate
# On Linux/Mac:
source deepvine_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Structure

#### For Classifier Training (Mandatory)
```
dataset/
├── albillo_mayor/          # Variety 1
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
├── albillo_real/           # Variety 2
│   ├── image_1.jpg
│   └── ...
├── garnacha/               # Variety 3
│   └── ...
└── [more varieties...]
```

**Requirements:**
- Minimum 20-30 images per variety (more is better)
- Images should be RGB JPG, PNG, or similar
- Varieties should be separate folders

#### For Detector Training (Optional)
```
annotations/
├── data.yaml              # COCO/YOLO configuration
├── images/
│   ├── train/
│   │   ├── image_1.jpg
│   │   └── ...
│   └── val/
│       ├── image_1.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── image_1.txt    # COCO format annotations
    │   └── ...
    └── val/
        ├── image_1.txt
        └── ...
```

The `data.yaml` should look like:
```yaml
path: /path/to/data
train: images/train
val: images/val
nc: 7  # Number of classes (leaf/bunch detection)
names: ['leaf', 'bunch']  # Class names
```

---

## Training Procedures

### OPTION 1: Quick Start (Classifier Only)
**Best for:** Testing, quick iteration, or when no detection dataset available

```bash
python run_train.py
```

This trains **only** the leaf variety classifier on the classification dataset.

**Configuration:** Automatically uses optimal settings for RTX 4060Ti:
- Batch Size: 8 (optimized for 8GB VRAM)
- Workers: 2
- Epochs: 30
- Mixed Precision: Enabled

---

### OPTION 2: Integrated Training (Both Detector + Classifier) ⭐ RECOMMENDED
**Best for:** Production-ready pipeline with both detection and classification

```bash
# Train both YOLO detector and classifier
python train_integrated.py --data-yaml annotations/data.yaml

# This runs SEQUENTIALLY:
# 1. YOLO detector training (detects leaves/bunches)
# 2. Classifier training (identifies varieties)
```

**Step-by-step:**

#### Step 1: Prepare Detection Dataset
If you have YOLO-formatted detection data:
```bash
python train_integrated.py \
    --data-yaml annotations/data.yaml \
    --dataset-dir ./dataset \
    --epochs 30 \
    --batch-size 8
```

#### Step 2: Monitor GPU Usage
Open another terminal and monitor:
```bash
nvidia-smi -l 1  # Refresh every 1 second
```

**Expected GPU Memory Usage:**
- Detector training: ~5-6 GB
- Classifier training: ~4-5 GB

#### Step 3: Wait for Completion
Training will output:
```
Step 1: Training YOLO Object Detector
[progress bar...]
✓ YOLO Detector training complete

Step 2: Training Leaf Variety Classifier  
[progress bar...]
✓ Classifier training complete
```

---

### OPTION 3: Manual Step-by-Step Training

#### Step A: Train YOLO Detector (optional)
Only if you have YOLO-formatted detection dataset.

```bash
python train_yolo.py \
    --data annotations/data.yaml \
    --epochs 30 \
    --batch 16 \
    --model yolov8n.pt \
    --img-size 640
```

**Arguments:**
- `--data`: Path to data.yaml
- `--epochs`: Training epochs (default: 30)
- `--batch`: Batch size (default: 16, reduce if OOM)
- `--model`: YOLO version (yolov8n.pt for small GPU ✓, yolov8s.pt for larger GPU)
- `--img-size`: Detection input size (default: 640)

**Output:** `checkpoints/yolo_leaf_detector.pt`

#### Step B: Train Classifier

```bash
python run_train.py
```

Or with custom configuration:

```python
from train import TrainingConfig, run_training

config = TrainingConfig(
    dataset_dir="./dataset",
    image_size=224,
    batch_size=8,           # RTX 4060Ti optimized
    epochs=30,
    learning_rate=3e-4,
    num_workers=2,          # RTX 4060Ti optimized
    backbone="resnet50",
    patience=5,
)

history = run_training(config)
print(f"Best F1: {history.best_macro_f1:.4f}")
print(f"Checkpoint: {history.best_checkpoint_path}")
```

**Output:** `checkpoints/best_model.pt`

---

## Advanced Configuration

### Adjust for Different GPUs

#### RTX 4090 / A100 (Very Large GPU)
```bash
python train_integrated.py \
    --data-yaml data.yaml \
    --batch-size 32 \
    --epochs 50
```

#### RTX 3070 / 4070 (Large GPU)
```bash
python train_integrated.py \
    --data-yaml data.yaml \
    --batch-size 16 \
    --epochs 40
```

#### RTX 4060Ti (8GB) - RECOMMENDED SETTINGS ✓
```bash
python train_integrated.py \
    --data-yaml data.yaml \
    --batch-size 8 \
    --epochs 30
```

#### RTX 3050 / 4050 (4GB GPU)
```bash
python train_integrated.py \
    --data-yaml data.yaml \
    --batch-size 4 \
    --epochs 25
```

### Resume Training from Checkpoint
If training was interrupted:

```bash
python train_integrated.py \
    --data-yaml data.yaml \
    --resume
```

Or:
```python
config = TrainingConfig(
    resume_from_checkpoint=True,
    # ... other settings
)
```

### Change Backbone Architecture

```bash
python train_integrated.py \
    --data-yaml data.yaml \
    --epochs 30
```

Then modify the classifier config in code:
```python
config = TrainingConfig(
    backbone="efficientnet_b0",  # Available: "resnet50", "efficientnet_b0"
    batch_size=10,  # May need to adjust per backbone
)
```

---

## Monitoring Training

### Real-time GPU Monitoring
```bash
nvidia-smi -l 1
```

### Expected Output
During training you should see:
```
Epoch 001/030 | train_loss=2.1234 train_acc=0.6543 | val_loss=1.8765 val_acc=0.7123 val_macro_f1=0.6890
Epoch 002/030 | train_loss=1.9876 train_acc=0.7012 | val_loss=1.7654 val_acc=0.7456 val_macro_f1=0.7234
...
```

### Troubleshooting

#### OOM Error (Out of Memory)
```
RuntimeError: CUDA out of memory. Tried to allocate 1.23 GB
```

**Solution:** Reduce batch size
```bash
python train_integrated.py --data-yaml data.yaml --batch-size 4
```

#### GPU Not Used
```
Using CPU (training will be VERY slow)
```

**Solution:** Verify CUDA installation
```bash
python -c "import torch; print(torch.cuda.is_available())"  # Should print: True
python -c "import torch; print(torch.cuda.get_device_name(0))"  # Should print: your GPU name
```

#### Dataset Not Found
```
FileNotFoundError: Dataset directory does not exist: ./dataset
```

**Solution:** Verify dataset path
```bash
ls dataset/  # On Linux/Mac
dir dataset  # On Windows
# Should show your variety folders
```

---

## Output Files

After training, you'll have:

```
checkpoints/
├── best_model.pt              # Best classifier checkpoint
└── yolo_leaf_detector.pt      # Best detector checkpoint (if trained)

runs/
└── detect/
    └── vine_leaf_detector/
        ├── weights/
        │   ├── best.pt        # YOLO best weights
        │   └── last.pt        # YOLO last checkpoint
        └── results.csv        # YOLO training metrics
```

### Using Trained Models

#### For Inference (Classification Only)
```python
from app import InferenceService, InferenceConfig

config = InferenceConfig(
    checkpoint_path="./checkpoints/best_model.pt",
)
service = InferenceService(config)

# Use in FastAPI server
```

#### For Inference (Full Pipeline)
```python
config = InferenceConfig(
    checkpoint_path="./checkpoints/best_model.pt",
    detector_path="./checkpoints/yolo_leaf_detector.pt",
)
service = InferenceService(config)
```

---

## Performance Metrics

### Typical Results (RTX 4060Ti)

**Training Speed:**
- Classifier: ~20-30 samples/sec
- Detector: ~30-50 images/sec
- Full pipeline (both): ~2-3 hours total

**Expected Metrics (after 30 epochs):**
- Classifier Macro F1: 0.75-0.85 (with good dataset)
- Detector mAP50: 0.70-0.85

**Memory Usage:**
- Detector training: 5-6 GB
- Classifier training: 4-5 GB
- Inference: <1 GB

---

## Quick Reference

| Task | Command |
|------|---------|
| **Quick test (classifier only)** | `python run_train.py` |
| **Full pipeline (recommended)** | `python train_integrated.py --data-yaml data.yaml` |
| **YOLO detector only** | `python train_yolo.py --data data.yaml` |
| **Monitor GPU** | `nvidia-smi -l 1` |
| **Resume training** | `python train_integrated.py --data-yaml data.yaml --resume` |
| **Custom batch size** | `python train_integrated.py --data-yaml data.yaml --batch-size 6` |

---

## Summary

✅ **For RTX 4060Ti, use:**
```bash
python train_integrated.py --data-yaml annotations/data.yaml
```

This automatically configures:
- Batch Size: 8
- Workers: 2  
- Mixed Precision: Enabled
- Epochs: 30

The system will:
1. First train YOLO detector (if data.yaml provided)
2. Then train classifier  
3. Save best models to `checkpoints/`

**Total time:** ~2-3 hours for complete training on RTX 4060Ti

---

**Questions?** Check the code comments in:
- `train_integrated.py` - Main training pipeline
- `train.py` - Classifier training details
- `train_yolo.py` - Detector training details
