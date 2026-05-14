# Tree Pest Detection Pipeline

Production-grade YOLO11m computer vision pipeline for detecting and classifying 103 pest species in precision agriculture. Optimized for RTX 4070 Laptop (8GB VRAM) with SAHI support for 4K inference.

## Features

- **YOLO11m Detection**: State-of-the-art object detection for 103 pest classes
- **Memory Optimized**: Dynamic batch sizing, gradient accumulation, mixed precision (AMP) for 8GB VRAM
- **Storage Efficient**: On-the-fly augmentation (zero disk overhead) for minority classes
- **4K Inference**: SAHI (Slicing Aided Hyper Inference) for high-resolution images without OOM
- **Production Ready**: Enterprise-grade code with logging, monitoring, error handling

## Project Structure

```
tree-pest-identification/
├── config/                    # Configuration files
│   ├── pest_detection.yaml    # Training hyperparameters
│   ├── augmentation.yaml      # Augmentation policies
│   └── sahi.yaml              # SAHI inference config
├── src/                       # Source code
│   ├── core/                  # Core utilities
│   ├── data/                  # Data pipeline
│   ├── training/              # Training orchestration
│   ├── inference/             # SAHI inference
│   └── evaluation/            # Metrics extraction
├── scripts/                   # Executable scripts
│   ├── download_dataset.py    # Download from Roboflow
│   ├── analyze_dataset.py     # Class distribution analysis
│   ├── prepare_augmentation.py # Augmentation planning
│   ├── train_pest_detector.py  # Training
│   ├── evaluate_model.py      # Model evaluation
│   └── run_inference.py       # SAHI inference
└── plan.md                    # Detailed implementation plan

```

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2. Download Dataset

```bash
# Download 75k pest images from Roboflow
python scripts/download_dataset.py \
    --url "https://app.roboflow.com/ds/YrFRz3Bfo9?key=YOUR_KEY" \
    --output data/raw/pests
```

### 3. Analyze Dataset

```bash
# Analyze class distribution
python scripts/analyze_dataset.py \
    --dataset data/raw/pests \
    --output outputs/metrics/class_distribution.json
```

### 4. Prepare Augmentation

```bash
# Create augmentation plan for minority classes
python scripts/prepare_augmentation.py \
    --dataset data/raw/pests \
    --distribution outputs/metrics/class_distribution.json \
    --output data/processed/augmentation_plan.json
```

### 5. Train Model

```bash
# Train YOLO11m (8-12 hours on RTX 4070)
python scripts/train_pest_detector.py \
    --config config/pest_detection.yaml \
    --data data/raw/pests/data.yaml \
    --augmentation data/processed/augmentation_plan.json
```

**Training Configuration (Optimized for 8GB VRAM + 103 classes):**
- Image size: 640×640
- Batch size: 8 (effective batch: 16 with gradient accumulation)
- Mixed precision (AMP): Enabled (~40% VRAM reduction)
- Epochs: 150 with early stopping (patience=20)

### 6. Evaluate Model

```bash
# Extract metrics (mAP, precision, recall, F1)
python scripts/evaluate_model.py \
    --model models/pest_detection/train/weights/best.pt \
    --data data/raw/pests/data.yaml \
    --output outputs/metrics/evaluation_results.json
```

### 7. Run Inference on 4K Images

```bash
# SAHI inference on high-resolution images
python scripts/run_inference.py \
    --model models/pest_detection/train/weights/best.pt \
    --input path/to/4k_images/ \
    --output outputs/predictions/
```

**SAHI Configuration:**
- Slice size: 640×640 (matches training)
- Overlap: 20% (prevents missing objects at boundaries)
- Expected performance: ~2.7s per 4K image
- VRAM usage: ~2.5GB (safe for 8GB GPU)

## Hardware Requirements

- **GPU**: NVIDIA RTX 4070 Laptop (8GB VRAM) or equivalent
- **RAM**: 32GB
- **Storage**: 130GB free space
- **CUDA**: 12.x

## Key Optimizations

### Memory Management (8GB VRAM)
- **Mixed Precision (AMP)**: 40% VRAM reduction with minimal accuracy impact
- **Dynamic Batch Sizing**: Automatically adjusts based on VRAM availability
- **Gradient Accumulation**: Simulates larger batches (batch=8, accumulate=2 → effective=16)
- **Aggressive Cleanup**: CUDA cache cleared after validation epochs

### Storage Efficiency (130GB Constraint)
- **On-the-Fly Augmentation**: All augmentation happens in-memory during training
- **Minority Class Focus**: Only augment underrepresented classes (< 10th percentile)
- **Zero Disk Overhead**: No duplicate images saved to disk

### SAHI for 4K Inference
- **Slicing**: 4K image (3840×2160) → 54 slices (640×640 each)
- **Overlap**: 20% to catch objects at slice boundaries
- **NMS Post-Processing**: Removes duplicate detections
- **Performance**: ~50ms per slice, ~2.7s total per 4K image

## Expected Results

Based on 103 classes, 75k images:

| Metric | Expected Range | Good Threshold |
|--------|---------------|----------------|
| mAP@50 | 0.75 - 0.90 | > 0.80 |
| mAP@50-95 | 0.55 - 0.75 | > 0.65 |
| Precision | 0.80 - 0.95 | > 0.85 |
| Recall | 0.70 - 0.90 | > 0.80 |
| F1 Score | 0.75 - 0.90 | > 0.82 |

**Training Time**: 8-12 hours (150 epochs)  
**Inference**: 2.5-3.5s per 4K image

## Troubleshooting

### CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce `batch_size` from 8 to 4 in `config/pest_detection.yaml`
2. Increase `accumulate` from 2 to 4
3. Reduce `img_size` from 640 to 512
4. Ensure `amp: true` is enabled

```yaml
# config/pest_detection.yaml
batch_size: 4    # Reduced
accumulate: 4    # Increased
```

### Storage Exceeded

**Symptoms**: Disk full during training

**Solutions**:
1. Verify augmentation plan uses on-the-fly mode (check `augmentation_plan.json`)
2. Delete old training runs: `rm -rf models/pest_detection/train*`
3. Keep only `best.pt`, delete `last.pt`

### Slow Training

**Symptoms**: > 15 hours for 150 epochs

**Solutions**:
1. Increase `workers` from 8 to 12
2. Reduce `patience` from 20 to 15
3. Use `close_mosaic: 10` in config

## Git Workflow (Development + Training on Separate Machines)

```bash
# Machine A (Development)
git add src/ scripts/ config/
git commit -m "feat: implement pest detection pipeline"
git push origin main

# Machine B (Training)
git clone <repository>
git pull origin main
pip install -r requirements.txt
python scripts/download_dataset.py --url <roboflow-url>
python scripts/train_pest_detector.py --config config/pest_detection.yaml --data data/raw/pests/data.yaml

# Commit trained model (via Git LFS)
git add models/pest_detection/train/weights/best.pt
git commit -m "chore: add trained model (mAP@50: 0.87)"
git push origin main
```

## File Paths Reference

- **Dataset**: `data/raw/pests/` (gitignored)
- **Config**: `config/pest_detection.yaml`
- **Augmentation Plan**: `data/processed/augmentation_plan.json`
- **Trained Models**: `models/pest_detection/train/weights/best.pt` (Git LFS)
- **Metrics**: `outputs/metrics/`
- **Predictions**: `outputs/predictions/`

## Next Steps

1. **Train model** on Machine B (8-12 hours)
2. **Evaluate** with `evaluate_model.py` (extract confusion matrix, PR curves, F1 scores)
3. **Run inference** on 4K test images with SAHI
4. **Analyze results** - identify minority classes needing more data
5. **Iterate** - adjust augmentation or hyperparameters if mAP < 0.70

## Documentation

- **Detailed Plan**: See `plan.md` for complete implementation strategy
- **Training Guide**: See `docs/training_guide.md` (to be created)
- **SAHI Guide**: See `docs/inference_guide.md` (to be created)

## License

MIT License - See LICENSE file

## Acknowledgments

- **YOLO11**: Ultralytics YOLO11m
- **SAHI**: Slicing Aided Hyper Inference by Obss
- **Dataset**: Roboflow pest detection dataset

---

**Contact**: Illia - Master's in Software Engineering  
**Project**: Precision Agriculture Pest Detection  
**Hardware**: RTX 4070 Laptop (8GB VRAM), 32GB RAM
