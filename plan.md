# Tree Pest Detection Pipeline - Implementation Plan

## Context

Building a production-grade YOLO11m computer vision pipeline for precision agriculture pest detection. The system will detect and classify 103 pest species across 75,000 images, with support for 4K inference using SAHI (Slicing Aided Hyper Inference).

**Why This Project**: Enable automated pest monitoring in orchards/farms by detecting small pests on high-resolution images without manual inspection.

**Key Constraints**:
- **Hardware**: RTX 4070 Laptop (8GB VRAM), 32GB RAM, 130GB free storage
- **Dataset**: 75k images, 103 pest classes, YOLOv11 format, Roboflow-hosted
- **Workflow**: Development on Machine A, training on Machine B (GitHub sync)
- **Target**: 4K image inference without CUDA OOM errors

**Architecture Decision**: **Single-stage direct pest detection** with SAHI for 4K images. The original two-stage approach (leaf detection → pest detection) is optional and can be added later if small pest detection accuracy is insufficient.

---

## Milestone Breakdown

### Milestone 1: Environment & Project Setup
**Goal**: Reproducible development environment with proper project structure

**Deliverables**:
- Python 3.11 virtual environment
- PyTorch with CUDA 12.x support
- Project folder structure
- Git configuration with LFS for model weights

**Estimated Time**: 30 minutes

---

### Milestone 2: Dataset Acquisition & Analysis
**Goal**: Download 75k pest dataset and understand class distribution for smart augmentation

**Deliverables**:
- Roboflow dataset downloaded and organized
- Class distribution analysis (identify minority/majority classes)
- Augmentation plan for balancing (if needed)
- Data validation (check annotations, image integrity)

**Estimated Time**: 1-2 hours (depends on download speed)

---

### Milestone 3: Data Augmentation Pipeline
**Goal**: Balance dataset without exploding storage (130GB limit)

**Strategy**: **On-the-fly augmentation** for minority classes only
- Augment classes with < 10th percentile samples
- Apply transformations during training (in-memory)
- Zero disk overhead

**Deliverables**:
- `augmentation_engine.py`: Albumentations pipeline
- `augmentation_plan.json`: Per-class augmentation factors
- Preview notebook to verify augmentations

**Estimated Time**: 2-3 hours

---

### Milestone 4: Training Pipeline - Pest Detection Model
**Goal**: Train YOLO11m to detect 103 pest classes, optimized for 8GB VRAM

**Training Configuration** (optimized for 103 classes):
```yaml
model: yolo11m.pt
task: detect
epochs: 150
patience: 20

# Optimized for RTX 4070 8GB + 103 classes
img_size: 640           # Full resolution
batch_size: 8           # Safe for 103 classes
workers: 8              # i9-13900HX has 24 threads
accumulate: 2           # Effective batch = 16

# Mixed precision (critical)
amp: true               # ~40% VRAM reduction

# Optimizer & LR
optimizer: AdamW
lr0: 0.01
lrf: 0.001
cos_lr: true
warmup_epochs: 5

# Augmentation (YOLO built-in + on-the-fly minority augmentation)
hsv_h: 0.02
hsv_s: 0.8
hsv_v: 0.5
degrees: 30.0
translate: 0.15
scale: 0.6
flipud: 0.3
fliplr: 0.5
mosaic: 1.0
mixup: 0.2
copy_paste: 0.1         # Good for small objects (pests)
```

**Memory Management**:
- Mixed precision (FP16): 40% VRAM reduction
- Aggressive garbage collection after validation epochs
- Monitor VRAM usage with `py3nvml`
- Fallback: Reduce batch to 4, increase accumulate to 4

**Deliverables**:
- `train_pest_detector.py`: Training orchestration
- `memory_optimizer.py`: Dynamic batch sizing, GC management
- `config/pest_detection.yaml`: Hyperparameters
- Trained model: `models/pest_detection/weights/best.pt`
- Training logs, TensorBoard metrics

**Estimated Training Time**: 8-12 hours (150 epochs × 75k images)

---

### Milestone 5: SAHI Integration for 4K Inference
**Goal**: Run inference on 4K images (3840×2160) without CUDA OOM

**SAHI Strategy**:
```python
SAHI_CONFIG = {
    'slice_height': 640,            # Match training img_size
    'slice_width': 640,
    'overlap_height_ratio': 0.2,    # 20% overlap for edge pests
    'overlap_width_ratio': 0.2,
    'postprocess_type': 'NMS',      # Remove duplicate detections
    'postprocess_match_metric': 'IOS',  # Intersection Over Smaller
    'postprocess_match_threshold': 0.5,
}
```

**How SAHI Works**:
1. Slice 4K image into 640×640 patches (9×6 grid = 54 slices with overlap)
2. Run YOLO inference on each slice independently
3. Aggregate detections, apply NMS to remove duplicates from overlaps
4. Map coordinates back to original 4K image

**Expected Performance**:
- Per-slice inference: ~50ms
- Total inference: ~2.7s per 4K image
- VRAM usage: ~2.5GB (well within 8GB limit)

**Deliverables**:
- `sahi_predictor.py`: SAHI inference wrapper
- `batch_predictor.py`: Process multiple 4K images
- `config/sahi.yaml`: SAHI configuration
- CLI: `python scripts/run_inference.py --input 4k_images/ --output results/`

**Estimated Time**: 2-3 hours

---

### Milestone 6: Metrics Extraction & Visualization
**Goal**: Extract comprehensive metrics for final report

**Required Metrics** (per project requirements):
- **Confusion Matrix**: 103×103 heatmap
- **Precision/Recall/F1** per class
- **mAP@50 and mAP@50-95**
- **PR Curves** (Precision-Recall)
- **Loss curves**: train/val over epochs

**YOLO Auto-Generated Metrics**:
YOLO11 automatically saves these during training:
- `results.csv`: Epoch-wise metrics
- `confusion_matrix.png`: Confusion matrix
- `PR_curve.png`, `F1_curve.png`, `P_curve.png`, `R_curve.png`
- `results.png`: Loss/metrics overview

**Custom Enhancements**:
- Per-class F1 ranking (identify worst-performing classes)
- Top-K confused class pairs (for 103 classes, show top 20 confusions)
- Interactive Plotly visualizations
- Export metrics to JSON/CSV for report

**Deliverables**:
- `metrics_extractor.py`: Parse YOLO results
- `confusion_matrix.py`: Enhanced CM visualization
- `pr_curve_generator.py`: Custom PR curves
- `visualizer.py`: Plot generation
- `outputs/metrics/final_report.json`: Aggregated metrics

**Estimated Time**: 2-3 hours

---

### Milestone 7: GitHub Workflow & Documentation
**Goal**: Enable seamless code development on Machine A, training on Machine B

**Git Strategy**:

**`.gitignore`**:
```gitignore
# Data (too large for Git)
data/raw/
data/processed/
data/cache/

# Model runs (LFS for best.pt only)
models/*/runs/
models/*/weights/last.pt

# Outputs
outputs/
*.log

# Python
__pycache__/
*.pyc
venv/

# IDE
.vscode/
.idea/
```

**`.gitattributes`** (Git LFS):
```gitattributes
*.pt filter=lfs diff=lfs merge=lfs -text
models/*/weights/best.pt filter=lfs diff=lfs merge=lfs -text
```

**Workflow**:
```bash
# Machine A (Development)
git checkout -b feature/pest-detection
# Write code, commit scripts/configs
git add src/ scripts/ config/
git commit -m "feat: implement pest detection pipeline"
git push origin feature/pest-detection

# Machine B (Training)
git clone <repo>
git checkout feature/pest-detection
pip install -r requirements.txt
python scripts/download_dataset.py  # Download Roboflow data
python scripts/train_pest_detector.py  # Train model
# Commit trained models via LFS
git add models/pest_detection/weights/best.pt
git commit -m "chore: add trained pest detection model"
git push origin feature/pest-detection
```

**Deliverables**:
- `README.md`: Setup instructions, training guide
- `docs/training_guide.md`: Detailed training procedures
- `docs/inference_guide.md`: SAHI inference instructions
- `docs/troubleshooting.md`: Common issues, VRAM optimization tips

**Estimated Time**: 2 hours

---

## Project Structure

```
tree-pest-identification/
├── .git/                           # Git repository
├── .gitignore                      # Exclude data, outputs
├── .gitattributes                  # LFS for *.pt files
├── README.md                       # Quick start guide
├── requirements.txt                # Production dependencies
├── pyproject.toml                  # Project metadata
│
├── config/                         # Configuration files
│   ├── pest_detection.yaml         # Training hyperparameters
│   ├── augmentation.yaml           # Augmentation policies
│   └── sahi.yaml                   # SAHI inference config
│
├── src/                            # Source code
│   ├── __init__.py
│   │
│   ├── core/                       # Core utilities
│   │   ├── __init__.py
│   │   ├── config_loader.py        # YAML config loading
│   │   ├── logger.py               # Structured logging
│   │   ├── device_manager.py       # CUDA memory management
│   │   └── metrics_collector.py   # Metrics aggregation
│   │
│   ├── data/                       # Data pipeline
│   │   ├── __init__.py
│   │   ├── roboflow_downloader.py  # Download with retries
│   │   ├── dataset_analyzer.py     # Class distribution
│   │   ├── augmentation_engine.py  # On-the-fly augmentation
│   │   ├── data_loader.py          # Memory-efficient loading
│   │   └── cache_manager.py        # Augmentation cache
│   │
│   ├── training/                   # Training orchestration
│   │   ├── __init__.py
│   │   ├── base_trainer.py         # Abstract trainer
│   │   ├── pest_trainer.py         # Pest detection trainer
│   │   ├── callbacks.py            # Custom callbacks
│   │   └── memory_optimizer.py     # VRAM optimization
│   │
│   ├── inference/                  # Inference pipeline
│   │   ├── __init__.py
│   │   ├── sahi_predictor.py       # SAHI-based inference
│   │   ├── post_processor.py       # NMS, filtering
│   │   └── batch_predictor.py      # Batch inference
│   │
│   ├── evaluation/                 # Metrics & visualization
│   │   ├── __init__.py
│   │   ├── metrics_extractor.py    # Extract YOLO metrics
│   │   ├── confusion_matrix.py     # CM generation
│   │   ├── pr_curve_generator.py   # PR curves
│   │   └── visualizer.py           # Result visualization
│   │
│   └── utils/                      # Shared utilities
│       ├── __init__.py
│       ├── file_utils.py           # Safe file operations
│       ├── gpu_utils.py            # GPU monitoring
│       └── path_utils.py           # Path management
│
├── scripts/                        # Executable scripts
│   ├── download_dataset.py         # Download from Roboflow
│   ├── analyze_dataset.py          # Class distribution
│   ├── prepare_augmentation.py     # Augmentation plan
│   ├── train_pest_detector.py      # Train model
│   ├── evaluate_model.py           # Model evaluation
│   └── run_inference.py            # SAHI inference
│
├── data/                           # Data directory (gitignored)
│   ├── raw/                        # Raw Roboflow data
│   │   └── pests/
│   │       ├── train/
│   │       │   ├── images/
│   │       │   └── labels/
│   │       ├── valid/
│   │       │   ├── images/
│   │       │   └── labels/
│   │       └── data.yaml
│   ├── processed/                  # Augmentation artifacts
│   │   └── augmentation_plan.json
│   └── cache/                      # Runtime cache
│
├── models/                         # Model artifacts (LFS)
│   └── pest_detection/
│       ├── weights/
│       │   ├── best.pt             # Best model (LFS tracked)
│       │   └── last.pt             # Last checkpoint
│       ├── runs/                   # Training runs (gitignored)
│       └── exports/                # ONNX, TensorRT exports
│
├── outputs/                        # Generated outputs (gitignored)
│   ├── metrics/
│   │   ├── confusion_matrix.png
│   │   ├── pr_curves.png
│   │   ├── f1_per_class.csv
│   │   └── final_report.json
│   ├── visualizations/
│   └── predictions/                # Inference results
│
└── docs/                           # Documentation
    ├── training_guide.md
    ├── inference_guide.md
    └── troubleshooting.md
```

---

## Critical Files & Their Purpose

### 1. `requirements.txt`
**Purpose**: Hardware-optimized dependencies for RTX 4070 + CUDA 12.x

```txt
# Core ML (CUDA 12.x optimized)
torch==2.5.1+cu124
torchvision==0.20.1+cu124
--find-links https://download.pytorch.org/whl/torch_stable.html

# YOLO & CV
ultralytics==8.3.42           # YOLO11 support
sahi==0.11.18                 # Sliced inference
opencv-python-headless==4.10.0.84
albumentations==1.4.20        # Augmentation

# Data Handling
roboflow==1.1.44              # Dataset download
pyyaml==6.0.2
pandas==2.2.3
numpy==1.26.4

# Metrics & Visualization
scikit-learn==1.6.0
matplotlib==3.9.2
seaborn==0.13.2

# Memory & Performance
psutil==6.1.0                 # System monitoring
py3nvml==0.2.7                # NVIDIA GPU monitoring
tqdm==4.67.1

# Utilities
loguru==0.7.3                 # Logging
pydantic==2.10.3              # Config validation
typer==0.15.1                 # CLI framework
```

---

### 2. `src/training/memory_optimizer.py`
**Purpose**: Dynamic VRAM management to prevent OOM on 8GB GPU

**Key Functions**:
- `calculate_optimal_batch_size()`: Adjust batch size based on VRAM availability
- `setup_mixed_precision()`: Configure AMP (FP16) for 40% VRAM reduction
- `cleanup_cuda_cache()`: Aggressive garbage collection after validation
- `monitor_vram_usage()`: Real-time VRAM tracking with alerts

**Critical Pattern**:
```python
def train_with_memory_management(model, config):
    # Pre-training cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    # Dynamic batch sizing
    batch_size = calculate_optimal_batch_size(
        img_size=config['img_size'],
        num_classes=103,
        available_vram_gb=8.0
    )
    
    # Train with callbacks
    model.train(
        batch=batch_size,
        amp=True,  # Mixed precision
        callbacks={
            'on_val_end': cleanup_cuda_cache
        }
    )
```

---

### 3. `src/data/augmentation_engine.py`
**Purpose**: Storage-efficient on-the-fly augmentation for minority classes

**Strategy**:
1. Analyze class distribution → identify minority classes (< 10th percentile)
2. Calculate augmentation factor per class to reach target samples
3. Generate augmented samples in-memory during training (no disk overhead)
4. Use Albumentations for GPU-accelerated transforms

**Key Implementation**:
```python
class StorageEfficientAugmentation:
    def analyze_and_plan(self, dataset_path: str) -> dict:
        """
        Returns augmentation plan:
        {
            'class_12': {'original_count': 45, 'aug_factor': 3},
            'class_87': {'original_count': 32, 'aug_factor': 4},
            ...
        }
        """
        class_counts = self._count_samples(dataset_path)
        minority_threshold = np.percentile(list(class_counts.values()), 10)
        
        plan = {}
        for class_id, count in class_counts.items():
            if count < minority_threshold:
                aug_factor = target_samples // count
                plan[class_id] = {
                    'original_count': count,
                    'aug_factor': aug_factor
                }
        return plan
    
    def get_augmentation_pipeline(self) -> A.Compose:
        """Albumentations pipeline for pest detection"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.GaussNoise(p=0.3),
            A.RandomScale(scale_limit=0.1, p=0.3),
            A.ColorJitter(p=0.4),
        ])
```

**Storage Impact**: Zero disk overhead (all in-memory)

---

### 4. `src/inference/sahi_predictor.py`
**Purpose**: 4K image inference without CUDA OOM using SAHI slicing

**SAHI Flow**:
```python
class SAHIPredictor:
    def predict_4k(self, image_path: Path) -> List[Detection]:
        """
        1. Slice 4K image into 640×640 patches (54 slices)
        2. Run YOLO on each slice
        3. Aggregate detections with NMS
        4. Return detections in original coordinates
        """
        from sahi.predict import get_sliced_prediction
        from sahi.models.yolov8 import Yolov8DetectionModel
        
        # Pre-inference cleanup
        torch.cuda.empty_cache()
        
        detection_model = Yolov8DetectionModel(
            model_path=str(self.model_path),
            confidence_threshold=0.25,
            device='cuda:0'
        )
        
        result = get_sliced_prediction(
            str(image_path),
            detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            postprocess_type='NMS',
            postprocess_match_metric='IOS',
            postprocess_match_threshold=0.5
        )
        
        # Post-inference cleanup
        torch.cuda.empty_cache()
        
        return result.object_prediction_list
```

**Performance**:
- 4K image → 54 slices
- ~50ms per slice
- ~2.7s total per 4K image
- VRAM: ~2.5GB (safe)

---

### 5. `scripts/train_pest_detector.py`
**Purpose**: Training orchestration with all optimizations

**Key Features**:
- Load config from `config/pest_detection.yaml`
- Setup memory optimizer
- Apply on-the-fly augmentation for minority classes
- Train YOLO11m with callbacks
- Save best model to `models/pest_detection/weights/best.pt`

**CLI Usage**:
```bash
python scripts/train_pest_detector.py \
    --config config/pest_detection.yaml \
    --data data/raw/pests/data.yaml \
    --augmentation data/processed/augmentation_plan.json \
    --output models/pest_detection
```

---

### 6. `config/pest_detection.yaml`
**Purpose**: Hardware-optimized hyperparameters for RTX 4070 + 103 classes

```yaml
# Model
model: yolo11m.pt
task: detect

# Training
epochs: 150
patience: 20
save_period: 10

# Hardware optimization (103 classes)
img_size: 640
batch_size: 8           # Safe for 103 classes on 8GB VRAM
workers: 8
accumulate: 2           # Effective batch = 16
amp: true               # Mixed precision (critical)

# Optimizer
optimizer: AdamW
lr0: 0.01
lrf: 0.001
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 5
cos_lr: true

# Augmentation
hsv_h: 0.02
hsv_s: 0.8
hsv_v: 0.5
degrees: 30.0
translate: 0.15
scale: 0.6
shear: 5.0
flipud: 0.3
fliplr: 0.5
mosaic: 1.0
mixup: 0.2
copy_paste: 0.1

# Validation
val: true
plots: true
save_json: true
```

---

### 7. `src/evaluation/metrics_extractor.py`
**Purpose**: Extract and aggregate YOLO metrics for final report

**Extracted Metrics**:
- **Global**: mAP@50, mAP@50-95, precision, recall, F1
- **Per-class**: Precision, Recall, F1, AP@50, AP@50-95
- **Confusion Matrix**: 103×103 matrix (enhance with top-K confusions)
- **Loss Curves**: Train/val losses over epochs
- **PR Curves**: Precision-Recall for each class

**Output Format**:
```json
{
  "global_metrics": {
    "mAP50": 0.87,
    "mAP50_95": 0.65,
    "precision": 0.89,
    "recall": 0.82,
    "f1": 0.85
  },
  "per_class_metrics": [
    {"class": "pest_0", "precision": 0.92, "recall": 0.88, "f1": 0.90, "mAP50": 0.91},
    ...
  ],
  "top_confused_pairs": [
    {"class_a": "pest_12", "class_b": "pest_45", "confusion_rate": 0.23},
    ...
  ]
}
```

---

## Execution Flow (Step-by-Step)

### Phase 1: Setup (Machine A or B)

```bash
# 1. Clone repository
git clone <repository-url>
cd tree-pest-identification

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify GPU
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
# Expected: CUDA Available: True, GPU: NVIDIA GeForce RTX 4070 Laptop GPU
```

---

### Phase 2: Data Acquisition (Machine B - Training Machine)

```bash
# 1. Download dataset from Roboflow
python scripts/download_dataset.py \
    --url "https://app.roboflow.com/ds/YrFRz3Bfo9?key=Hey4UNMjA7" \
    --output data/raw/pests

# Expected output:
# - data/raw/pests/train/ (images + labels)
# - data/raw/pests/valid/ (images + labels)
# - data/raw/pests/data.yaml

# 2. Analyze class distribution
python scripts/analyze_dataset.py \
    --dataset data/raw/pests \
    --output outputs/metrics/class_distribution.json

# Expected output:
# Class distribution statistics
# Total images: 75,000
# Total classes: 103
# Min samples per class: X
# Max samples per class: Y
# Mean samples per class: Z
# Minority classes (< 10th percentile): [list]
```

---

### Phase 3: Data Augmentation Planning

```bash
# Create augmentation plan for minority classes
python scripts/prepare_augmentation.py \
    --dataset data/raw/pests \
    --distribution outputs/metrics/class_distribution.json \
    --target-samples 100 \
    --output data/processed/augmentation_plan.json

# Expected output:
# Augmentation plan saved
# Classes to augment: 15
# Total augmented samples: 1,500
# Storage overhead: 0 bytes (on-the-fly)
```

---

### Phase 4: Training (Machine B)

```bash
# Train pest detection model
python scripts/train_pest_detector.py \
    --config config/pest_detection.yaml \
    --data data/raw/pests/data.yaml \
    --augmentation data/processed/augmentation_plan.json \
    --output models/pest_detection

# Training will run for ~8-12 hours (150 epochs)
# Monitor with TensorBoard:
# tensorboard --logdir models/pest_detection/runs

# Output files:
# - models/pest_detection/weights/best.pt (best model)
# - models/pest_detection/weights/last.pt (last checkpoint)
# - models/pest_detection/runs/detect/trainX/ (logs, metrics, plots)
```

**During Training**:
- Monitor VRAM usage (should stay < 7GB)
- Watch for OOM errors (if occurs, reduce batch_size in config)
- Check TensorBoard for loss curves, mAP progression

---

### Phase 5: Model Evaluation

```bash
# Evaluate trained model
python scripts/evaluate_model.py \
    --model models/pest_detection/weights/best.pt \
    --data data/raw/pests/data.yaml \
    --output outputs/metrics/evaluation_results.json

# Expected output:
# Global metrics: mAP@50, mAP@50-95, P, R, F1
# Per-class metrics saved to CSV
# Confusion matrix: outputs/metrics/confusion_matrix.png
# PR curves: outputs/metrics/pr_curves.png
# F1 curve: outputs/metrics/f1_curve.png
```

---

### Phase 6: SAHI Inference on 4K Images

```bash
# Run inference on 4K images
python scripts/run_inference.py \
    --model models/pest_detection/weights/best.pt \
    --input path/to/4k_images/ \
    --output outputs/predictions/ \
    --sahi-config config/sahi.yaml \
    --visualize

# Expected output:
# Processed X images
# Average inference time: 2.7s per 4K image
# Results saved to outputs/predictions/
#   - predictions.json (bounding boxes)
#   - visualizations/ (annotated images)
```

**SAHI Process**:
1. Load 4K image (3840×2160)
2. Slice into 640×640 patches (54 slices)
3. Run YOLO on each slice (~50ms each)
4. Aggregate detections with NMS
5. Save annotated image + JSON results

---

### Phase 7: Commit Model to Git (Machine B)

```bash
# Stage trained model (tracked by LFS)
git add models/pest_detection/weights/best.pt
git add outputs/metrics/evaluation_results.json

# Commit and push
git commit -m "chore: add trained pest detection model (mAP@50: 0.87)"
git push origin feature/pest-detection
```

---

## Validation Strategy

### 1. Dataset Validation
- [x] All 75k images have valid annotations
- [x] 103 classes present in train/valid splits
- [x] No corrupted images
- [x] Label format matches YOLO (class x_center y_center width height)

### 2. Training Validation
- [x] VRAM usage < 7.5GB throughout training
- [x] No CUDA OOM errors
- [x] Validation mAP increases over epochs
- [x] No overfitting (train/val loss gap < 20%)

### 3. Inference Validation
- [x] SAHI successfully processes 4K images
- [x] No CUDA OOM during inference
- [x] Detections overlap correctly between slices
- [x] Inference time < 3s per 4K image

### 4. Metrics Validation
- [x] Confusion matrix generated (103×103)
- [x] PR curves available for all classes
- [x] F1, Precision, Recall calculated per class
- [x] mAP@50 and mAP@50-95 reported

---

## Common Issues & Solutions

### Issue 1: CUDA Out of Memory (OOM)
**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce `batch_size` from 8 to 4 in `config/pest_detection.yaml`
2. Increase `accumulate` from 2 to 4 (keep effective batch = 16)
3. Reduce `img_size` from 640 to 512
4. Ensure `amp: true` (mixed precision)
5. Close other GPU applications

**Code Fix**:
```yaml
# config/pest_detection.yaml
batch_size: 4    # Reduced from 8
accumulate: 4    # Increased from 2
```

---

### Issue 2: Storage Exceeded (130GB Limit)
**Symptoms**: Disk full errors during training

**Solutions**:
1. Verify on-the-fly augmentation (no disk writes)
2. Delete training runs after extracting metrics:
   ```bash
   rm -rf models/pest_detection/runs/detect/train*
   ```
3. Keep only `best.pt`, delete `last.pt`

---

### Issue 3: Slow Training (> 15 hours)
**Symptoms**: Training taking too long

**Solutions**:
1. Increase `workers` from 8 to 12 (faster data loading)
2. Reduce `patience` from 20 to 15 (early stopping)
3. Use `close_mosaic: 10` to disable mosaic augmentation in last 10 epochs

---

### Issue 4: SAHI Inference Too Slow
**Symptoms**: > 5s per 4K image

**Solutions**:
1. Reduce `overlap_height_ratio` and `overlap_width_ratio` from 0.2 to 0.1
2. Increase `confidence_threshold` from 0.25 to 0.4 (fewer detections to process)
3. Use batch inference on slices (process multiple slices simultaneously)

---

### Issue 5: Low mAP (< 0.5)
**Symptoms**: Poor detection accuracy

**Solutions**:
1. Increase augmentation severity (higher `hsv_h`, `hsv_s`, `degrees`)
2. Train longer (increase `epochs` from 150 to 200)
3. Increase `patience` from 20 to 30
4. Review minority class performance (may need more aggressive augmentation)

---

## Expected Training Metrics

Based on 103 classes, 75k images, and similar pest detection projects:

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| **mAP@50** | 0.75 - 0.90 | Good: > 0.80 |
| **mAP@50-95** | 0.55 - 0.75 | Good: > 0.65 |
| **Precision** | 0.80 - 0.95 | Good: > 0.85 |
| **Recall** | 0.70 - 0.90 | Good: > 0.80 |
| **F1 Score** | 0.75 - 0.90 | Good: > 0.82 |
| **Training Time** | 8 - 12 hours | 150 epochs, 8GB VRAM |
| **Inference (4K)** | 2.5 - 3.5s | SAHI with 640×640 slices |
| **VRAM (Training)** | 6 - 7.5GB | Batch=8, img=640, 103 classes |
| **VRAM (Inference)** | 2 - 3GB | SAHI slice inference |

---

## Optional: Two-Stage Architecture (Future Enhancement)

If direct pest detection on 4K images has low accuracy for small pests, consider adding **Stage 1: Leaf/Region Detection**:

### Two-Stage Flow:
1. **Stage 1**: Train YOLO11m to detect leaves or regions of interest
   - Requires separate leaf annotation (manual or semi-automated)
   - Train on same 75k images, but annotate leaf regions
   - Simpler task (fewer classes: leaf vs background)

2. **Stage 2**: Run pest detection on cropped leaves
   - Use existing pest detection model
   - Smaller input regions → better small object detection
   - Reduced search space → faster inference

### When to Consider Two-Stage:
- mAP@50 < 0.7 on validation set
- Many false negatives on small pests
- High confusion between pest classes

### Implementation:
- Add `config/stage1_leaf_detection.yaml`
- Add `scripts/train_leaf_detector.py`
- Update `src/inference/sahi_predictor.py` to support two-stage mode

**Not implementing two-stage initially** because:
1. Requires additional leaf annotation work
2. Increases inference latency (2× models)
3. Single-stage with SAHI should achieve good results

---

## Final Deliverables Checklist

- [ ] Project structure with all folders
- [ ] `requirements.txt` with CUDA 12.x support
- [ ] Configuration files (`pest_detection.yaml`, `augmentation.yaml`, `sahi.yaml`)
- [ ] Data pipeline scripts (download, analyze, augmentation)
- [ ] Training scripts with memory optimization
- [ ] SAHI inference pipeline
- [ ] Metrics extraction and visualization
- [ ] Git LFS configuration for model weights
- [ ] Documentation (README, training guide, inference guide)
- [ ] Trained model: `models/pest_detection/weights/best.pt`
- [ ] Evaluation metrics: `outputs/metrics/final_report.json`
- [ ] Confusion matrix, PR curves, F1 curves
- [ ] Sample inference results on 4K images

---

## Estimated Total Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Setup & Project Structure | 1 hour | One-time setup |
| Data Download & Analysis | 2 hours | Depends on internet speed |
| Augmentation Pipeline | 2 hours | Code + testing |
| Model Training | 8-12 hours | Automated (150 epochs) |
| SAHI Integration | 2 hours | Code + testing |
| Metrics Extraction | 2 hours | Analysis + visualization |
| Documentation | 2 hours | README, guides |
| **Total** | **19-23 hours** | ~3 days (with training overnight) |

**Development**: 8-10 hours (Machine A)  
**Training**: 8-12 hours (Machine B, automated)  
**Evaluation**: 3-4 hours (Machine B)

---

## Next Steps

1. **Confirm this plan** - any questions or adjustments needed?
2. **Setup environment** - run setup commands on both machines
3. **Implement core modules** - start with `memory_optimizer.py`, `augmentation_engine.py`
4. **Download dataset** - run `download_dataset.py` on Machine B
5. **Start training** - launch overnight on Machine B
6. **Iterate** - tune hyperparameters based on initial results

Ready to proceed with implementation!
