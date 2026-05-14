# Training Machine Setup Guide (Machine B)

**Purpose**: Complete guide for second developer to run training on the dedicated training machine.

**Project**: Tree Pest Detection using YOLO11m  
**Dataset**: 75,000 pest images, 103 classes  
**Expected Training Time**: 30 minutes - 12 hours (depending on mode)

---

## 📋 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA RTX 4070 Laptop (8GB VRAM) or equivalent
- **RAM**: 32GB
- **Storage**: 130GB free space minimum
- **CUDA**: 12.x
- **OS**: Windows 11 / Linux

### Software Requirements
- Python 3.10 or 3.11
- Git with Git LFS installed
- CUDA 12.x drivers
- Internet connection (for dataset download)

---

## 🚀 Step-by-Step Setup

### 1. Clone Repository

```bash
# Clone from GitHub
git clone <repository-url>
cd tree-pest-identification

# Verify you're on the correct branch
git branch
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install all required packages (this may take 10-15 minutes)
pip install -r requirements.txt

# Verify installation
pip list | grep torch
pip list | grep ultralytics
```

### 4. Verify GPU Access

```bash
# Test CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'CUDA Version: {torch.version.cuda}')"
```

**Expected Output:**
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
CUDA Version: 12.4
```

If this fails, install CUDA 12.x drivers from NVIDIA.

---

## 📦 Download Dataset

### Get Roboflow API Key

You'll need the Roboflow dataset URL with API key:
```
https://app.roboflow.com/ds/YrFRz3Bfo9?key=Hey4UNMjA7
```

### Download Command

```bash
# Download 75k pest images (~10-30 minutes depending on connection)
python scripts/download_dataset.py \
    --url "https://app.roboflow.com/ds/YrFRz3Bfo9?key=Hey4UNMjA7" \
    --output data/raw/pests
```

**Expected Output:**
```
✓ Dataset downloaded successfully
✓ Train images: 67,500
✓ Valid images: 7,500
✓ Data YAML: data/raw/pests/data.yaml
```

**Storage Used**: ~15-20 GB

---

## 📊 Analyze Dataset

```bash
# Analyze class distribution (creates plots and statistics)
python scripts/analyze_dataset.py \
    --dataset data/raw/pests \
    --output outputs/metrics/class_distribution.json
```

**What This Does:**
- Counts samples per class
- Identifies minority classes (< 10th percentile)
- Creates visualization plots
- Saves statistics to JSON

**Output Files:**
- `outputs/metrics/class_distribution.json` - Class statistics
- `outputs/metrics/class_distribution.png` - Visualization plots

---

## 🎨 Prepare Augmentation

```bash
# Create augmentation plan for minority classes
python scripts/prepare_augmentation.py \
    --dataset data/raw/pests \
    --distribution outputs/metrics/class_distribution.json \
    --output data/processed/augmentation_plan.json
```

**What This Does:**
- Identifies minority classes needing augmentation
- Creates augmentation plan (JSON)
- **ZERO disk overhead** - all augmentation happens on-the-fly during training

**Output**: `data/processed/augmentation_plan.json`

---

## 🏋️ Training Modes

Choose one of three training modes based on your time constraints:

### Mode 1: FAST TEST (30-60 minutes) ⚡

**Best For**: Quick validation, testing pipeline, debugging

```bash
python scripts/train_pest_detector.py \
    --config config/pest_detection_fast.yaml \
    --data data/raw/pests/data.yaml \
    --subset 10000
```

**Configuration:**
- Model: YOLO11n (smallest)
- Epochs: 50
- Images: 10,000 (subset)
- Image size: 512×512
- Batch size: 16

**Expected Results:**
- Training time: **30-60 minutes**
- VRAM usage: 4-5 GB
- mAP@50: 0.60-0.70
- Final model: `models/pest_detection_fast/train/weights/best.pt`

**When to Use:**
- First time running the pipeline
- Want to verify everything works
- Testing changes to code/config
- Don't have time for full training

---

### Mode 2: BALANCED (3-5 hours) ⚖️

**Best For**: Production prototype, good accuracy in reasonable time

**Setup:**
1. Edit `config/pest_detection.yaml`:
   ```yaml
   # Line 7: Change model
   model: yolo11s.pt  # Changed from yolo11m.pt
   
   # Line 12: Change epochs
   epochs: 100  # Changed from 150
   ```

2. Run training:
   ```bash
   python scripts/train_pest_detector.py \
       --config config/pest_detection.yaml \
       --data data/raw/pests/data.yaml \
       --augmentation data/processed/augmentation_plan.json
   ```

**Configuration:**
- Model: YOLO11s (small)
- Epochs: 100
- Images: 75,000 (full dataset)
- Image size: 640×640
- Batch size: 8-12

**Expected Results:**
- Training time: **3-5 hours**
- VRAM usage: 5-6 GB
- mAP@50: 0.75-0.85
- Final model: `models/pest_detection/train/weights/best.pt`

**When to Use:**
- Need good results quickly
- Running during work hours
- Production prototype acceptable

---

### Mode 3: OPTIMIZED FULL (5-8 hours) 🎯

**Best For**: Near-optimal accuracy, can run overnight

**Setup:**
1. Edit `config/pest_detection.yaml`:
   ```yaml
   # Line 7: Keep original model
   model: yolo11m.pt  # Keep as is
   
   # Line 12: Reduce epochs
   epochs: 100  # Changed from 150
   
   # Line 13: More aggressive early stopping
   patience: 10  # Changed from 20
   ```

2. Run training:
   ```bash
   python scripts/train_pest_detector.py \
       --config config/pest_detection.yaml \
       --data data/raw/pests/data.yaml \
       --augmentation data/processed/augmentation_plan.json
   ```

**Configuration:**
- Model: YOLO11m (medium - original)
- Epochs: 100
- Images: 75,000 (full dataset)
- Image size: 640×640
- Batch size: 8
- Gradient accumulation: 2 (effective batch = 16)

**Expected Results:**
- Training time: **5-8 hours**
- VRAM usage: 6-7.5 GB
- mAP@50: 0.78-0.88
- Final model: `models/pest_detection/train/weights/best.pt`

**When to Use:**
- Best results needed
- Can run overnight or during off-hours
- Near-production quality acceptable

---

### Mode 4: MAXIMUM QUALITY (8-12 hours) 🏆

**Best For**: Best possible accuracy, publication-ready

**Setup:**
Use original config without changes:

```bash
python scripts/train_pest_detector.py \
    --config config/pest_detection.yaml \
    --data data/raw/pests/data.yaml \
    --augmentation data/processed/augmentation_plan.json
```

**Configuration:**
- Model: YOLO11m (medium)
- Epochs: 150
- Images: 75,000 (full dataset)
- Image size: 640×640
- Batch size: 8
- Patience: 20

**Expected Results:**
- Training time: **8-12 hours**
- VRAM usage: 6-7.5 GB
- mAP@50: 0.80-0.90 (best possible)
- Final model: `models/pest_detection/train/weights/best.pt`

**When to Use:**
- Final production model
- Research/publication quality needed
- Have time for overnight training

---

## 📈 Mode Comparison Table

| Mode | Time | Model | Epochs | Dataset | Expected mAP@50 | VRAM | Use Case |
|------|------|-------|--------|---------|-----------------|------|----------|
| **Fast Test** | 30-60 min | YOLO11n | 50 | 10k subset | 0.60-0.70 | 4-5 GB | Pipeline validation |
| **Balanced** | 3-5 hrs | YOLO11s | 100 | 75k full | 0.75-0.85 | 5-6 GB | Production prototype |
| **Optimized** | 5-8 hrs | YOLO11m | 100 | 75k full | 0.78-0.88 | 6-7 GB | Near-optimal |
| **Maximum** | 8-12 hrs | YOLO11m | 150 | 75k full | 0.80-0.90 | 6-7 GB | Best possible |

---

## 🔍 Monitoring Training

### Real-Time Logs

Training will show real-time progress:
```
Epoch 1/100: 100%|██████████| 8437/8437 [12:34<00:00, 11.17it/s]
      Class     Images  Instances      P      R    mAP50  mAP50-95
        all      7500      45621   0.751   0.682    0.723     0.512

[Epoch 1] VRAM: 6.42/8.00 GB (Reserved: 6.85 GB, Free: 1.58 GB) | GPU Util: 94.3%
```

### TensorBoard (Optional)

```bash
# In a separate terminal
tensorboard --logdir models/pest_detection/train
# Open browser: http://localhost:6006
```

### Key Metrics to Watch

- **VRAM usage**: Should stay < 7.5 GB
- **mAP@50**: Should increase each epoch
- **Loss**: Should decrease over time
- **GPU Utilization**: Should be 90-100%

### Early Stopping

Training will automatically stop if validation metrics don't improve for `patience` epochs:
- Mode 1: patience=10 (stops after 10 epochs without improvement)
- Mode 2: patience=15
- Mode 3: patience=10
- Mode 4: patience=20

---

## ✅ After Training Completes

### 1. Evaluate Model

```bash
python scripts/evaluate_model.py \
    --model models/pest_detection/train/weights/best.pt \
    --data data/raw/pests/data.yaml \
    --output outputs/metrics/evaluation_results.json
```

**Output Files:**
- `outputs/metrics/evaluation_results.json` - Metrics summary
- `outputs/metrics/evaluation_results_per_class.csv` - Per-class metrics
- Training run directory contains:
  - `confusion_matrix.png` - 103×103 confusion matrix
  - `PR_curve.png` - Precision-Recall curve
  - `F1_curve.png` - F1 score curve
  - `results.png` - Training/validation metrics

### 2. Review Results

**Global Metrics** (in JSON):
```json
{
  "global_metrics": {
    "mAP50": 0.87,
    "mAP50_95": 0.65,
    "precision": 0.89,
    "recall": 0.82,
    "f1": 0.85
  }
}
```

**Quality Thresholds:**
- **Excellent**: mAP@50 > 0.85
- **Good**: mAP@50 > 0.75
- **Acceptable**: mAP@50 > 0.65
- **Needs Improvement**: mAP@50 < 0.65

### 3. Test SAHI Inference (Optional)

If you have 4K test images:

```bash
python scripts/run_inference.py \
    --model models/pest_detection/train/weights/best.pt \
    --input path/to/test_images/ \
    --output outputs/predictions/
```

**Performance:**
- ~2.7s per 4K image
- VRAM usage: ~2.5 GB
- Creates annotated visualizations

---

## 📤 Commit Results to Git

### Important Files to Commit

```bash
# 1. Add trained model (tracked by Git LFS)
git add models/pest_detection/train/weights/best.pt

# 2. Add evaluation metrics
git add outputs/metrics/evaluation_results.json
git add outputs/metrics/evaluation_results_per_class.csv

# 3. Add confusion matrix and plots (optional)
git add models/pest_detection/train/confusion_matrix.png
git add models/pest_detection/train/PR_curve.png
git add models/pest_detection/train/F1_curve.png

# 4. Commit with descriptive message
git commit -m "chore: add trained model - Mode 3 (mAP@50: 0.87, 5.2 hours)"

# 5. Push to remote
git push origin main
```

### What NOT to Commit

- ❌ `data/raw/` - Dataset (too large, gitignored)
- ❌ `models/*/train/runs/` - Training runs (gitignored)
- ❌ `models/*/weights/last.pt` - Last checkpoint (gitignored)
- ❌ `venv/` - Virtual environment (gitignored)

---

## 🐛 Troubleshooting

### Issue 1: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
1. Reduce batch size in config:
   ```yaml
   batch_size: 4  # Reduced from 8
   accumulate: 4  # Increased from 2
   ```

2. Or reduce image size:
   ```yaml
   img_size: 512  # Reduced from 640
   ```

3. Ensure no other GPU processes:
   ```bash
   # Check GPU usage
   nvidia-smi
   
   # Kill other processes if needed
   ```

---

### Issue 2: Slow Training (< 5 it/s)

**Symptoms**: Training stuck at 1-2 iterations per second

**Solutions**:
1. Increase `workers` in config:
   ```yaml
   workers: 12  # Increased from 8
   ```

2. Check CPU usage:
   ```bash
   # Windows
   Task Manager → Performance
   
   # Linux
   htop
   ```

3. Verify SSD not HDD (data loading bottleneck)

---

### Issue 3: Training Crashes Mid-Run

**Error**: Training stops unexpectedly

**Solutions**:
1. Check VRAM usage was below 7.5 GB
2. Resume from last checkpoint:
   ```bash
   python scripts/train_pest_detector.py \
       --config config/pest_detection.yaml \
       --data data/raw/pests/data.yaml \
       --resume models/pest_detection/train/weights/last.pt
   ```

3. Check system logs for hardware issues

---

### Issue 4: Low mAP@50 (< 0.60)

**Symptoms**: Model performance below expectations

**Solutions**:
1. Train longer (increase epochs):
   ```yaml
   epochs: 150  # Increase
   patience: 25  # Increase
   ```

2. Check class distribution - might need more aggressive augmentation

3. Verify dataset quality - check for corrupted images/labels:
   ```bash
   python scripts/analyze_dataset.py --dataset data/raw/pests
   ```

---

### Issue 5: Storage Full During Training

**Error**: "No space left on device"

**Solutions**:
1. Verify augmentation is on-the-fly (check `augmentation_plan.json` exists)

2. Delete old training runs:
   ```bash
   # Windows
   rmdir /s models\pest_detection\train_old
   
   # Linux
   rm -rf models/pest_detection/train_old
   ```

3. Keep only `best.pt`:
   ```bash
   # Delete last.pt after training
   del models\pest_detection\train\weights\last.pt  # Windows
   rm models/pest_detection/train/weights/last.pt   # Linux
   ```

---

## 📞 Contact & Support

**Primary Developer**: Illia  
**Machine A** (Development machine)  
**Machine B** (Training machine - You are here)

### When to Contact

- Training fails with unexpected errors
- VRAM issues persist after following troubleshooting
- Results significantly below expected ranges
- Questions about which training mode to use

### Information to Provide

When reporting issues, include:

1. **Training mode used** (Fast/Balanced/Optimized/Maximum)
2. **Error message** (full stack trace)
3. **System info**:
   ```bash
   nvidia-smi
   python --version
   pip list | grep torch
   ```
4. **Training logs** (last 50 lines)
5. **VRAM usage** when error occurred

---

## 📚 Additional Resources

### Documentation
- `README.md` - Project overview and quick start
- `plan.md` - Detailed implementation plan
- `config/pest_detection.yaml` - Training configuration (commented)

### Key Directories
- `models/` - Trained models (commit `best.pt` only)
- `outputs/metrics/` - Evaluation results (commit)
- `data/raw/` - Dataset (do not commit)
- `logs/` - Training logs

### Useful Commands

```bash
# Check disk space
df -h  # Linux
dir   # Windows

# Monitor GPU in real-time
nvidia-smi -l 1  # Updates every second

# Check training progress
tail -f outputs/pest_detection.log  # Linux
type outputs\pest_detection.log     # Windows

# Find best model
ls -lh models/pest_detection/train/weights/best.pt
```

---

## ✅ Checklist Before Starting

- [ ] GPU verified (CUDA available)
- [ ] 130+ GB free disk space
- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip list | grep ultralytics`)
- [ ] Dataset downloaded (75k images)
- [ ] Training mode selected
- [ ] Config file edited (if using Mode 2 or 3)
- [ ] Ready to commit 5-12 hours (depending on mode)

---

## 🎯 Recommended Workflow

**First Time:**
1. Run **Mode 1 (Fast Test)** - 30-60 minutes
2. Verify pipeline works, check VRAM usage
3. If successful, proceed to longer training

**Production Model:**
1. Run **Mode 3 (Optimized)** overnight - 5-8 hours
2. Evaluate results next morning
3. If mAP@50 > 0.75, commit and push
4. If mAP@50 < 0.75, consider Mode 4

**Best Results:**
1. Run **Mode 4 (Maximum)** overnight - 8-12 hours
2. This is the final production model
3. Expected mAP@50 > 0.80

---

**Good luck with training! 🚀**

Last updated: 2026-05-14
