# Quick Validation Guide - Run on Your Machine

## Step 1: Fresh Environment Setup

```bash
# Navigate to project directory
cd tree-pest-identification

# Create fresh virtual environment (delete old one if exists)
# Windows: rmdir /s venv
# Linux: rm -rf venv

python -m venv venv

# Activate
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux
```

## Step 2: Install Dependencies in Correct Order

```bash
# Upgrade pip first
pip install --upgrade pip

# Step A: Install PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Step B: Install remaining dependencies
pip install -r requirements.txt
```

## Step 3: Verify GPU Works

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'CUDA Version: {torch.version.cuda}')"
```

**Expected Output:**
```
PyTorch: 2.5.1
CUDA Available: True
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
CUDA Version: 12.4
```

## Step 4: Download Dataset (15-30 minutes)

```bash
# Download 75k pest images from Roboflow
python scripts/download_dataset.py \
    --url "https://app.roboflow.com/ds/YrFRz3Bfo9?key=Hey4UNMjA7" \
    --output data/raw/pests
```

**Expected Output:**
```
✓ Dataset downloaded successfully
✓ Train images: 67,500
✓ Valid images: 7,500
```

## Step 5: Analyze Dataset (2-3 minutes)

```bash
python scripts/analyze_dataset.py \
    --dataset data/raw/pests \
    --output outputs/metrics/class_distribution.json
```

**Expected Output:**
```
✓ Analysis complete
✓ 103 classes found
✓ 75,000 total instances
```

## Step 6: Prepare Augmentation (1 minute)

```bash
python scripts/prepare_augmentation.py \
    --dataset data/raw/pests \
    --distribution outputs/metrics/class_distribution.json \
    --output data/processed/augmentation_plan.json
```

**Expected Output:**
```
✓ Augmentation plan created
✓ Classes to augment: 15
✓ Storage overhead: 0 bytes (on-the-fly)
```

## Step 7: Run FAST TEST Mode (30-60 minutes)

This is the validation test - quick and confirms everything works:

```bash
python scripts/train_pest_detector.py \
    --config config/pest_detection_fast.yaml \
    --data data/raw/pests/data.yaml \
    --subset 10000
```

**What This Does:**
- Uses YOLO11n (small model)
- Trains on 10k images (subset)
- 50 epochs
- Should take 30-60 minutes

**Expected Output During Training:**
```
Epoch 1/50: 100%|██████████| 625/625 [03:45<00:00, 2.78it/s]
      Class     Images  Instances      P      R    mAP50  mAP50-95
        all      1500       9124   0.456   0.421    0.412     0.198

[Epoch 1] VRAM: 4.85/8.00 GB (Reserved: 5.12 GB, Free: 3.15 GB) | GPU Util: 92.1%
```

**After Training Complete (Expected):**
```
✓ Training complete! Best model: models/pest_detection_fast/train/weights/best.pt
✓ Expected mAP@50: 0.60-0.70
```

## Step 8: Evaluate Model (2-3 minutes)

```bash
python scripts/evaluate_model.py \
    --model models/pest_detection_fast/train/weights/best.pt \
    --data data/raw/pests/data.yaml \
    --output outputs/metrics/evaluation_results_fast.json
```

**Expected Output:**
```
Global Metrics:
  mAP@50:     0.65
  mAP@50-95:  0.42
  Precision:  0.70
  Recall:     0.60
  F1 Score:   0.65

✓ Evaluation complete!
```

---

## Complete Timeline

| Step | Duration | Status |
|------|----------|--------|
| Setup + GPU verify | 10 min | Quick ✓ |
| Download dataset | 15-30 min | Depends on internet |
| Analyze | 2 min | Quick ✓ |
| Augmentation | 1 min | Quick ✓ |
| **Fast Test Training** | **30-60 min** | Main test |
| Evaluate | 2 min | Quick ✓ |
| **TOTAL** | **~1.5-2 hours** | |

---

## What If Something Fails?

### Installation Fails
- Check `PYTORCH_INSTALL.md` for troubleshooting
- Try: `pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124`

### GPU Not Found
```bash
# Check NVIDIA drivers
nvidia-smi

# Should show your GPU and CUDA version
```

### Download Fails
- Check internet connection
- Try again with: `python scripts/download_dataset.py --url <url> --output data/raw/pests`

### VRAM Issues During Training
- Expected VRAM for Fast Test: 4-5 GB (safe for 8GB GPU)
- If OOM: reduce batch size in `config/pest_detection_fast.yaml`

### Training Too Slow
- Expected: ~3-4 iterations per second
- If slower: check CPU/disk usage

---

## Success Criteria

✅ **All Green If:**
1. PyTorch imports with CUDA support
2. GPU detected
3. Dataset downloads (67.5k train images)
4. Training starts without OOM
5. mAP@50 ends around 0.65
6. Evaluation completes successfully

---

## Next: After Validation Passes

If everything works:

```bash
# Option A: Run Mode 3 (Optimized) overnight
# Edit config/pest_detection.yaml:
# - Change: model: yolo11m.pt
# - Change: epochs: 100
# - Change: patience: 10

python scripts/train_pest_detector.py \
    --config config/pest_detection.yaml \
    --data data/raw/pests/data.yaml \
    --augmentation data/processed/augmentation_plan.json

# Expected: 5-8 hours, mAP@50: 0.78-0.88
```

---

## Commands Quick Reference

```bash
# Activate environment
venv\Scripts\activate

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Download data
python scripts/download_dataset.py --url <url> --output data/raw/pests

# Analyze
python scripts/analyze_dataset.py --dataset data/raw/pests

# Augmentation
python scripts/prepare_augmentation.py --dataset data/raw/pests

# FAST TEST (validation)
python scripts/train_pest_detector.py --config config/pest_detection_fast.yaml --data data/raw/pests/data.yaml --subset 10000

# Evaluate
python scripts/evaluate_model.py --model models/pest_detection_fast/train/weights/best.pt --data data/raw/pests/data.yaml

# SAHI Inference (optional, for 4K images)
python scripts/run_inference.py --model models/pest_detection_fast/train/weights/best.pt --input path/to/images/
```

---

**Run this now and let me know:**
1. ✓/✗ Does GPU show in Step 3?
2. ✓/✗ Does dataset download in Step 4?
3. ✓/✗ Does Fast Test training start (Step 7)?
4. ✓/✗ What's the final mAP@50?

I'll help troubleshoot any issues! 🚀
