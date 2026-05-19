# Pre-Flight Training Checklist

Run these checks BEFORE starting training to avoid wasting another hour.

## 1. Verify Config File Has Aggressive Augmentation

```bash
grep -E "hsv_h:|degrees:|mixup:|copy_paste:" config/pest_detection_fast.yaml
```

**Expected output:**
```
hsv_h: 0.15             # Should be 0.15 (not 0.015)
degrees: 45.0           # Should be 45.0 (not 15.0)  
mixup: 0.3              # Should be 0.3 (not 0.1)
copy_paste: 0.3         # Should be 0.3 (not 0.0)
```

✅ **PASSED** if all 4 values match above
❌ **FAILED** if any value is wrong → config file wasn't saved correctly

---

## 2. Verify Config Loader Has Augmentation Fields

```bash
grep -A 2 "# Augmentation (YOLO built-in)" src/core/config_loader.py | head -5
```

**Expected output:**
```python
    # Augmentation (YOLO built-in)
    hsv_h: float = Field(ge=0, le=1, default=0.015, description="HSV hue augmentation")
    hsv_s: float = Field(ge=0, le=1, default=0.7, description="HSV saturation augmentation")
```

✅ **PASSED** if you see Pydantic Field definitions for augmentation
❌ **FAILED** if no augmentation fields → schema wasn't updated

---

## 3. Verify Training Script Logs Augmentation

Check that the training script will log augmentation parameters (lines 131-137):

```bash
grep -A 5 "Augmentation Configuration:" scripts/train_pest_detector.py
```

**Expected output:**
```python
    logger.info("\nAugmentation Configuration:")
    logger.info(f"  - Color variation (HSV): H={config_dict.get('hsv_h', 0.015)}, ...")
    ...
```

✅ **PASSED** if logging code exists
❌ **FAILED** if no augmentation logging → script wasn't updated

---

## 4. Delete Old Trained Models (Critical!)

```bash
rm -rf models/pest_detection_fast/train*/
```

**Why:** Prevents accidentally evaluating the OLD model with weak augmentation.

---

## 5. Run Training with VERBOSE logging

Use this exact command:

```bash
python scripts/train_pest_detector.py \
  --config config/pest_detection_fast.yaml \
  --data data/raw/pests/data.yaml \
  --subset 10000 2>&1 | tee training_output.log
```

---

## 6. CHECK LOGS IMMEDIATELY (First 50 lines)

**Within 30 seconds of training start**, check:

```bash
head -50 training_output.log | grep -A 5 "Augmentation Configuration"
```

**Expected output:**
```
Augmentation Configuration:
  - Color variation (HSV): H=0.15, S=0.8, V=0.5      ← Must be 0.15!
  - Spatial transforms: Rotation=45°, ...             ← Must be 45°!
  - Advanced: Mosaic=1.0, Mixup=0.3, CopyPaste=0.3   ← Must show 0.3!
```

✅ **PASSED** → Continue training (augmentation is active)
❌ **FAILED** → **KILL TRAINING IMMEDIATELY** (Ctrl+C) if you see old values like:
  - `H=0.015` (should be 0.15)
  - `Rotation=0°` (should be 45°)
  - `Mixup=0` (should be 0.3)

---

## Validation Bounds Check

These Pydantic constraints MIGHT reject config values:

```python
perspective: float = Field(ge=0, le=0.001, default=0.0)  # Config has 0.001 (exactly at limit)
```

If training fails with `ValidationError`, run:

```bash
grep "perspective:" config/pest_detection_fast.yaml
```

If it shows `perspective: 0.001`, change to `0.0009` and retry.

---

## Summary

**Run checks 1-4 BEFORE training.**  
**Run check 6 within 30 seconds AFTER starting training.**

If check 6 fails, you've wasted only 30 seconds instead of 1 hour.
