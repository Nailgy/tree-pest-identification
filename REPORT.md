# Project Report — Detection and Classification of Fruit-Tree Pests

**Stack:** YOLO11m (Ultralytics 8.3.40) · PyTorch 2.5.1 + CUDA 12.4 · Python 3.11
**Training hardware:** Intel Core i9-13900HX · 32 GB RAM · NVIDIA RTX 4070 Laptop 8 GB

---

## 1. Overview

The system locates fruit-tree pests on high-resolution (4K) orchard images through a
three-model pipeline:

```
4K image
   │
   ▼  Stage 1 + 2 — leaf detector (YOLO11m) run with SAHI tiling
leaf boxes (full-image coordinates)
   │
   ▼  crop each leaf  (spatial gate: only real foliage reaches the pest model)
leaf crops
   │
   ▼  Stage 3 — pest detector (YOLO11m) classifies the pest on each crop
pest species + confidence
   │
   ▼  reproject onto the 4K image
annotated 4K image + pest coordinate list
```

The leaf detector acts as a **spatial gate** so the pest model is never asked to
judge background. Output is **leaf-level pest identification**: each detection is a
leaf region labelled with a pest species and confidence.

### How the assignment is addressed

| # | Task | Where in the project |
|---|------|----------------------|
| 1 | Analyse existing datasets | PlantDoc (leaves) + IP02 (pests) |
| 2 | Build own set via transfer-data | `main.py` (PlantDoc subset + class remap); IP02 18-class subset |
| 3 | Augmentation + class balancing | online augmentation in train configs; `scripts/balance_pest_classes.py` |
| 4 | YOLO11m to extract leaves | Stage 1 — `train_detector.py` + `configs/leaf_detection.yaml` |
| 5 | Process 4K images by slicing | `scripts/detect_leaves_sahi.py` (SAHI) |
| 6 | Second YOLO11m for pests on leaves | Stage 3 + `scripts/detect_pests_pipeline.py` |
| 7 | Report (metrics, confusion matrix, F1, …) | **this document** |

---

## 2. Architecture — what each file does

| Path | Role |
|------|------|
| `main.py` | One-off: subset PlantDoc to fruit-tree leaves and remap all kept classes to a single `leaf` class. |
| `configs/leaf_detection.yaml` | Stage-1 training hyperparameters. |
| `configs/pest_detection.yaml` | Stage-3 training hyperparameters (18 classes). |
| `configs/sahi_inference.yaml` | Stage-2 SAHI slicing parameters. |
| `configs/pipeline.yaml` | End-to-end app config: both model paths + detection thresholds. |
| `scripts/train_detector.py` | Generic, config-driven trainer (`model.train()`); used for Stages 1 and 3. |
| `scripts/evaluate_detector.py` | Generic validation: aggregate metrics + per-class table. |
| `scripts/detect_leaves_sahi.py` | Stage 2: SAHI sliced leaf detection → crops + manifests. |
| `scripts/balance_pest_classes.py` | Offline minority-class augmentation for the pest train split. |
| `scripts/detect_pests_pipeline.py` | **The app**: 4K image → leaves → pests → annotated image + coordinates. |
| `dataset/` | Stage-1 data: PlantDoc fruit-tree subset (single `leaf` class). |
| `YOLO_Fruit_Pests_dataset/` | Stage-3 data: IP02 fruit-pest subset (18 classes). |
| `models/` | Trained `best.pt` weights (leaf + pest). |
| `requirements.txt` | Dependencies (CUDA torch installed separately). |

Both inference scripts (`detect_leaves_sahi.py`, `detect_pests_pipeline.py`) accept a
single image or a folder; the pipeline annotates every image one by one with
`[i/N]` progress and per-image error isolation.

---

## 3. Stage 1 — Leaf detector

### 3.1 Dataset (transfer-data from PlantDoc)

PlantDoc was filtered to fruit-tree leaf classes and **remapped to a single class
`leaf`** (`main.py`). Original PlantDoc class IDs kept and merged:

| Original PlantDoc class | Tree |
|---|---|
| 0 Apple Scab Leaf, 1 Apple leaf, 2 Apple rust leaf | Apple |
| 6 Cherry leaf | Cherry |
| 10 Peach leaf | Peach |
| 28 grape leaf black rot, 29 grape leaf | Grape |

All → class `0` (`leaf`). Images with no kept class were removed.

| Split | Images | Resolution |
|---|---|---|
| train | 432 | 416×416 |
| valid | 71 | 416×416 |
| test | 68 | 416×416 |

### 3.2 Augmentation & training

Online (on-the-fly) augmentation only — mosaic, mixup, HSV, flips (H+V), rotation,
scale, translate. Key config: `yolo11m.pt` (COCO transfer learning), imgsz 416,
batch 16, AMP, optimizer AdamW, `lr0 0.001`, cosine LR, 150 epochs, `patience 30`,
`close_mosaic 10`.

### 3.3 Intermediate results (training trajectory, validation set)

| Epoch | mAP@50 | mAP@50-95 |
|---:|---:|---:|
| 1 | 0.257 | 0.089 |
| 10 | 0.603 | 0.307 |
| 50 | 0.757 | 0.469 |
| 100 | 0.820 | 0.554 |
| 150 | 0.844 | 0.575 |

A transient instability at epochs 3–5 (warm-up bias LR overshoot) self-corrected by
epoch 6 with no lasting effect. Loss/metric curves: `runs/train/leaf_yolo11m/results.png`.

### 3.4 Final results

| Set | Precision | Recall | **F1** | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|---:|
| Validation (71 img) | 0.856 | 0.755 | **0.802** | 0.844 | 0.575 |
| **Test (68 img)** | **0.951** | **0.905** | **0.927** | **0.965** | **0.731** |

The leaf detector generalises strongly (test ≥ validation), giving a reliable gate
for Stage 2. *(F1 = 2·P·R / (P+R).)*

---

## 4. Stage 2 — SAHI slicing for 4K images

A 4K image is far larger than the 416-px scale the detector learned, so it is
processed with **SAHI (Slicing Aided Hyper Inference)**: the image is cut into
overlapping **416×416 tiles**, the leaf detector runs on each tile, and detections
are merged back to full-image coordinates.

| Parameter | Value |
|---|---|
| Tile size | 416 × 416 |
| Overlap | 20 % (H and W) |
| Full-image pass | enabled (catches leaves larger than a tile) |
| Merge | GREEDYNMM, match metric IOS, threshold 0.5 |

Implemented in `scripts/detect_leaves_sahi.py` (standalone) and inside the pipeline.
Each detected leaf is cropped (with 5 % padding) and its full-image origin recorded so
pest boxes can later be reprojected onto the 4K image.

---

## 5. Stage 3 — Pest detector

### 5.1 Dataset (IP02 subset, 18 classes)

| Split | Images |
|---|---|
| train | 5 124 |
| valid | 851 |
| test | 2 574 |

Images are variable resolution (~200–400 px). **Every label is a full-frame box
(one pest per image)** — i.e. the data is effectively fine-grained *classification*
expressed in detection format. Consequence: localization is trivial, so
`mAP@50 ≈ mAP@50-95`, and both track classification accuracy. The detection head was
chosen deliberately for pipeline uniformity and mAP reporting.

**Classes and train-set distribution (before balancing):**

| ID | Class | Train instances | ID | Class | Train instances |
|---:|---|---:|---:|---|---:|
| 0 | apple_green_aphid | 210 | 9 | black_scale | 44 |
| 1 | brown_aphid | 113 | 10 | clearwing_moth_peach | 414 |
| 2 | black_aphid | 135 | 11 | clearwing_moth_glass | 84 |
| 3 | red_spider_mite | 317 | 12 | apple_leafminer | 242 |
| 4 | longlegged_mite | 147 | 13 | limacodidae_caterpillar | 840 |
| 5 | comstock_mealybug | 184 | 14 | cherry_fruit_fly | 263 |
| 6 | cottony_cushion_scale | 433 | 15 | green_leafhopper | 767 |
| 7 | ruby_scale | 154 | 16 | epicometis_hirta | 339 |
| 8 | florida_red_scale | 135 | 17 | gall_midge | 303 |

The set is heavily imbalanced (**≈19:1**, limacodidae_caterpillar 840 vs black_scale 44).

### 5.2 Class balancing & augmentation

`scripts/balance_pest_classes.py` performs **offline minority-class augmentation**:
under-represented classes are oversampled with label-preserving augmentations
(flips, rotation, colour/HSV, blur, noise — the full-frame box is unchanged) up to a
target count, applied to the **train split only** (valid/test keep their true
distribution). Online augmentation runs on top during training. This addresses task
requirement #3 ("вирівняти кількість зображень по класах").

### 5.3 Training

`yolo11m.pt`, **imgsz 416** (reduced from 640 after the larger size overflowed the
8 GB VRAM into shared memory), batch 16, AMP, AdamW, `lr0 0.001`, cosine LR,
200 epochs, `patience 50`, `close_mosaic 10`. Peak VRAM ≈ 3.9 GB; ~76 s/epoch.

Training **early-stopped at epoch 151** (no improvement for 50 epochs); **best model
at epoch 101**. Total time ≈ 2.8 h.

### 5.4 Intermediate results (training trajectory, validation set)

| Epoch | mAP@50 | mAP@50-95 |
|---:|---:|---:|
| 5 | 0.283 | 0.270 |
| 10 | 0.419 | 0.406 |
| 20 | 0.579 | 0.563 |
| 50 | 0.723 | 0.704 |
| 100 | 0.823 | 0.805 |
| **101 (best)** | **0.833** | **0.816** |
| 151 (stop) | 0.809 | 0.762 |

Curves: `runs/train/pest_yolo11m6/results.png`.

### 5.5 Final results — aggregate

| Set | Precision | Recall | **F1** | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|---:|
| Validation (851 img) | 0.791 | 0.767 | **0.779** | 0.833 | 0.816 |
| **Test (2 574 img)** | **0.795** | **0.732** | **0.762** | **0.811** | **0.787** |

The small validation→test drop on a 3× larger held-out set confirms good
generalization (no significant over-fitting).

### 5.6 Final results — per class (test split, 2 574 images)

| Class | Precision | Recall | **F1** | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|---:|
| limacodidae_caterpillar | 0.916 | 0.948 | **0.932** | 0.978 | 0.965 |
| green_leafhopper | 0.934 | 0.930 | **0.932** | 0.975 | 0.969 |
| epicometis_hirta | 0.898 | 0.900 | **0.899** | 0.946 | 0.913 |
| cottony_cushion_scale | 0.864 | 0.816 | **0.839** | 0.904 | 0.893 |
| ruby_scale | 0.860 | 0.790 | **0.824** | 0.887 | 0.850 |
| clearwing_moth_peach | 0.796 | 0.837 | **0.816** | 0.897 | 0.860 |
| apple_green_aphid | 0.807 | 0.790 | **0.798** | 0.843 | 0.838 |
| apple_leafminer | 0.857 | 0.737 | **0.793** | 0.861 | 0.826 |
| cherry_fruit_fly | 0.753 | 0.811 | **0.781** | 0.850 | 0.822 |
| clearwing_moth_glass | 0.752 | 0.786 | **0.769** | 0.794 | 0.746 |
| red_spider_mite | 0.770 | 0.759 | **0.764** | 0.820 | 0.808 |
| black_scale | 0.715 | 0.783 | **0.747** | 0.745 | 0.745 |
| florida_red_scale | 0.764 | 0.667 | **0.712** | 0.747 | 0.743 |
| longlegged_mite | 0.757 | 0.608 | **0.674** | 0.726 | 0.726 |
| comstock_mealybug | 0.924 | 0.524 | **0.669** | 0.764 | 0.761 |
| gall_midge | 0.741 | 0.601 | **0.664** | 0.746 | 0.698 |
| black_aphid | 0.665 | 0.471 | **0.551** | 0.604 | 0.497 |
| brown_aphid | 0.530 | 0.416 | **0.466** | 0.503 | 0.502 |

**Effect of balancing:** classes that were rare before balancing (e.g.
clearwing_moth_glass = 84, black_scale = 44 originals) now score well (F1 0.77 / 0.75).
Class size no longer predicts performance — the residual weakness is *separability*,
not sample count.

### 5.7 Confusion matrix & weakest classes

The full 18-way confusion matrix is saved at
`runs/.../confusion_matrix.png` (validation run). The metrics above localize the only
real weaknesses:

- **Aphid trio** — `brown_aphid` (F1 0.47) and `black_aphid` (F1 0.55) are the weakest;
  `apple_green_aphid` is fine (F1 0.80). The two dark aphids are visually near-identical
  and are mutually confused — a fine-grained separability limit, not a data-volume one.
- **comstock_mealybug** — high precision (0.92) but low recall (0.52): it under-detects
  but is rarely wrong when it fires; recoverable with a lower per-class threshold.
- **gall_midge** — F1 0.66 despite 303 training instances → intrinsically hard, not rare.

### 5.8 Losses (illustrative)

| Model (best epoch) | train box / cls / dfl | val box / cls / dfl |
|---|---|---|
| Leaf (ep 150) | 1.036 / 0.707 / 1.317 | 1.101 / 0.746 / 1.331 |
| Pest (ep 101) | 0.182 / 0.797 / 0.928 | 0.128 / 0.572 / 0.406 |

The pest model's very low **box loss** reflects the full-frame boxes (trivial
localization); its informative signal is the **classification (cls) loss**.

---

## 6. End-to-end pipeline (the app)

`scripts/detect_pests_pipeline.py` chains everything (config: `configs/pipeline.yaml`):

1. SAHI-tiled leaf detection on the 4K image (Stage 1/2).
2. Crop each detected leaf (5 % padding).
3. Pest detector on each crop; take the top species above `pest_conf`.
4. Reproject the pest box onto the 4K image (crop origin + box).
5. Write outputs per image.

**Outputs** (`runs/pipeline/`): `annotated/<img>.jpg` (boxes + `species conf`),
`predictions/<img>.json` and `.txt` (full-image coordinates), `summary.json`
(per-image counts / errors).

**Key thresholds:** `--leaf-conf` (gate sensitivity, default 0.25); `--pest-conf`
(default in config) — the pest model has no "healthy" class, so this threshold is what
keeps clean leaves unlabelled.

```bash
python scripts/detect_pests_pipeline.py --source path/to/4k_image_or_folder
```

---

## 7. Limitations & future work

- **Leaf-level localization.** Because the IP02 subset has only full-frame boxes, the
  pipeline reports *which leaf carries which pest*, not tight per-insect boxes. True
  per-insect localization would require pest data with localized bounding boxes.
- **No "healthy" class.** The pest model always returns a species; `pest_conf` is the
  only guard against labelling clean leaves. A dedicated "healthy/background" class
  would make presence/absence explicit.
- **Domain gap.** The pest model trained on pest *close-ups* but receives whole-leaf
  *crops* at inference; expect to tune `pest_conf`, and a small fine-tune on real leaf
  crops would tighten results.
- **Fine-grained aphids.** brown/black aphid confusion is the main accuracy ceiling; more
  diverse aphid imagery or merging indistinguishable sub-classes would help.

---

## 8. Reproducibility

- Python **3.11**; `pip install torch==2.5.1 torchvision==0.20.1 --index-url
  https://download.pytorch.org/whl/cu124` then `pip install -r requirements.txt`.
- Train: `python scripts/train_detector.py --config configs/<stage>.yaml`
- Evaluate: `python scripts/evaluate_detector.py --weights <best.pt> --data <data.yaml> --split test`
- Per-run artefacts (loss curves `results.png`, `confusion_matrix.png`, PR curves,
  `results.csv`) are written under each `runs/train/<name>/` directory.
