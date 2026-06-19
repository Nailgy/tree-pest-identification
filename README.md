# Tree Pest Identification (YOLO11m · 3-stage pipeline)

Detection and classification of fruit-tree pests, built in three stages:

1. **Stage 1 — leaf detector:** a YOLO11m detector for a single `leaf` class on a
   fruit-tree subset of PlantDoc (apple, cherry, peach, grape).
2. **Stage 2 — SAHI gate:** slice 4K orchard images into 416×416 tiles and run the
   Stage-1 detector on each tile to locate leaves.
3. **Stage 3 — pest detector:** a second YOLO11m model over 18 fruit-pest classes
   (IP02 subset), run on the leaf crops from Stage 2.

> Training runs on the GPU PC; the leaf and pest models are trained independently,
> then **chained end-to-end** by `scripts/detect_pests_pipeline.py` — see
> [Run the full pipeline](#run-the-full-pipeline-4k-image--pests) for the one-command app.

> `old_project_version/` is kept only as a reference for the working dependency
> versions and structure — it is **not** part of the current pipeline.

## Dataset

`dataset/` is a PlantDoc export, subsetted to fruit-tree leaves and remapped to a
single class via `main.py`:

| Split | Images |
|-------|--------|
| train | 432    |
| valid | 71     |
| test  | 68     |

All images are 416×416; every label uses class `0` (`leaf`). Paths are defined in
[`dataset/data.yaml`](dataset/data.yaml).

## Hardware target (training PC)

Intel Core i9-13900HX · 32 GB RAM · RTX 4070 8 GB · ~130 GB free SSD.

## Setup (Python 3.11)

Python **3.11** is required — newer versions break this dependency stack.

```bash
# Windows
py -3.11 -m venv venv
venv\Scripts\activate

# 1) CUDA build of torch (from PyTorch's index, NOT PyPI)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 2) the rest
pip install -r requirements.txt
```

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# -> True NVIDIA GeForce RTX 4070 ...
```

## Stage 1 — train the leaf detector

```bash
python scripts/train_detector.py                                   # default = leaf config
python scripts/train_detector.py --config configs/leaf_detection.yaml
```

Hyperparameters live in [`configs/leaf_detection.yaml`](configs/leaf_detection.yaml)
(imgsz 416, batch 16, AMP, AdamW + cosine LR, built-in online augmentation). The
pretrained `yolo11m.pt` weights auto-download on the first run.

Outputs land in `runs/train/leaf_yolo11m/`:
- `weights/best.pt`, `weights/last.pt`
- `results.png` (loss / metric curves), `confusion_matrix.png`, val-batch previews.

**Smoke test first:** temporarily set `epochs: 2` to confirm the data loads
(432 train / 71 val), the GPU is used, and `batch: 16` fits in VRAM (watch
`nvidia-smi`). If you hit OOM, lower `batch` to 12 or 8.

### Evaluate

```bash
python scripts/evaluate_detector.py
# custom weights / split:
python scripts/evaluate_detector.py --weights runs/train/leaf_yolo11m/weights/best.pt --split test
```

Prints precision, recall, mAP@50 and mAP@50-95 on the held-out test split — use
these as the readiness gate before Stage 2.

## Stage 2 — SAHI sliced leaf detection (4K images)

Tiles a high-res image into overlapping 416×416 patches, runs the Stage-1
detector on each tile, and merges the boxes back to full-image coordinates.
Leaf detection is an **intermediate spatial gate** — it exists so Stage 3 only
runs the pest model on real leaf regions, not background.

```bash
python scripts/detect_leaves_sahi.py --source path/to/4k_image_or_folder
# tune for recall / pick device / change output dir:
python scripts/detect_leaves_sahi.py --source img.jpg --conf 0.2 --device cuda:0 --output runs/stage2
```

Parameters live in [`configs/sahi_inference.yaml`](configs/sahi_inference.yaml)
(slice 416 to match training, 20% overlap, GREEDYNMM/IOS merge, conf 0.25,
5% crop padding). Outputs under `runs/stage2/`:

- `crops/<stem>_leaf_<i>.jpg` — padded leaf crops → **Stage-3 pest-model input**
- `annotated/<stem>.jpg` — full image with leaf boxes (visual QA of the gate)
- `manifests/<stem>.json` — every leaf's full-image bbox + `crop_origin_xy`
- `detections.json` — run-level summary

**Stage-3 reprojection contract:** each manifest leaf stores `crop_origin_xy`
(the crop's top-left in full-image pixels). A pest box `(px1,py1,px2,py2)` found
in a crop maps back to the 4K image as `(px1+ox, py1+oy, px2+ox, py2+oy)` — that
is how Stage 3 will draw final pest boxes and emit coordinates on the big image.

> Source images are supplied by you (not in the repo). The 416px PlantDoc images
> are *not* valid Stage-2 input — they are a single tile each. Feed real 4K
> orchard photos.

## Stage 3 — train the pest detector (18 classes)

A second YOLO11m detector over 18 fruit-pest classes from an IP02 subset in
[`YOLO_Fruit_Pests_dataset/`](YOLO_Fruit_Pests_dataset/) (5124 train / 851 valid
/ 2574 test). It reuses the **same generic trainer** as Stage 1 — only the config
differs.

> The dataset (~320 MB) is git-ignored — copy it to the GPU machine separately,
> as you did with the Stage-1 `dataset/`.
>
> **Refresh deps on the GPU machine** (no new packages for Stage 3, but this
> syncs the Stage-2 additions — `sahi`, the opencv 4.9 pin):
> ```bash
> venv\Scripts\activate
> pip install -r requirements.txt
> ```
>
> **Note on this data:** every label is a full-frame box (one pest fills each
> image), so this is effectively fine-grained *classification*. Aggregate mAP
> will look high because localization is trivial — judge quality by the
> **per-class table** (below), especially recall on the rare classes
> (black_scale, clearwing_moth_glass, brown_aphid).

**Optional — balance the classes first.** The split is 19:1 imbalanced. This
augments minority classes (TRAIN only) up to a target count so the model isn't
swamped by the big classes:

```bash
python scripts/balance_pest_classes.py                 # target = median class count
python scripts/balance_pest_classes.py --target 300    # lift the small classes higher
```

It adds `aug_*` images into `YOLO_Fruit_Pests_dataset/train/`, leaves valid/test
untouched, and is re-runnable (regenerates `aug_*` each time). No `data.yaml`
change needed — training just sees more train images.

```bash
python scripts/train_detector.py --config configs/pest_detection.yaml
```

Parameters: [`configs/pest_detection.yaml`](configs/pest_detection.yaml) — imgsz
416, batch 16 (drop to 8 to cut VRAM further), 200 epochs, patience 50, built-in
online augmentation. Outputs land in `runs/train/pest_yolo11m/`.

**Smoke test first** (set `epochs: 2`): confirm the train/val images load and
`batch: 16 @ 416` fits VRAM (~5 GB, no overflow into shared memory). Each epoch
is ~5–8× a Stage-1 epoch. Revert to `epochs: 200` and delete the smoke run first.

### Evaluate (per-class)

```bash
python scripts/evaluate_detector.py --weights runs/train/pest_yolo11m/weights/best.pt \
    --data YOLO_Fruit_Pests_dataset/data.yaml --split test --imgsz 416
```

Prints the aggregate metrics plus a **per-class table sorted by weakest recall**,
and points to the `confusion_matrix.png` for the 18-way breakdown.

## Run the full pipeline (4K image → pests)

The app: drop in a high-res orchard photo, get back the same image annotated with
pest boxes + species, and a coordinate list. It chains all three models —
SAHI-tiled leaf detection (Stage 1/2) → crop each leaf → pest detection (Stage 3)
→ reproject the pest box onto the 4K image.

**Setup — place both trained models** where the config expects them (paths in
[`configs/pipeline.yaml`](configs/pipeline.yaml)):

```
models/best.pt                        <- Stage-1 leaf detector  (leaf_weights)
models/pest_yolo11m6/weights/best.pt  <- Stage-3 pest detector  (pest_weights)
```

**Run:**

```bash
# single image
python scripts/detect_pests_pipeline.py --source orchard.jpg
# whole folder — every image is processed one by one
python scripts/detect_pests_pipeline.py --source path/to/images_folder
# CPU machine / tune sensitivity:
python scripts/detect_pests_pipeline.py --source images_folder --device cpu --pest-conf 0.4
```

Point `--source` at a folder and it annotates **every** image in it one by one,
printing `[i/N] <name>` progress and writing a separate annotated image + coord
file per input. A failure on one image is logged to `summary.json` and skipped,
so a bad file never aborts the batch.

Outputs under `runs/pipeline/`:
- `annotated/<stem>.jpg` — the 4K image with pest boxes + `species conf` labels
- `predictions/<stem>.json` — every pest: full-image bbox, species, confidence, parent leaf box
- `predictions/<stem>.txt` — `species conf x1 y1 x2 y2` per line
- `summary.json` — per-image leaf/pest counts

**Two knobs that matter:**
- `--leaf-conf` (default 0.25) — lower to catch more leaves (the gate). 
- `--pest-conf` (default 0.35) — the pest model has **no "healthy" class**, so this
  threshold is what stops clean leaves from being labelled. Raise it for fewer,
  higher-confidence pest calls; lower it to surface more.

> **Scope of the result:** detections are **leaf-level** — the pest box equals the
> leaf region, labelled with the species. The pest model was trained on pest
> *close-ups* but receives whole-leaf crops here (a domain gap), so expect to tune
> `--pest-conf` on your real images; a small fine-tune on real leaf crops would
> tighten it further. Tight per-insect boxes would need pest data with localized
> boxes + a healthy class (a future dataset task).

## How this maps to the assignment

| # | Task | Where |
|---|------|-------|
| 1 | Analyse existing datasets | PlantDoc (leaves) + IP02 (pests) |
| 2 | Build own set via transfer-data | `main.py` (PlantDoc subset+remap); IP02 18-class subset |
| 3 | Augmentation + class balancing | online aug in train configs; `scripts/balance_pest_classes.py` |
| 4 | YOLO11m to extract leaves | Stage 1 — `train_detector.py` + `configs/leaf_detection.yaml` |
| 5 | Process 4K images by slicing | `scripts/detect_leaves_sahi.py` (SAHI) |
| 6 | Second YOLO11m for pests on leaves | Stage 3 + the full pipeline `scripts/detect_pests_pipeline.py` |
| 7 | Report (metrics, confusion matrix, F1, …) | [`REPORT.md`](REPORT.md) |

## Layout

```
configs/leaf_detection.yaml      Stage-1 training hyperparameters
configs/pest_detection.yaml      Stage-3 training hyperparameters (18 classes)
configs/sahi_inference.yaml      Stage-2 SAHI inference parameters
configs/pipeline.yaml            full-pipeline config (both models + thresholds)
scripts/train_detector.py        generic config-driven trainer (Stages 1 & 3)
scripts/evaluate_detector.py     generic YOLO.val() + per-class metrics table
scripts/detect_leaves_sahi.py    Stage-2: SAHI sliced leaf detection -> crops + manifests
scripts/balance_pest_classes.py  Stage-3: offline minority-class augmentation (train split)
scripts/detect_pests_pipeline.py THE APP: 4K image -> leaves -> pests -> annotated 4K + coords
models/                          place trained best.pt weights here (git-ignored)
dataset/                         Stage-1: PlantDoc fruit-tree subset (single `leaf` class)
YOLO_Fruit_Pests_dataset/        Stage-3: IP02 fruit-pest subset (18 classes)
main.py                          one-off: subset + remap PlantDoc labels to class 0
requirements.txt                 lean deps (torch installed separately)
REPORT.md                        full project report (datasets, metrics, F1, architecture)
```
