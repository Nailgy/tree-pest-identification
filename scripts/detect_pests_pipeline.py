"""End-to-end pipeline — a 4K orchard image in, pest detections on the full image out.

Chains all three trained models (task steps 4-6):
  Stage 1 (leaf detector) via SAHI tiling   ->  leaf boxes on the 4K image
  crop each leaf (the spatial gate)          ->  Stage 3 (pest detector) on the crop
  reproject the pest box back onto the 4K image

Per input image, under --output:
  annotated/<stem>.jpg     4K image with pest boxes + species labels
  predictions/<stem>.json  structured detections (full-image coords, species, conf)
  predictions/<stem>.txt   one line per pest: <species> <conf> <x1> <y1> <x2> <y2>
plus a run-level summary.json.

Why a leaf gate: the pest model only ever sees real leaf regions, never background.
Because the pest model was trained on full-frame pest crops, its predicted box spans
the crop, so the reprojected pest box ≈ the leaf region. The model also has no
"healthy" class, so `pest_conf` is what keeps clean leaves from being labelled.

Usage:
    python scripts/detect_pests_pipeline.py --source path/to/4k_image_or_folder
    python scripts/detect_pests_pipeline.py --source img.jpg --device cpu --pest-conf 0.4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from sahi.models.yolov8 import Yolov8DetectionModel  # loads via ultralytics YOLO -> works with yolo11
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "pipeline.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "pipeline"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_config(path: Path) -> dict:
    """Read pipeline.yaml and resolve both model paths to absolute (repo-root relative)."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("leaf_weights", "pest_weights"):
        p = Path(cfg[key])
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"{key} not found: {p}\nPlace the .pt file there or edit {path.name}.")
        cfg[key] = str(p)
    return cfg


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        imgs = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not imgs:
            raise ValueError(f"No images found in: {source}")
        return imgs
    raise FileNotFoundError(f"Source not found: {source}")


def pad_clamp(x1, y1, x2, y2, pad, w, h):
    """Pad a bbox by `pad` fraction of its size, then clamp to image bounds."""
    dx, dy = int(round((x2 - x1) * pad)), int(round((y2 - y1) * pad))
    return max(0, x1 - dx), max(0, y1 - dy), min(w, x2 + dx), min(h, y2 + dy)


def color_for(class_id: int) -> tuple[int, int, int]:
    """Deterministic, reasonably distinct BGR colour per class."""
    hsv = np.uint8([[[(class_id * 37) % 180, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def draw(img, x1, y1, x2, y2, label, color) -> None:
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def process_image(img_path, leaf_model, pest_model, cfg, device, pest_conf, out_annot, out_pred) -> dict:
    """Run the full chain on one image and write annotated/json/txt. Returns its summary record."""
    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError("could not read image")
    h, w = image.shape[:2]

    # --- Stage 1: locate leaves on the full-res image via sliced inference ---
    leaf_result = get_sliced_prediction(
        str(img_path), leaf_model,
        slice_height=cfg["slice_height"], slice_width=cfg["slice_width"],
        overlap_height_ratio=cfg["overlap_height_ratio"],
        overlap_width_ratio=cfg["overlap_width_ratio"],
        perform_standard_pred=cfg["perform_standard_pred"],
        postprocess_type=cfg["postprocess_type"],
        postprocess_match_metric=cfg["postprocess_match_metric"],
        postprocess_match_threshold=cfg["postprocess_match_threshold"],
        verbose=0,
    )

    annotated = image.copy()
    pests = []
    leaves = leaf_result.object_prediction_list

    # --- Stage 3: classify the pest on each leaf crop, reproject onto the 4K image ---
    for obj in leaves:
        lx1, ly1 = int(obj.bbox.minx), int(obj.bbox.miny)
        lx2, ly2 = int(obj.bbox.maxx), int(obj.bbox.maxy)
        cx1, cy1, cx2, cy2 = pad_clamp(lx1, ly1, lx2, ly2, cfg["crop_padding"], w, h)
        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        res = pest_model.predict(crop, imgsz=cfg["pest_imgsz"], conf=pest_conf,
                                 device=device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue  # no confident pest -> leaf treated as clean

        confs = res.boxes.conf.cpu().numpy()
        k = int(confs.argmax())                       # one pest per leaf -> take the top detection
        conf = float(confs[k])
        cls_id = int(res.boxes.cls[k].item())
        species = pest_model.names[cls_id]
        bx1, by1, bx2, by2 = res.boxes.xyxy[k].cpu().numpy().tolist()
        # reproject crop-space pest box -> full-image coords (add the crop origin)
        X1, Y1, X2, Y2 = int(bx1 + cx1), int(by1 + cy1), int(bx2 + cx1), int(by2 + cy1)

        draw(annotated, X1, Y1, X2, Y2, f"{species} {conf:.2f}", color_for(cls_id))
        pests.append({
            "species": species, "class_id": cls_id, "confidence": round(conf, 4),
            "bbox_xyxy": [X1, Y1, X2, Y2],
            "leaf_bbox_xyxy": [lx1, ly1, lx2, ly2],
        })

    stem = img_path.stem
    cv2.imwrite(str(out_annot / f"{stem}.jpg"), annotated)
    with (out_pred / f"{stem}.json").open("w", encoding="utf-8") as fh:
        json.dump({"image": img_path.name, "image_size_wh": [w, h],
                   "num_leaves": len(leaves), "num_pests": len(pests), "pests": pests}, fh, indent=2)
    with (out_pred / f"{stem}.txt").open("w", encoding="utf-8") as fh:
        for p in pests:
            x1, y1, x2, y2 = p["bbox_xyxy"]
            fh.write(f"{p['species']} {p['confidence']:.4f} {x1} {y1} {x2} {y2}\n")
    return {"image": img_path.name, "num_leaves": len(leaves), "num_pests": len(pests)}


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end pest detection on 4K images.")
    parser.add_argument("--source", type=Path, required=True, help="4K image file or a folder of images.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Pipeline config YAML.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory.")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu, cuda:0).")
    parser.add_argument("--leaf-conf", type=float, default=None, help="Override leaf-detection confidence.")
    parser.add_argument("--pest-conf", type=float, default=None, help="Override pest 'infested' confidence.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device or cfg["device"]
    leaf_conf = args.leaf_conf if args.leaf_conf is not None else cfg["leaf_conf"]
    pest_conf = args.pest_conf if args.pest_conf is not None else cfg["pest_conf"]

    images = collect_images(args.source)
    out_annot = args.output / "annotated"
    out_pred = args.output / "predictions"
    out_annot.mkdir(parents=True, exist_ok=True)
    out_pred.mkdir(parents=True, exist_ok=True)

    print(f"[pipeline] leaf   : {cfg['leaf_weights']}")
    print(f"[pipeline] pest   : {cfg['pest_weights']}")
    print(f"[pipeline] device : {device}  leaf_conf={leaf_conf}  pest_conf={pest_conf}")
    print(f"[pipeline] images : {len(images)}  ->  {args.output}")

    leaf_model = Yolov8DetectionModel(
        model_path=cfg["leaf_weights"], confidence_threshold=leaf_conf,
        device=device, image_size=cfg["slice_width"], load_at_init=True,
    )
    pest_model = YOLO(cfg["pest_weights"])

    # Process every image one by one. A failure on one image is logged and skipped
    # so a single bad file never aborts a whole-folder batch.
    summary = []
    total = len(images)
    for idx, img_path in enumerate(images, 1):
        print(f"[{idx}/{total}] {img_path.name}")
        try:
            rec = process_image(img_path, leaf_model, pest_model, cfg, device, pest_conf, out_annot, out_pred)
            print(f"    -> {rec['num_leaves']} leaves, {rec['num_pests']} pests")
        except Exception as exc:  # noqa: BLE001 — keep the batch alive
            print(f"    ! failed: {exc}")
            rec = {"image": img_path.name, "error": str(exc)}
        summary.append(rec)

    failed = [r for r in summary if "error" in r]
    with (args.output / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    done = total - len(failed)
    print(f"\n[pipeline] Done: {done}/{total} image(s) processed"
          + (f", {len(failed)} failed (see summary.json)" if failed else ""))
    print(f"[pipeline] Annotated -> {out_annot}")
    print(f"[pipeline] Coords    -> {out_pred}")


if __name__ == "__main__":
    main()
