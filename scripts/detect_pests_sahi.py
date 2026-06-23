"""Direct pest detection on high-res images via SAHI — NO leaf gate.

Slices the image into tiles and runs the PEST model on every tile; with
`perform_standard_pred` it also runs the model on the whole image downscaled to
tile size, so a large / close-up pest that fills much of the frame is detected
directly. SAHI merges all detections to full-image coordinates.

This deliberately SKIPS leaf detection (unlike detect_pests_pipeline.py). Trade-off:
every tile — including background — reaches the pest model, which has no 'healthy'
class, so expect more false positives. Keep `pest_conf` relatively high.

Per image, under --output:
  annotated/<stem>.jpg     image with pest boxes + species labels
  predictions/<stem>.json  detections (full-image coords, species, confidence)
  predictions/<stem>.txt   <species> <conf> <x1> <y1> <x2> <y2> per line
plus a run-level summary.json.

Usage:
    python scripts/detect_pests_sahi.py --source path/to/image_or_folder
    python scripts/detect_pests_sahi.py --source img.jpg --device cpu --pest-conf 0.6 --slice 512
    python scripts/detect_pests_sahi.py --source img.jpg --no-full-image    # tiles only
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

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "pest_sahi.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "pests_sahi"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_config(path: Path) -> dict:
    """Read pest_sahi.yaml and resolve pest_weights to an absolute (repo-root) path."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    p = Path(cfg["pest_weights"])
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"pest_weights not found: {p}\nPlace the .pt file there or edit {path.name}.")
    cfg["pest_weights"] = str(p)
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


def process_image(img_path, model, cfg, out_annot, out_pred) -> dict:
    """Run SAHI pest detection on one image; write annotated/json/txt. Returns its record."""
    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError("could not read image")
    h, w = image.shape[:2]

    result = get_sliced_prediction(
        str(img_path), model,
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
    for obj in result.object_prediction_list:
        x1, y1 = int(obj.bbox.minx), int(obj.bbox.miny)
        x2, y2 = int(obj.bbox.maxx), int(obj.bbox.maxy)
        conf = float(obj.score.value)
        cls_id = int(obj.category.id)
        species = obj.category.name
        draw(annotated, x1, y1, x2, y2, f"{species} {conf:.2f}", color_for(cls_id))
        pests.append({
            "species": species, "class_id": cls_id, "confidence": round(conf, 4),
            "bbox_xyxy": [x1, y1, x2, y2],
        })

    stem = img_path.stem
    cv2.imwrite(str(out_annot / f"{stem}.jpg"), annotated)
    with (out_pred / f"{stem}.json").open("w", encoding="utf-8") as fh:
        json.dump({"image": img_path.name, "image_size_wh": [w, h],
                   "num_pests": len(pests), "pests": pests}, fh, indent=2)
    with (out_pred / f"{stem}.txt").open("w", encoding="utf-8") as fh:
        for p in pests:
            x1, y1, x2, y2 = p["bbox_xyxy"]
            fh.write(f"{p['species']} {p['confidence']:.4f} {x1} {y1} {x2} {y2}\n")
    return {"image": img_path.name, "num_pests": len(pests)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct pest detection on high-res images via SAHI (no leaf gate).")
    parser.add_argument("--source", type=Path, required=True, help="Image file or a folder of images.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Config YAML (default values).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory.")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu, cuda:0).")
    parser.add_argument("--pest-conf", type=float, default=None, help="Override pest confidence threshold.")
    parser.add_argument("--slice", type=int, default=None, help="Override slice size (sets height & width).")
    parser.add_argument("--overlap", type=float, default=None, help="Override slice overlap ratio (H & W).")
    parser.add_argument("--no-full-image", action="store_true",
                        help="Disable the full-image-downscaled pass (tiles only).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg["device"] = args.device
    if args.pest_conf is not None:
        cfg["pest_conf"] = args.pest_conf
    if args.slice is not None:
        cfg["slice_height"] = cfg["slice_width"] = args.slice
    if args.overlap is not None:
        cfg["overlap_height_ratio"] = cfg["overlap_width_ratio"] = args.overlap
    if args.no_full_image:
        cfg["perform_standard_pred"] = False

    images = collect_images(args.source)
    out_annot = args.output / "annotated"
    out_pred = args.output / "predictions"
    out_annot.mkdir(parents=True, exist_ok=True)
    out_pred.mkdir(parents=True, exist_ok=True)

    print(f"[pest-sahi] pest   : {cfg['pest_weights']}")
    print(f"[pest-sahi] device : {cfg['device']}  pest_conf={cfg['pest_conf']}  "
          f"slice={cfg['slice_width']}  full_image_pass={cfg['perform_standard_pred']}")
    print(f"[pest-sahi] images : {len(images)}  ->  {args.output}")

    model = Yolov8DetectionModel(
        model_path=cfg["pest_weights"], confidence_threshold=cfg["pest_conf"],
        device=cfg["device"], image_size=cfg["pest_imgsz"], load_at_init=True,
    )

    # Process every image one by one; a failure on one image is logged and skipped.
    summary = []
    total = len(images)
    for idx, img_path in enumerate(images, 1):
        print(f"[{idx}/{total}] {img_path.name}")
        try:
            rec = process_image(img_path, model, cfg, out_annot, out_pred)
            print(f"    -> {rec['num_pests']} pests")
        except Exception as exc:  # noqa: BLE001 — keep the batch alive
            print(f"    ! failed: {exc}")
            rec = {"image": img_path.name, "error": str(exc)}
        summary.append(rec)

    failed = [r for r in summary if "error" in r]
    with (args.output / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    done = total - len(failed)
    print(f"\n[pest-sahi] Done: {done}/{total} image(s) processed"
          + (f", {len(failed)} failed (see summary.json)" if failed else ""))
    print(f"[pest-sahi] Annotated -> {out_annot}")
    print(f"[pest-sahi] Coords    -> {out_pred}")


if __name__ == "__main__":
    main()
