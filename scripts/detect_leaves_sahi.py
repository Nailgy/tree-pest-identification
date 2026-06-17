"""Stage 2 — SAHI sliced leaf detection on high-res (4K) images.

Tiles each input image into overlapping 416x416 patches, runs the Stage-1 leaf
detector (best.pt) on every tile, merges the detections back to full-image
coordinates, and writes the Stage-3 handoff artifacts.

Leaf detection here is an intermediate SPATIAL GATE: its only purpose is to
constrain Stage-3 pest detection to real leaf regions instead of background.

Per input image, under --output:
    crops/<stem>_leaf_<i>.jpg   padded leaf crops      -> Stage-3 pest-model input
    annotated/<stem>.jpg        full image + leaf boxes -> visual QA of the gate
    manifests/<stem>.json       detections + crop geometry in full-image coords

plus a run-level detections.json summary at the output root.

Reprojection contract for Stage 3: each manifest leaf stores `crop_origin_xy`
(the crop's top-left in full-image pixels). A pest box (px1,py1,px2,py2) found
in a crop maps back to the 4K image as (px1+ox, py1+oy, px2+ox, py2+oy).

Usage:
    python scripts/detect_leaves_sahi.py --source path/to/image_or_folder
    python scripts/detect_leaves_sahi.py --source img.jpg --conf 0.2 --output runs/stage2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from sahi.models.yolov8 import Yolov8DetectionModel  # loads via ultralytics YOLO -> works with yolo11
from sahi.predict import get_sliced_prediction
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "sahi_inference.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "stage2"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_config(config_path: Path, weights_override: str | None = None) -> dict:
    """Read the YAML config and resolve `weights` relative to the repo root.

    `weights_override` (from --weights) takes precedence over the config value.
    """
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    weights = Path(weights_override or cfg["weights"])
    if not weights.is_absolute():
        weights = (REPO_ROOT / weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(
            f"Stage-1 weights not found: {weights}\n"
            "Copy best.pt there, pass --weights <path>, or fix `weights:` in the config."
        )
    cfg["weights"] = str(weights)
    return cfg


def collect_images(source: Path) -> list[Path]:
    """Resolve a file or directory into a sorted list of image paths."""
    if source.is_file():
        return [source]
    if source.is_dir():
        imgs = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not imgs:
            raise ValueError(f"No images found in directory: {source}")
        return imgs
    raise FileNotFoundError(f"Source not found: {source}")


def build_model(cfg: dict) -> Yolov8DetectionModel:
    """Instantiate the SAHI detection model from the Stage-1 weights."""
    return Yolov8DetectionModel(
        model_path=cfg["weights"],
        confidence_threshold=cfg["confidence_threshold"],
        device=cfg["device"],
        image_size=cfg["slice_width"],  # run the detector at its native (slice) resolution
        load_at_init=True,
    )


def _pad_clamp(x1: int, y1: int, x2: int, y2: int, pad: float, w: int, h: int):
    """Pad a bbox by `pad` fraction of its size, then clamp to image bounds."""
    bw, bh = x2 - x1, y2 - y1
    dx, dy = int(round(bw * pad)), int(round(bh * pad))
    return (max(0, x1 - dx), max(0, y1 - dy), min(w, x2 + dx), min(h, y2 + dy))


def draw_box(img, x1, y1, x2, y2, label):
    """Draw a single detection box + label in place (BGR)."""
    color = (0, 200, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
    cv2.putText(img, label, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def process_image(image_path: Path, model: Yolov8DetectionModel, cfg: dict, out: dict) -> dict:
    """Run sliced detection on one image and write crops / annotated / manifest."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    h, w = image.shape[:2]

    result = get_sliced_prediction(
        str(image_path),
        model,
        slice_height=cfg["slice_height"],
        slice_width=cfg["slice_width"],
        overlap_height_ratio=cfg["overlap_height_ratio"],
        overlap_width_ratio=cfg["overlap_width_ratio"],
        perform_standard_pred=cfg["perform_standard_pred"],
        postprocess_type=cfg["postprocess_type"],
        postprocess_match_metric=cfg["postprocess_match_metric"],
        postprocess_match_threshold=cfg["postprocess_match_threshold"],
        postprocess_class_agnostic=cfg["postprocess_class_agnostic"],
        verbose=0,
    )

    stem = image_path.stem
    annotated = image.copy()
    leaves = []

    for i, obj in enumerate(result.object_prediction_list):
        x1, y1 = int(obj.bbox.minx), int(obj.bbox.miny)
        x2, y2 = int(obj.bbox.maxx), int(obj.bbox.maxy)
        conf = float(obj.score.value)

        # Padded, clamped crop for the Stage-3 pest model.
        cx1, cy1, cx2, cy2 = _pad_clamp(x1, y1, x2, y2, cfg["crop_padding"], w, h)
        crop_name = f"{stem}_leaf_{i:03d}.jpg"
        cv2.imwrite(str(out["crops"] / crop_name), image[cy1:cy2, cx1:cx2])

        draw_box(annotated, x1, y1, x2, y2, f"leaf {conf:.2f}")

        leaves.append({
            "id": i,
            "bbox_xyxy": [x1, y1, x2, y2],          # leaf box in full-image (4K) pixels
            "confidence": round(conf, 4),
            "crop_file": crop_name,
            "crop_origin_xy": [cx1, cy1],           # add this to a crop-space pest box -> 4K coords
            "crop_size_wh": [cx2 - cx1, cy2 - cy1],
        })

    cv2.imwrite(str(out["annotated"] / f"{stem}.jpg"), annotated)

    manifest = {
        "image": image_path.name,
        "image_size_wh": [w, h],
        "num_leaves": len(leaves),
        "leaves": leaves,
    }
    with (out["manifests"] / f"{stem}.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"  {image_path.name}: {len(leaves)} leaves ({w}x{h})")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 — SAHI sliced leaf detection.")
    parser.add_argument("--source", type=Path, required=True,
                        help="Input image file or a directory of images.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="SAHI inference config YAML.")
    parser.add_argument("--weights", type=str, default=None,
                        help="Override path to the Stage-1 .pt weights (takes precedence over config).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output directory for crops / annotated / manifests.")
    parser.add_argument("--conf", type=float, default=None,
                        help="Override confidence_threshold (e.g. 0.2 to boost recall).")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (e.g. cpu, cuda:0).")
    args = parser.parse_args()

    cfg = load_config(args.config, args.weights)
    if args.conf is not None:
        cfg["confidence_threshold"] = args.conf
    if args.device is not None:
        cfg["device"] = args.device

    images = collect_images(args.source)
    out = {sub: args.output / sub for sub in ("crops", "annotated", "manifests")}
    for d in out.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"[stage2] weights : {cfg['weights']}")
    print(f"[stage2] device  : {cfg['device']}  conf={cfg['confidence_threshold']}  slice={cfg['slice_width']}")
    print(f"[stage2] images  : {len(images)}  ->  {args.output}")

    model = build_model(cfg)

    manifests = [process_image(p, model, cfg, out) for p in images]
    total = sum(m["num_leaves"] for m in manifests)

    summary = {
        "weights": cfg["weights"],
        "config": {k: cfg[k] for k in (
            "confidence_threshold", "slice_height", "slice_width",
            "overlap_height_ratio", "overlap_width_ratio",
            "postprocess_type", "postprocess_match_metric", "crop_padding",
        )},
        "num_images": len(images),
        "total_leaves": total,
        "manifests": [f"manifests/{Path(m['image']).stem}.json" for m in manifests],
    }
    with (args.output / "detections.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[stage2] Done. {total} leaves across {len(images)} images.")
    print(f"[stage2] Crops -> {out['crops']}  (Stage-3 input)")


if __name__ == "__main__":
    main()
