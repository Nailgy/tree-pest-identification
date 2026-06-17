"""Evaluate a trained YOLO11m detector on a dataset split (config-driven, stage-agnostic).

Runs Ultralytics validation and prints aggregate + PER-CLASS metrics. The
per-class table is the meaningful signal for multi-class models (e.g. the
18-class pest detector), where a high aggregate mAP can hide weak rare classes.

Usage:
    # Stage 1 (leaf) — defaults
    python scripts/evaluate_detector.py
    # Stage 3 (pests)
    python scripts/evaluate_detector.py --weights runs/train/pest_yolo11m/weights/best.pt \
        --data YOLO_Fruit_Pests_dataset/data.yaml --split test --imgsz 640
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
# Stage-1 (leaf) defaults so the zero-arg command still evaluates the leaf model.
DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "train" / "leaf_yolo11m" / "weights" / "best.pt"
DEFAULT_DATA = REPO_ROOT / "dataset" / "data.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO11m detector.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help="Path to trained .pt weights.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Path to dataset data.yaml.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Dataset split to evaluate on.")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size.")
    args = parser.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    print(f"[eval] Weights: {args.weights}")
    print(f"[eval] Data:    {args.data}  (split={args.split}, imgsz={args.imgsz})")

    model = YOLO(str(args.weights))
    results = model.val(data=str(args.data), split=args.split, imgsz=args.imgsz)

    box = results.box
    print("\n[eval] Aggregate")
    print(f"  Precision : {box.mp:.4f}")
    print(f"  Recall    : {box.mr:.4f}")
    print(f"  mAP@50    : {box.map50:.4f}")
    print(f"  mAP@50-95 : {box.map:.4f}")

    # Per-class table for multi-class models — sorted by recall so the weakest
    # (often the rare) classes surface at the top.
    names = model.names  # {class_index: name}
    if len(box.ap_class_index) > 1:
        rows = [
            (names[int(c)], float(box.p[i]), float(box.r[i]),
             float(box.ap50[i]), float(box.ap[i]))
            for i, c in enumerate(box.ap_class_index)
        ]
        print("\n[eval] Per-class (weakest recall first)")
        print(f"  {'class':24s} {'P':>7s} {'R':>7s} {'mAP50':>7s} {'mAP50-95':>9s}")
        for name, p, r, ap50, ap in sorted(rows, key=lambda x: x[2]):
            print(f"  {name:24s} {p:7.3f} {r:7.3f} {ap50:7.3f} {ap:9.3f}")

    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        print(f"\n[eval] Plots (incl. confusion_matrix.png) saved under: {save_dir}")


if __name__ == "__main__":
    main()
