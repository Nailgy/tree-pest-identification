"""Stage 1 — evaluate the trained leaf detector on the held-out test split.

Runs Ultralytics validation against `split: test` and prints the core
detection metrics. Use the reported mAP as the readiness gate before moving
on to Stage 2 (SAHI slicing).

Usage:
    python scripts/evaluate_leaf_detector.py
    python scripts/evaluate_leaf_detector.py --weights runs/train/leaf_yolo11m/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "train" / "leaf_yolo11m" / "weights" / "best.pt"
DEFAULT_DATA = REPO_ROOT / "dataset" / "data.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Stage-1 leaf detector.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help="Path to trained .pt weights (default: best.pt from training).")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Path to dataset data.yaml.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Dataset split to evaluate on.")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size.")
    args = parser.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    print(f"[eval] Weights: {args.weights}")
    print(f"[eval] Data:    {args.data}  (split={args.split})")

    model = YOLO(str(args.weights))
    results = model.val(data=str(args.data), split=args.split, imgsz=args.imgsz)

    box = results.box
    print("\n[eval] Results")
    print(f"  Precision : {box.mp:.4f}")
    print(f"  Recall    : {box.mr:.4f}")
    print(f"  mAP@50    : {box.map50:.4f}")
    print(f"  mAP@50-95 : {box.map:.4f}")


if __name__ == "__main__":
    main()
