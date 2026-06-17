"""Generic YOLO11m detector trainer (config-driven, stage-agnostic).

Lean wrapper around Ultralytics: load hyperparameters from a YAML config and
hand them straight to ``model.train()``. Ultralytics handles AMP, device
placement, the DataLoader and logging natively, so there is no custom
device/memory code here.

The config selects the dataset/model, so the same script trains every stage:
    Stage 1 (leaf, default):  python scripts/train_detector.py
    Stage 3 (pests):          python scripts/train_detector.py --config configs/pest_detection.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

# Repo root = parent of this script's directory (scripts/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "leaf_detection.yaml"


def load_config(config_path: Path) -> dict:
    """Read the YAML config and resolve the `data:` path relative to the repo root.

    Making `data` absolute means training works regardless of the current
    working directory it is launched from.
    """
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    data_path = Path(cfg["data"])
    if not data_path.is_absolute():
        data_path = (REPO_ROOT / data_path).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"data.yaml not found at: {data_path}")
    cfg["data"] = str(data_path)

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Stage-1 leaf detector.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the training hyperparameter YAML.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[train] Config:        {args.config}")
    print(f"[train] data.yaml:     {cfg['data']}")
    print(f"[train] Base weights:  {cfg['model']}")

    model = YOLO(cfg.pop("model"))
    results = model.train(**cfg)

    # Ultralytics writes weights to <save_dir>/weights/best.pt
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n[train] Done. Best weights: {best}")


if __name__ == "__main__":
    # Guard is required on Windows for the DataLoader's worker processes.
    main()
