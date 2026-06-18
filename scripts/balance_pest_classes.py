"""Stage 3 helper — offline class balancing for the IP02 pest train split.

The dataset is heavily imbalanced (19:1) and every image is a single full-frame
pest (label = `<cls> 0.5 0.5 1.0 1.0`). This script lifts under-represented
classes toward a target count by generating augmented copies of their images,
which "slightly aligns" the per-class distribution before training.

Because the object fills the whole frame, augmentation is label-preserving:
the box stays full-frame, so each new image just reuses the same one-line label.
Only the TRAIN split is touched — valid/test keep their true distribution so
metrics stay honest.

Re-runnable: it first deletes any previously generated `aug_*` files, so you can
tweak --target and run again without compounding. To fully revert, delete the
`aug_*` files (or re-run with --target 0).

Usage (run from repo root on the GPU machine):
    python scripts/balance_pest_classes.py
    python scripts/balance_pest_classes.py --target 300 --max-mult 8 --seed 42
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from statistics import median

import albumentations as A
import cv2
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "YOLO_Fruit_Pests_dataset" / "data.yaml"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def build_pipeline() -> A.Compose:
    """Label-preserving image augmentations (object fills the frame, so no bbox math).

    Colour/lighting + flips/rotations + mild blur/noise — enough variety to avoid
    near-duplicate copies without distorting the pest beyond recognition.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.4),
        A.OneOf([A.GaussianBlur(p=1.0), A.MotionBlur(p=1.0)], p=0.2),
        A.GaussNoise(p=0.2),
        A.CLAHE(p=0.2),
    ])


def resolve_dirs(data_yaml: Path) -> tuple[Path, Path, list[str]]:
    """Return (train_images_dir, train_labels_dir, class_names) from data.yaml."""
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    images_dir = (data_yaml.parent / cfg["train"]).resolve()
    labels_dir = images_dir.parent / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"train images/labels not found under {images_dir.parent}")
    return images_dir, labels_dir, list(cfg.get("names", []))


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def clean_previous(images_dir: Path, labels_dir: Path) -> int:
    removed = 0
    for d, pat in ((images_dir, "aug_*"), (labels_dir, "aug_*")):
        for f in d.glob(pat):
            f.unlink()
            removed += 1
    return removed


def scan_classes(labels_dir: Path) -> dict[int, list[str]]:
    """Map class_id -> list of (original) image stems. 1 box per image is assumed."""
    by_class: dict[int, list[str]] = defaultdict(list)
    for lbl in labels_dir.glob("*.txt"):
        if lbl.name.startswith("aug_"):
            continue
        first = lbl.read_text(encoding="utf-8").split()
        if first:
            by_class[int(first[0])].append(lbl.stem)
    return by_class


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline class balancing for the pest train split.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to pest data.yaml.")
    parser.add_argument("--target", type=int, default=None,
                        help="Per-class target count for minority classes (default: median of class counts).")
    parser.add_argument("--max-mult", type=float, default=8.0,
                        help="Cap on synthetic images per class as a multiple of its originals (anti over-duplication).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    images_dir, labels_dir, names = resolve_dirs(args.data)

    removed = clean_previous(images_dir, labels_dir)
    if removed:
        print(f"[balance] removed {removed} previous aug_* files")

    by_class = scan_classes(labels_dir)
    counts = {c: len(s) for c, s in by_class.items()}
    target = args.target if args.target is not None else int(median(counts.values()))
    pipeline = build_pipeline()

    def cname(c: int) -> str:
        return names[c] if 0 <= c < len(names) else f"class_{c}"

    print(f"[balance] target per class: {target}   (max {args.max_mult}x originals)")
    print(f"  {'class':24s} {'orig':>6s} {'added':>6s} {'final':>6s}")

    total_added = 0
    for cid in sorted(by_class):
        stems = by_class[cid]
        orig = len(stems)
        added = 0
        if orig < target:
            needed = min(target - orig, int(orig * args.max_mult))
            for i in range(needed):
                src = find_image(images_dir, random.choice(stems))
                if src is None:
                    continue
                img = cv2.imread(str(src))
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                out = pipeline(image=rgb)["image"]
                stem = f"aug_{cid:02d}_{cname(cid)}_{i:04d}"
                cv2.imwrite(str(images_dir / f"{stem}.jpg"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
                (labels_dir / f"{stem}.txt").write_text(f"{cid} 0.5 0.5 1.0 1.0\n", encoding="utf-8")
                added += 1
        total_added += added
        print(f"  {cname(cid):24s} {orig:6d} {added:6d} {orig + added:6d}")

    print(f"\n[balance] done. Added {total_added} augmented images to {images_dir}")
    print("[balance] valid/test left untouched. Re-run anytime; aug_* files are regenerated.")


if __name__ == "__main__":
    main()
