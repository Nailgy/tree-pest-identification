"""
Storage-efficient on-the-fly augmentation engine.
Augments only minority classes in-memory during training.
Zero disk overhead - critical for 130GB storage constraint.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
from collections import Counter
import albumentations as A
from loguru import logger


class StorageEfficientAugmentation:
    """
    Analyzes dataset and creates augmentation plan for minority classes only.
    All augmentation happens on-the-fly during training (no disk writes).
    """

    def __init__(
        self,
        minority_percentile: int = 10,
        target_samples_per_class: int = 100
    ):
        """
        Initialize augmentation engine.

        Args:
            minority_percentile: Classes below this percentile are minority
            target_samples_per_class: Target samples after augmentation
        """
        self.minority_percentile = minority_percentile
        self.target_samples = target_samples_per_class
        self.augmentation_plan: Dict = {}

    def analyze_class_distribution(
        self,
        dataset_path: Path,
        split: str = "train"
    ) -> Dict[int, int]:
        """
        Analyze class distribution in dataset.

        Args:
            dataset_path: Path to dataset root
            split: Dataset split (train/valid)

        Returns:
            Dictionary mapping class_id -> sample_count
        """
        labels_dir = dataset_path / split / "labels"

        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

        # Count classes across all label files
        class_counts = Counter()

        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1

        logger.info(f"Found {len(class_counts)} classes with {sum(class_counts.values())} total instances")

        return dict(class_counts)

    def create_augmentation_plan(
        self,
        class_counts: Dict[int, int]
    ) -> Dict[int, Dict]:
        """
        Create augmentation plan for minority classes.

        Args:
            class_counts: Dictionary of class_id -> count

        Returns:
            Augmentation plan: class_id -> {original_count, aug_factor, target_count}
        """
        if not class_counts:
            logger.warning("No classes found in dataset")
            return {}

        counts = list(class_counts.values())
        minority_threshold = np.percentile(counts, self.minority_percentile)

        plan = {}
        total_augmented = 0

        for class_id, count in class_counts.items():
            if count < minority_threshold:
                # Calculate augmentation factor
                aug_factor = max(1, self.target_samples // count)
                target_count = count * aug_factor

                plan[class_id] = {
                    'original_count': count,
                    'augmentation_factor': aug_factor,
                    'target_count': target_count
                }
                total_augmented += (target_count - count)

        logger.info(
            f"Augmentation plan created:\n"
            f"  - Minority threshold: {minority_threshold:.0f} samples\n"
            f"  - Classes to augment: {len(plan)}\n"
            f"  - Additional samples: {total_augmented}\n"
            f"  - Storage overhead: 0 bytes (on-the-fly)"
        )

        self.augmentation_plan = plan
        return plan

    def get_augmentation_pipeline(
        self,
        severity: str = "medium"
    ) -> A.Compose:
        """
        Get Albumentations augmentation pipeline.

        Args:
            severity: Augmentation severity (light, medium, heavy)

        Returns:
            Albumentations Compose pipeline
        """
        if severity == "light":
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15,
                    contrast_limit=0.15,
                    p=0.3
                ),
                A.Rotate(limit=15, p=0.3),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels']
            ))

        elif severity == "medium":
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.Rotate(limit=30, p=0.5),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.RandomScale(scale_limit=0.1, p=0.3),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.4
                ),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels']
            ))

        else:  # heavy
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.4),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.6
                ),
                A.Rotate(limit=45, p=0.6),
                A.GaussNoise(var_limit=(10, 80), p=0.4),
                A.RandomScale(scale_limit=0.2, p=0.4),
                A.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.15,
                    p=0.5
                ),
                A.Blur(blur_limit=3, p=0.2),
                A.CLAHE(clip_limit=2.0, p=0.2),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels']
            ))

    def should_augment(self, class_id: int) -> bool:
        """
        Check if a class should be augmented.

        Args:
            class_id: Class ID to check

        Returns:
            True if class is in augmentation plan
        """
        return class_id in self.augmentation_plan

    def get_augmentation_factor(self, class_id: int) -> int:
        """
        Get augmentation factor for a class.

        Args:
            class_id: Class ID

        Returns:
            Augmentation factor (1 if not in plan)
        """
        return self.augmentation_plan.get(class_id, {}).get('augmentation_factor', 1)

    def save_plan(self, save_path: Path):
        """
        Save augmentation plan to JSON.

        Args:
            save_path: Path to save JSON
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump({
                'minority_percentile': self.minority_percentile,
                'target_samples_per_class': self.target_samples,
                'augmentation_plan': self.augmentation_plan
            }, f, indent=2)

        logger.info(f"Augmentation plan saved to: {save_path}")

    @classmethod
    def load_plan(cls, plan_path: Path) -> 'StorageEfficientAugmentation':
        """
        Load augmentation plan from JSON.

        Args:
            plan_path: Path to JSON file

        Returns:
            StorageEfficientAugmentation instance
        """
        with open(plan_path, 'r') as f:
            data = json.load(f)

        instance = cls(
            minority_percentile=data['minority_percentile'],
            target_samples_per_class=data['target_samples_per_class']
        )
        instance.augmentation_plan = data['augmentation_plan']

        logger.info(f"Augmentation plan loaded from: {plan_path}")
        return instance

    def get_statistics(self) -> Dict:
        """
        Get augmentation statistics.

        Returns:
            Dictionary with statistics
        """
        if not self.augmentation_plan:
            return {
                'classes_to_augment': 0,
                'total_additional_samples': 0,
                'storage_overhead_bytes': 0
            }

        total_additional = sum(
            plan['target_count'] - plan['original_count']
            for plan in self.augmentation_plan.values()
        )

        return {
            'classes_to_augment': len(self.augmentation_plan),
            'total_additional_samples': total_additional,
            'storage_overhead_bytes': 0,  # All in-memory
            'minority_percentile': self.minority_percentile,
            'target_samples_per_class': self.target_samples
        }


def parse_yolo_label(label_line: str) -> Tuple[int, List[float]]:
    """
    Parse YOLO format label line.

    Args:
        label_line: Line from label file (class x_center y_center width height)

    Returns:
        Tuple of (class_id, bbox coordinates)
    """
    parts = label_line.strip().split()
    class_id = int(parts[0])
    bbox = [float(x) for x in parts[1:5]]
    return class_id, bbox


def format_yolo_label(class_id: int, bbox: List[float]) -> str:
    """
    Format YOLO label line.

    Args:
        class_id: Class ID
        bbox: Bounding box [x_center, y_center, width, height]

    Returns:
        Formatted label string
    """
    return f"{class_id} {' '.join(f'{x:.6f}' for x in bbox)}\n"
