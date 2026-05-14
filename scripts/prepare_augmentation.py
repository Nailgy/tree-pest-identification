#!/usr/bin/env python3
"""
Prepare augmentation plan for minority classes.
"""
import argparse
from pathlib import Path
import sys
import json
from loguru import logger

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.logger import setup_logger
from src.data.augmentation_engine import StorageEfficientAugmentation


def prepare_augmentation_plan(
    dataset_path: Path,
    distribution_path: Path,
    output_path: Path,
    target_samples: int = 100,
    minority_percentile: int = 10
):
    """
    Create augmentation plan for minority classes.

    Args:
        dataset_path: Path to dataset root
        distribution_path: Path to class distribution JSON
        output_path: Path to save augmentation plan
        target_samples: Target samples per minority class
        minority_percentile: Percentile threshold for minority classes
    """
    logger.info(f"Creating augmentation plan for dataset: {dataset_path}")

    # Load class distribution
    if distribution_path.exists():
        with open(distribution_path, 'r') as f:
            distribution = json.load(f)
        class_counts = {int(k): v for k, v in distribution['class_distribution'].items()}
        logger.info(f"Loaded class distribution from: {distribution_path}")
    else:
        # Analyze dataset if distribution not provided
        logger.info("Class distribution not found, analyzing dataset...")
        aug_engine = StorageEfficientAugmentation(
            minority_percentile=minority_percentile,
            target_samples_per_class=target_samples
        )
        class_counts = aug_engine.analyze_class_distribution(
            dataset_path=dataset_path,
            split="train"
        )

    # Create augmentation engine
    aug_engine = StorageEfficientAugmentation(
        minority_percentile=minority_percentile,
        target_samples_per_class=target_samples
    )

    # Create augmentation plan
    plan = aug_engine.create_augmentation_plan(class_counts)

    if not plan:
        logger.warning("No minority classes found. No augmentation needed.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                'minority_percentile': minority_percentile,
                'target_samples_per_class': target_samples,
                'augmentation_plan': {}
            }, f, indent=2)
        return

    # Save plan
    aug_engine.save_plan(output_path)

    # Get statistics
    stats = aug_engine.get_statistics()

    # Log summary
    logger.info(
        f"Augmentation Plan Summary:\n"
        f"  - Classes to augment: {stats['classes_to_augment']}\n"
        f"  - Additional samples: {stats['total_additional_samples']:,}\n"
        f"  - Storage overhead: {stats['storage_overhead_bytes']} bytes (on-the-fly)\n"
        f"  - Minority percentile: {stats['minority_percentile']}%\n"
        f"  - Target samples per class: {stats['target_samples_per_class']}"
    )

    # Show example classes
    logger.info("\nExample minority classes:")
    for i, (class_id, info) in enumerate(list(plan.items())[:5]):
        logger.info(
            f"  Class {class_id}: {info['original_count']} samples → "
            f"{info['target_count']} samples (×{info['augmentation_factor']})"
        )

    logger.info(f"\n✓ Augmentation plan saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare augmentation plan for minority classes"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset root directory"
    )
    parser.add_argument(
        "--distribution",
        type=Path,
        default=Path("outputs/metrics/class_distribution.json"),
        help="Path to class distribution JSON (optional)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/augmentation_plan.json"),
        help="Output path for augmentation plan"
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=100,
        help="Target samples per minority class (default: 100)"
    )
    parser.add_argument(
        "--minority-percentile",
        type=int,
        default=10,
        help="Percentile threshold for minority classes (default: 10)"
    )

    args = parser.parse_args()

    # Setup logger
    setup_logger(level="INFO")

    try:
        prepare_augmentation_plan(
            dataset_path=args.dataset,
            distribution_path=args.distribution,
            output_path=args.output,
            target_samples=args.target_samples,
            minority_percentile=args.minority_percentile
        )

        logger.info(f"✓ Augmentation plan ready")
        logger.info(f"✓ Next step: python scripts/train_pest_detector.py --augmentation {args.output}")

    except Exception as e:
        logger.error(f"Preparation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
