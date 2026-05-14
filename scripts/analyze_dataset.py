#!/usr/bin/env python3
"""
Analyze dataset class distribution to identify minority classes.
"""
import argparse
from pathlib import Path
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.logger import setup_logger
from src.data.augmentation_engine import StorageEfficientAugmentation


def analyze_dataset(
    dataset_path: Path,
    output_path: Path,
    create_plots: bool = True
):
    """
    Analyze dataset class distribution.

    Args:
        dataset_path: Path to dataset root
        output_path: Path to save analysis JSON
        create_plots: Create visualization plots
    """
    logger.info(f"Analyzing dataset: {dataset_path}")

    # Initialize augmentation engine for analysis
    aug_engine = StorageEfficientAugmentation()

    # Analyze train split
    class_counts = aug_engine.analyze_class_distribution(
        dataset_path=dataset_path,
        split="train"
    )

    if not class_counts:
        logger.error("No classes found in dataset")
        sys.exit(1)

    # Calculate statistics
    counts = list(class_counts.values())
    total_images = sum(counts)
    num_classes = len(class_counts)
    min_count = min(counts)
    max_count = max(counts)
    mean_count = sum(counts) / len(counts)
    median_count = sorted(counts)[len(counts) // 2]

    # Find minority classes
    import numpy as np
    minority_threshold = np.percentile(counts, 10)
    minority_classes = {
        class_id: count
        for class_id, count in class_counts.items()
        if count < minority_threshold
    }

    # Create analysis report
    analysis = {
        'dataset_path': str(dataset_path),
        'total_instances': total_images,
        'num_classes': num_classes,
        'min_samples_per_class': min_count,
        'max_samples_per_class': max_count,
        'mean_samples_per_class': mean_count,
        'median_samples_per_class': median_count,
        'minority_threshold_10pct': minority_threshold,
        'num_minority_classes': len(minority_classes),
        'class_distribution': class_counts,
        'minority_classes': minority_classes
    }

    # Save analysis
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    logger.info(f"Analysis saved to: {output_path}")

    # Log summary
    logger.info(
        f"Dataset Analysis Summary:\n"
        f"  - Total instances: {total_images:,}\n"
        f"  - Total classes: {num_classes}\n"
        f"  - Samples per class: min={min_count}, max={max_count}, mean={mean_count:.1f}, median={median_count:.1f}\n"
        f"  - Minority threshold (10th percentile): {minority_threshold:.0f}\n"
        f"  - Minority classes: {len(minority_classes)}"
    )

    # Create visualizations
    if create_plots:
        create_distribution_plots(class_counts, minority_threshold, output_path.parent)

    return analysis


def create_distribution_plots(
    class_counts: dict,
    minority_threshold: float,
    output_dir: Path
):
    """
    Create class distribution visualization plots.

    Args:
        class_counts: Dictionary of class_id -> count
        minority_threshold: Minority class threshold
        output_dir: Directory to save plots
    """
    logger.info("Creating distribution plots...")

    # Convert to DataFrame
    df = pd.DataFrame([
        {'class_id': k, 'count': v}
        for k, v in class_counts.items()
    ]).sort_values('count', ascending=False)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dataset Class Distribution Analysis', fontsize=16, fontweight='bold')

    # 1. Top 20 classes bar plot
    ax1 = axes[0, 0]
    top_20 = df.head(20)
    ax1.bar(range(len(top_20)), top_20['count'], color='steelblue')
    ax1.set_xlabel('Class Rank')
    ax1.set_ylabel('Sample Count')
    ax1.set_title('Top 20 Classes by Sample Count')
    ax1.grid(axis='y', alpha=0.3)

    # 2. Bottom 20 classes bar plot
    ax2 = axes[0, 1]
    bottom_20 = df.tail(20)
    colors = ['red' if c < minority_threshold else 'orange' for c in bottom_20['count']]
    ax2.bar(range(len(bottom_20)), bottom_20['count'], color=colors)
    ax2.axhline(y=minority_threshold, color='red', linestyle='--', label=f'Minority threshold: {minority_threshold:.0f}')
    ax2.set_xlabel('Class Rank (from bottom)')
    ax2.set_ylabel('Sample Count')
    ax2.set_title('Bottom 20 Classes (Minority Classes in Red)')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 3. Distribution histogram
    ax3 = axes[1, 0]
    ax3.hist(df['count'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.axvline(x=minority_threshold, color='red', linestyle='--', label=f'Minority threshold: {minority_threshold:.0f}')
    ax3.set_xlabel('Samples per Class')
    ax3.set_ylabel('Number of Classes')
    ax3.set_title('Distribution of Samples per Class')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # 4. Cumulative distribution
    ax4 = axes[1, 1]
    cumsum = df['count'].cumsum()
    ax4.plot(range(len(df)), cumsum / cumsum.max() * 100, color='steelblue', linewidth=2)
    ax4.set_xlabel('Class Rank')
    ax4.set_ylabel('Cumulative % of Samples')
    ax4.set_title('Cumulative Distribution of Samples')
    ax4.grid(alpha=0.3)
    ax4.set_ylim([0, 100])

    plt.tight_layout()
    plot_path = output_dir / 'class_distribution.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Distribution plots saved to: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze pest detection dataset class distribution"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset root directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metrics/class_distribution.json"),
        help="Output path for analysis JSON"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip creating visualization plots"
    )

    args = parser.parse_args()

    # Setup logger
    setup_logger(level="INFO")

    try:
        analysis = analyze_dataset(
            dataset_path=args.dataset,
            output_path=args.output,
            create_plots=not args.no_plots
        )

        logger.info(f"✓ Analysis complete")
        logger.info(f"✓ Next step: python scripts/prepare_augmentation.py --dataset {args.dataset} --distribution {args.output}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
