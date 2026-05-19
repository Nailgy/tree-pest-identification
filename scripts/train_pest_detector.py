#!/usr/bin/env python3
"""
Train YOLO11m pest detection model with memory optimization and on-the-fly augmentation.
Optimized for RTX 4070 8GB VRAM and 103 pest classes.
"""
import argparse
from pathlib import Path
import sys
import yaml
import numpy as np
from collections import Counter
import torch
from ultralytics import YOLO
from loguru import logger

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.logger import setup_logger
from src.core.config_loader import load_training_config
from src.core.device_manager import DeviceManager
from src.training.memory_optimizer import MemoryOptimizer, estimate_vram_usage
from src.data.augmentation_engine import StorageEfficientAugmentation


def calculate_class_weights(data_yaml_path, num_classes):
    """Calculate inverse frequency weights for class balancing.

    Weight formula: weight_i = max_frequency / frequency_i
    Normalized so mean weight = 1.0
    """
    logger.info("Calculating class weights from dataset...")

    class_counts = Counter()
    train_dir = Path(data_yaml_path).parent / 'train' / 'labels'

    if not train_dir.exists():
        logger.warning(f"Train labels directory not found: {train_dir}")
        return None

    # Count instances per class from label files
    label_files = list(train_dir.glob('*.txt'))
    logger.info(f"Scanning {len(label_files)} label files...")

    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
        except Exception as e:
            logger.warning(f"Error reading {label_file}: {e}")

    if not class_counts:
        logger.warning("No class instances found in dataset")
        return None

    # Calculate inverse frequency weights
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())

    weights = np.zeros(num_classes, dtype=np.float32)
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 1)
        weights[class_id] = max_count / (count + 1)

    # Normalize to mean=1.0
    weights = weights / weights.mean()

    logger.info(f"\nClass Weight Statistics:")
    logger.info(f"  - Min class count: {min_count}")
    logger.info(f"  - Max class count: {max_count}")
    logger.info(f"  - Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    logger.info(f"  - Mean weight: {weights.mean():.3f}")

    return torch.tensor(weights, dtype=torch.float32)


class WeightedLossCallback:
    """Callback to apply class weights to YOLO's loss function during training."""

    def __init__(self, weights):
        self.weights = weights
        self.applied = False

    def on_train_start(self, trainer):
        """Apply class weights when training starts."""
        if self.weights is None or self.applied:
            return

        try:
            device = trainer.device
            weights = self.weights.to(device) if hasattr(self.weights, 'to') else self.weights

            # Access YOLO's loss function through the trainer
            if hasattr(trainer, 'criterion') and trainer.criterion is not None:
                criterion = trainer.criterion
                # YOLO uses BCEWithPosWeight for classification
                if hasattr(criterion, 'pos_weight'):
                    criterion.pos_weight = weights
                    logger.info(f"✓ Class weights applied to criterion on {device}")
                    self.applied = True
                elif hasattr(criterion, 'weight'):
                    criterion.weight = weights
                    logger.info(f"✓ Class weights applied to criterion.weight on {device}")
                    self.applied = True
            else:
                logger.warning("Could not access trainer.criterion for class weighting")
        except Exception as e:
            logger.warning(f"Failed to apply class weights: {e}")


def train_pest_detector(
    config_path: Path,
    data_yaml: Path,
    augmentation_plan: Path = None,
    output_dir: Path = None
):
    """
    Train YOLO11m pest detection model.

    Args:
        config_path: Path to training config YAML
        data_yaml: Path to dataset YAML
        augmentation_plan: Path to augmentation plan JSON (optional)
        output_dir: Override output directory (optional)
    """
    logger.info("=" * 80)
    logger.info("YOLO11m Pest Detection Training")
    logger.info("=" * 80)

    # Load configuration
    logger.info(f"Loading config from: {config_path}")
    config = load_training_config(config_path)
    config_dict = config.model_dump()

    # Verify data YAML exists
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data YAML not found: {data_yaml}")

    logger.info(f"Dataset: {data_yaml}")

    # Load dataset info to get number of classes
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    num_classes = data_config.get('nc', 103)
    class_names = data_config.get('names', [])
    logger.info(f"Number of classes: {num_classes}")

    # Calculate class weights for imbalanced dataset
    class_weights = calculate_class_weights(data_yaml, num_classes)

    # Initialize device manager
    device_manager = DeviceManager(device=config_dict['device'])
    device_manager.monitor_memory(prefix="[Initial]")

    # Initialize memory optimizer
    memory_optimizer = MemoryOptimizer(
        device_manager=device_manager,
        alert_threshold_gb=7.5,
        aggressive_cleanup=True
    )

    # Estimate VRAM usage
    estimated_vram = estimate_vram_usage(
        img_size=config_dict['img_size'],
        batch_size=config_dict['batch_size'],
        num_classes=num_classes,
        model_size='yolo11m'
    )

    if estimated_vram > 7.5:
        logger.warning(
            f"Estimated VRAM ({estimated_vram:.2f} GB) exceeds safe threshold (7.5 GB)."
        )
        logger.info("Optimizing training configuration...")

        # Optimize config
        config_dict = memory_optimizer.optimize_training_config(
            config=config_dict,
            num_classes=num_classes
        )

    # Load augmentation plan if provided
    aug_engine = None
    if augmentation_plan and augmentation_plan.exists():
        logger.info(f"Loading augmentation plan: {augmentation_plan}")
        aug_engine = StorageEfficientAugmentation.load_plan(augmentation_plan)
        stats = aug_engine.get_statistics()
        logger.info(
            f"Augmentation: {stats['classes_to_augment']} minority classes, "
            f"{stats['total_additional_samples']} additional samples (on-the-fly)"
        )
    else:
        if augmentation_plan:
            logger.warning(f"Augmentation plan not found: {augmentation_plan}")
        logger.info("Using config-based augmentation (no augmentation plan)")

    # Override output directory if provided
    if output_dir:
        config_dict['project'] = str(output_dir)

    # Create memory management callbacks
    memory_callbacks = memory_optimizer.create_cleanup_callback()

    # Prepare callback list for YOLO (YOLO accepts list of callback objects)
    callbacks_list = []

    # Add weighted loss callback if class weights were calculated
    if class_weights is not None:
        callbacks_list.append(WeightedLossCallback(class_weights))
        logger.info("Added weighted loss callback to training pipeline")

    # Note: memory_callbacks is a dict returned by create_cleanup_callback()
    # YOLO will use callbacks_list which contains our callback objects
    # The memory cleanup should happen automatically through YOLO's internal mechanisms

    # Initialize YOLO model
    logger.info(f"Initializing YOLO11m model: {config_dict['model']}")
    model = YOLO(config_dict['model'])

    # Log training configuration
    logger.info("\nTraining Configuration:")
    logger.info(f"  - Epochs: {config_dict['epochs']}")
    logger.info(f"  - Image size: {config_dict['img_size']}")
    logger.info(f"  - Batch size: {config_dict['batch_size']}")
    logger.info(f"  - Gradient accumulation: {config_dict['accumulate']} (effective batch: {config_dict['batch_size'] * config_dict['accumulate']})")
    logger.info(f"  - Workers: {config_dict['workers']}")
    logger.info(f"  - Mixed precision (AMP): {config_dict['amp']}")
    logger.info(f"  - Optimizer: {config_dict['optimizer']}")
    logger.info(f"  - Learning rate: {config_dict['lr0']} → {config_dict['lrf']}")
    logger.info(f"  - Device: {config_dict['device']}")

    # Log augmentation configuration
    logger.info("\nAugmentation Configuration:")
    logger.info(f"  - Color variation (HSV): H={config_dict.get('hsv_h', 0.015)}, S={config_dict.get('hsv_s', 0.7)}, V={config_dict.get('hsv_v', 0.4)}")
    logger.info(f"  - Spatial transforms: Rotation={config_dict.get('degrees', 0)}°, Translate={config_dict.get('translate', 0.1)}, Scale={config_dict.get('scale', 0.5)}")
    logger.info(f"  - Flips: UD={config_dict.get('flipud', 0)}, LR={config_dict.get('fliplr', 0.5)}")
    logger.info(f"  - Advanced: Mosaic={config_dict.get('mosaic', 1.0)}, Mixup={config_dict.get('mixup', 0)}, CopyPaste={config_dict.get('copy_paste', 0)}")
    if aug_engine:
        logger.info(f"  - Augmentation Plan: {augmentation_plan}")

    # Pre-training cleanup
    logger.info("\nPreparing for training...")
    memory_optimizer.cleanup_cuda_cache(aggressive=True)
    device_manager.monitor_memory(prefix="[Pre-training]")

    # Start training
    logger.info("\n" + "=" * 80)
    logger.info("Starting Training...")
    logger.info("=" * 80 + "\n")

    try:
        results = model.train(
            data=str(data_yaml.resolve()),
            epochs=config_dict['epochs'],
            patience=config_dict['patience'],
            save_period=config_dict['save_period'],
            imgsz=config_dict['img_size'],
            batch=config_dict['batch_size'],
            workers=config_dict['workers'],
            device=config_dict['device'],

            # Memory optimization
            amp=config_dict['amp'],
            close_mosaic=config_dict.get('close_mosaic', 10),

            # Optimizer
            optimizer=config_dict['optimizer'],
            lr0=config_dict['lr0'],
            lrf=config_dict['lrf'],
            momentum=config_dict['momentum'],
            weight_decay=config_dict['weight_decay'],
            warmup_epochs=config_dict['warmup_epochs'],
            warmup_momentum=config_dict.get('warmup_momentum', 0.8),
            warmup_bias_lr=config_dict.get('warmup_bias_lr', 0.1),
            cos_lr=config_dict['cos_lr'],

            # Augmentation
            hsv_h=config_dict.get('hsv_h', 0.015),
            hsv_s=config_dict.get('hsv_s', 0.7),
            hsv_v=config_dict.get('hsv_v', 0.4),
            degrees=config_dict.get('degrees', 0.0),
            translate=config_dict.get('translate', 0.1),
            scale=config_dict.get('scale', 0.5),
            shear=config_dict.get('shear', 0.0),
            perspective=config_dict.get('perspective', 0.0),
            flipud=config_dict.get('flipud', 0.0),
            fliplr=config_dict.get('fliplr', 0.5),
            mosaic=config_dict.get('mosaic', 1.0),
            mixup=config_dict.get('mixup', 0.0),
            copy_paste=config_dict.get('copy_paste', 0.0),

            # Validation
            val=config_dict['val'],
            plots=config_dict['plots'],
            save_json=config_dict['save_json'],

            # Project
            project=config_dict['project'],
            name=config_dict['name'],
            exist_ok=config_dict['exist_ok'],

            # Callbacks for custom training logic (class weighting)
            callbacks=callbacks_list if callbacks_list else None
        )

        logger.info("\n" + "=" * 80)
        logger.info("Training Completed Successfully!")
        logger.info("=" * 80)

        # Final memory stats
        device_manager.monitor_memory(prefix="[Post-training]")

        # Get best model path
        best_model = Path(config_dict['project']) / config_dict['name'] / 'weights' / 'best.pt'
        last_model = Path(config_dict['project']) / config_dict['name'] / 'weights' / 'last.pt'

        logger.info(f"\nModel Checkpoints:")
        logger.info(f"  - Best model: {best_model}")
        logger.info(f"  - Last model: {last_model}")

        # Log training results
        logger.info(f"\nTraining Results:")
        logger.info(f"  - Final mAP@50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        logger.info(f"  - Final mAP@50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

        logger.info(f"\n✓ Next step: python scripts/evaluate_model.py --model {best_model} --data {data_yaml}")

        return best_model

    except Exception as e:
        logger.error(f"\nTraining failed: {e}")
        device_manager.monitor_memory(prefix="[Error]")
        raise

    finally:
        # Cleanup
        memory_optimizer.cleanup_cuda_cache(aggressive=True)


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO11m pest detection model"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pest_detection.yaml"),
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to dataset YAML (e.g., data/raw/pests/data.yaml)"
    )
    parser.add_argument(
        "--augmentation",
        type=Path,
        default=None,
        help="Path to augmentation plan JSON (optional)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output directory (default: from config)"
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Train on subset of N images (for fast testing)"
    )

    args = parser.parse_args()

    # Create subset if requested
    if args.subset:
        logger.info(f"Creating dataset subset with {args.subset} images...")
        from src.utils.dataset_subset import create_dataset_subset

        subset_dir = Path("data/processed/subset")
        args.data = create_dataset_subset(
            source_dir=args.data.parent,
            output_dir=subset_dir,
            num_images=args.subset
        )

    # Setup logger
    setup_logger(level="INFO")

    try:
        best_model = train_pest_detector(
            config_path=args.config,
            data_yaml=args.data,
            augmentation_plan=args.augmentation,
            output_dir=args.output
        )

        logger.info(f"\n{'=' * 80}")
        logger.info(f"✓ Training complete! Best model: {best_model}")
        logger.info(f"{'=' * 80}\n")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
