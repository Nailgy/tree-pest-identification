"""
Utility to create a subset of the dataset for faster training/testing.
"""
import random
import shutil
from pathlib import Path
from typing import Optional
from loguru import logger


def create_dataset_subset(
    source_dir: Path,
    output_dir: Path,
    num_images: int,
    seed: int = 42
) -> Path:
    """
    Create a subset of dataset for faster training.

    Args:
        source_dir: Source dataset directory
        output_dir: Output directory for subset
        num_images: Number of images to sample
        seed: Random seed for reproducibility

    Returns:
        Path to subset data.yaml
    """
    random.seed(seed)

    logger.info(f"Creating dataset subset: {num_images} images")

    # Create output structure
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (output_dir / "valid" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "valid" / "labels").mkdir(parents=True, exist_ok=True)

    # Sample train images
    train_images = list((source_dir / "train" / "images").glob("*.jpg"))
    train_images.extend(list((source_dir / "train" / "images").glob("*.png")))

    if len(train_images) < num_images:
        logger.warning(f"Requested {num_images} but only {len(train_images)} available")
        num_images = len(train_images)

    sampled_images = random.sample(train_images, num_images)

    # Copy train subset
    logger.info(f"Copying {len(sampled_images)} train images...")
    for img_path in sampled_images:
        # Copy image
        shutil.copy(img_path, output_dir / "train" / "images" / img_path.name)

        # Copy label
        label_path = source_dir / "train" / "labels" / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy(label_path, output_dir / "train" / "labels" / f"{img_path.stem}.txt")

    # Copy valid subset (smaller, 20% of train)
    valid_images = list((source_dir / "valid" / "images").glob("*.jpg"))
    valid_images.extend(list((source_dir / "valid" / "images").glob("*.png")))

    num_valid = min(num_images // 5, len(valid_images))
    sampled_valid = random.sample(valid_images, num_valid)

    logger.info(f"Copying {len(sampled_valid)} valid images...")
    for img_path in sampled_valid:
        # Copy image
        shutil.copy(img_path, output_dir / "valid" / "images" / img_path.name)

        # Copy label
        label_path = source_dir / "valid" / "labels" / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy(label_path, output_dir / "valid" / "labels" / f"{img_path.stem}.txt")

    # Create data.yaml
    import yaml
    with open(source_dir / "data.yaml", 'r') as f:
        data_config = yaml.safe_load(f)

    # Use an absolute dataset root with relative split paths to prevent
    # Ultralytics from prefixing paths twice (which causes missing image errors).
    data_config['path'] = str(output_dir.resolve())
    data_config['train'] = "train/images"
    data_config['val'] = "valid/images"

    subset_yaml = output_dir / "data.yaml"
    with open(subset_yaml, 'w') as f:
        yaml.dump(data_config, f)

    logger.info(f"✓ Subset created at: {output_dir}")
    logger.info(f"  - Train: {len(sampled_images)} images")
    logger.info(f"  - Valid: {len(sampled_valid)} images")
    logger.info(f"  - data.yaml: {subset_yaml}")

    return subset_yaml
