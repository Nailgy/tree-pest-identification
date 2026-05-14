#!/usr/bin/env python3
"""
Download dataset from Roboflow with retry logic and progress tracking.
"""
import argparse
from pathlib import Path
import sys
from loguru import logger
from roboflow import Roboflow

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.logger import setup_logger


def download_roboflow_dataset(
    url: str,
    output_dir: Path,
    format: str = "yolov11"
) -> Path:
    """
    Download dataset from Roboflow.

    Args:
        url: Roboflow dataset URL
        output_dir: Output directory
        format: Dataset format (yolov11, yolov8, coco, etc.)

    Returns:
        Path to downloaded dataset
    """
    logger.info(f"Downloading dataset from Roboflow to: {output_dir}")

    # Extract API key from URL
    # Format: https://app.roboflow.com/ds/YrFRz3Bfo9?key=Hey4UNMjA7
    if "?key=" in url:
        workspace_url, api_key = url.split("?key=")
        dataset_id = workspace_url.split("/ds/")[1]
    else:
        raise ValueError("URL must contain API key (?key=YOUR_KEY)")

    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)

    # Get project
    logger.info(f"Fetching dataset: {dataset_id}")
    project = rf.workspace().project(dataset_id)

    # Download dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = project.version(1).download(
        model_format=format,
        location=str(output_dir)
    )

    logger.info(f"Dataset downloaded successfully to: {dataset.location}")

    # Verify download
    data_yaml = Path(dataset.location) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    # Check directories
    train_dir = Path(dataset.location) / "train"
    valid_dir = Path(dataset.location) / "valid"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not valid_dir.exists():
        logger.warning(f"Valid directory not found: {valid_dir}")

    # Count images
    train_images = list(train_dir.glob("images/*.jpg")) + list(train_dir.glob("images/*.png"))
    train_labels = list(train_dir.glob("labels/*.txt"))

    logger.info(
        f"Dataset verification:\n"
        f"  - Train images: {len(train_images)}\n"
        f"  - Train labels: {len(train_labels)}\n"
        f"  - Data YAML: {data_yaml}"
    )

    return Path(dataset.location)


def main():
    parser = argparse.ArgumentParser(
        description="Download pest detection dataset from Roboflow"
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Roboflow dataset URL with API key"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/pests"),
        help="Output directory (default: data/raw/pests)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="yolov11",
        choices=["yolov11", "yolov8", "yolov5", "coco"],
        help="Dataset format (default: yolov11)"
    )

    args = parser.parse_args()

    # Setup logger
    setup_logger(level="INFO")

    try:
        dataset_path = download_roboflow_dataset(
            url=args.url,
            output_dir=args.output,
            format=args.format
        )

        logger.info(f"✓ Dataset ready at: {dataset_path}")
        logger.info(f"✓ Next step: python scripts/analyze_dataset.py --dataset {dataset_path}")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
