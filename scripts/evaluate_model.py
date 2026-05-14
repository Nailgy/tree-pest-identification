#!/usr/bin/env python3
"""
Evaluate trained pest detection model and extract comprehensive metrics.
"""
import argparse
from pathlib import Path
import sys
import json
import pandas as pd
from ultralytics import YOLO
from loguru import logger

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.logger import setup_logger


def evaluate_model(
    model_path: Path,
    data_yaml: Path,
    output_path: Path,
    img_size: int = 640,
    batch_size: int = 8
):
    """
    Evaluate YOLO model and extract metrics.

    Args:
        model_path: Path to trained model
        data_yaml: Path to dataset YAML
        output_path: Path to save metrics JSON
        img_size: Image size for validation
        batch_size: Batch size for validation
    """
    logger.info("=" * 80)
    logger.info("Model Evaluation")
    logger.info("=" * 80)

    logger.info(f"Model: {model_path}")
    logger.info(f"Dataset: {data_yaml}")

    # Load model
    model = YOLO(str(model_path))

    # Run validation
    logger.info("\nRunning validation...")
    results = model.val(
        data=str(data_yaml),
        imgsz=img_size,
        batch=batch_size,
        plots=True,
        save_json=True,
        device='cuda:0'
    )

    # Extract metrics
    metrics = {
        'model_path': str(model_path),
        'dataset': str(data_yaml),
        'global_metrics': {
            'mAP50': float(results.box.map50),
            'mAP50_95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'f1': float(2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-6))
        }
    }

    # Per-class metrics
    class_names = list(results.names.values())
    per_class_metrics = []

    for i, class_name in enumerate(class_names):
        class_metrics = {
            'class_id': i,
            'class_name': class_name,
            'precision': float(results.box.p[i]) if i < len(results.box.p) else 0.0,
            'recall': float(results.box.r[i]) if i < len(results.box.r) else 0.0,
            'mAP50': float(results.box.ap50[i]) if i < len(results.box.ap50) else 0.0,
            'mAP50_95': float(results.box.ap[i]) if i < len(results.box.ap) else 0.0
        }

        # Calculate F1
        p, r = class_metrics['precision'], class_metrics['recall']
        class_metrics['f1'] = float(2 * (p * r) / (p + r + 1e-6))

        per_class_metrics.append(class_metrics)

    metrics['per_class_metrics'] = per_class_metrics

    # Save metrics
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\nMetrics saved to: {output_path}")

    # Save per-class metrics to CSV
    csv_path = output_path.parent / f"{output_path.stem}_per_class.csv"
    df = pd.DataFrame(per_class_metrics)
    df.to_csv(csv_path, index=False)
    logger.info(f"Per-class metrics saved to: {csv_path}")

    # Log global metrics
    logger.info("\n" + "=" * 80)
    logger.info("Global Metrics")
    logger.info("=" * 80)
    logger.info(f"  mAP@50:     {metrics['global_metrics']['mAP50']:.4f}")
    logger.info(f"  mAP@50-95:  {metrics['global_metrics']['mAP50_95']:.4f}")
    logger.info(f"  Precision:  {metrics['global_metrics']['precision']:.4f}")
    logger.info(f"  Recall:     {metrics['global_metrics']['recall']:.4f}")
    logger.info(f"  F1 Score:   {metrics['global_metrics']['f1']:.4f}")

    # Find worst performing classes
    sorted_classes = sorted(per_class_metrics, key=lambda x: x['f1'])
    logger.info("\n" + "=" * 80)
    logger.info("Worst Performing Classes (Bottom 10 by F1)")
    logger.info("=" * 80)
    for i, cls in enumerate(sorted_classes[:10]):
        logger.info(f"  {i+1}. {cls['class_name']} (ID: {cls['class_id']})")
        logger.info(f"     F1: {cls['f1']:.4f}, P: {cls['precision']:.4f}, R: {cls['recall']:.4f}, mAP50: {cls['mAP50']:.4f}")

    # Find best performing classes
    logger.info("\n" + "=" * 80)
    logger.info("Best Performing Classes (Top 10 by F1)")
    logger.info("=" * 80)
    for i, cls in enumerate(reversed(sorted_classes[-10:])):
        logger.info(f"  {i+1}. {cls['class_name']} (ID: {cls['class_id']})")
        logger.info(f"     F1: {cls['f1']:.4f}, P: {cls['precision']:.4f}, R: {cls['recall']:.4f}, mAP50: {cls['mAP50']:.4f}")

    # Log validation results location
    logger.info(f"\n" + "=" * 80)
    logger.info("Validation Results")
    logger.info("=" * 80)
    logger.info(f"Results directory: {results.save_dir}")
    logger.info(f"  - Confusion matrix: {results.save_dir}/confusion_matrix.png")
    logger.info(f"  - PR curve: {results.save_dir}/PR_curve.png")
    logger.info(f"  - F1 curve: {results.save_dir}/F1_curve.png")
    logger.info(f"  - Results: {results.save_dir}/results.png")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained pest detection model"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model (.pt file)"
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to dataset YAML"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metrics/evaluation_results.json"),
        help="Output path for metrics JSON"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Image size for validation (default: 640)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for validation (default: 8)"
    )

    args = parser.parse_args()

    # Setup logger
    setup_logger(level="INFO")

    # Verify model exists
    if not args.model.exists():
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)

    # Verify data YAML exists
    if not args.data.exists():
        logger.error(f"Data YAML not found: {args.data}")
        sys.exit(1)

    try:
        metrics = evaluate_model(
            model_path=args.model,
            data_yaml=args.data,
            output_path=args.output,
            img_size=args.img_size,
            batch_size=args.batch_size
        )

        logger.info(f"\n{'=' * 80}")
        logger.info(f"✓ Evaluation complete!")
        logger.info(f"{'=' * 80}\n")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
