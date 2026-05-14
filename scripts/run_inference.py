#!/usr/bin/env python3
"""
Run SAHI inference on 4K images with trained pest detection model.
"""
import argparse
from pathlib import Path
import sys
import json
from time import time
from loguru import logger

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.logger import setup_logger
from src.inference.sahi_predictor import SAHIPredictor, load_sahi_config


def run_inference(
    model_path: Path,
    input_path: Path,
    output_dir: Path,
    sahi_config_path: Path = None,
    visualize: bool = True
):
    """
    Run SAHI inference on images.

    Args:
        model_path: Path to trained model
        input_path: Path to image or directory of images
        output_dir: Output directory for results
        sahi_config_path: Path to SAHI config YAML
        visualize: Create visualization images
    """
    logger.info("=" * 80)
    logger.info("SAHI Pest Detection Inference")
    logger.info("=" * 80)

    # Load SAHI config
    if sahi_config_path and sahi_config_path.exists():
        logger.info(f"Loading SAHI config: {sahi_config_path}")
        sahi_config = load_sahi_config(sahi_config_path)
    else:
        logger.info("Using default SAHI config")
        sahi_config = {
            'slice_height': 640,
            'slice_width': 640,
            'overlap_height_ratio': 0.2,
            'overlap_width_ratio': 0.2,
            'confidence_threshold': 0.25,
            'device': 'cuda:0'
        }

    # Initialize predictor
    predictor = SAHIPredictor(
        model_path=model_path,
        device=sahi_config.get('device', 'cuda:0'),
        confidence_threshold=sahi_config.get('confidence_threshold', 0.25),
        slice_height=sahi_config.get('slice_height', 640),
        slice_width=sahi_config.get('slice_width', 640),
        overlap_height_ratio=sahi_config.get('overlap_height_ratio', 0.2),
        overlap_width_ratio=sahi_config.get('overlap_width_ratio', 0.2)
    )

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations" if visualize else None
    if viz_dir:
        viz_dir.mkdir(parents=True, exist_ok=True)

    # Collect image paths
    image_paths = []
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_paths.extend(input_path.glob(ext))
    else:
        raise ValueError(f"Invalid input path: {input_path}")

    if not image_paths:
        raise ValueError(f"No images found in: {input_path}")

    logger.info(f"Found {len(image_paths)} images to process")

    # Run inference
    results = []
    total_time = 0

    for i, image_path in enumerate(image_paths):
        logger.info(f"\n[{i+1}/{len(image_paths)}] Processing: {image_path.name}")

        start_time = time()

        try:
            result = predictor.predict_image(
                image_path=image_path,
                visualize=visualize,
                output_path=viz_dir / f"{image_path.stem}_predicted.jpg" if viz_dir else None
            )

            elapsed = time() - start_time
            total_time += elapsed

            result['inference_time_sec'] = elapsed
            results.append(result)

            logger.info(f"  - Detections: {result['num_detections']}")
            logger.info(f"  - Time: {elapsed:.2f}s")

            # Log VRAM usage
            vram = predictor.get_vram_usage()
            logger.info(f"  - VRAM: {vram['allocated']:.2f}/{vram['total']:.2f} GB")

        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {e}")
            results.append({
                'image_path': str(image_path),
                'error': str(e)
            })

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("Inference Complete")
    logger.info("=" * 80)

    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]

    logger.info(f"\nSummary:")
    logger.info(f"  - Total images: {len(image_paths)}")
    logger.info(f"  - Successful: {len(successful)}")
    logger.info(f"  - Failed: {len(failed)}")
    logger.info(f"  - Total time: {total_time:.2f}s")
    logger.info(f"  - Average time: {total_time / len(successful):.2f}s per image" if successful else "  - Average time: N/A")

    # Total detections
    total_detections = sum(r['num_detections'] for r in successful)
    logger.info(f"  - Total detections: {total_detections}")
    logger.info(f"  - Average detections: {total_detections / len(successful):.1f} per image" if successful else "  - Average detections: N/A")

    # Save results to JSON
    results_path = output_dir / "predictions.json"
    with open(results_path, 'w') as f:
        json.dump({
            'model': str(model_path),
            'config': sahi_config,
            'summary': {
                'total_images': len(image_paths),
                'successful': len(successful),
                'failed': len(failed),
                'total_time_sec': total_time,
                'average_time_sec': total_time / len(successful) if successful else 0,
                'total_detections': total_detections
            },
            'results': results
        }, f, indent=2)

    logger.info(f"\nResults saved to: {results_path}")
    if visualize:
        logger.info(f"Visualizations saved to: {viz_dir}")

    # Log failed images
    if failed:
        logger.warning(f"\nFailed images ({len(failed)}):")
        for r in failed:
            logger.warning(f"  - {Path(r['image_path']).name}: {r['error']}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SAHI inference on 4K images"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained YOLO model (.pt file)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to image or directory of images"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/predictions"),
        help="Output directory (default: outputs/predictions)"
    )
    parser.add_argument(
        "--sahi-config",
        type=Path,
        default=Path("config/sahi.yaml"),
        help="Path to SAHI config YAML (optional)"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip creating visualization images"
    )

    args = parser.parse_args()

    # Setup logger
    setup_logger(level="INFO")

    # Verify model exists
    if not args.model.exists():
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)

    # Verify input exists
    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        sys.exit(1)

    try:
        run_inference(
            model_path=args.model,
            input_path=args.input,
            output_dir=args.output,
            sahi_config_path=args.sahi_config,
            visualize=not args.no_visualize
        )

        logger.info(f"\n{'=' * 80}")
        logger.info(f"✓ Inference complete!")
        logger.info(f"{'=' * 80}\n")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
