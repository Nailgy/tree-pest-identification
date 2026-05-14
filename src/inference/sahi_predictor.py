"""
SAHI (Slicing Aided Hyper Inference) predictor for 4K images.
Enables inference on high-resolution images without CUDA OOM errors.
"""
from pathlib import Path
from typing import List, Optional, Dict, Union
import gc
import torch
import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.models.yolov8 import Yolov8DetectionModel
from loguru import logger

from src.core.device_manager import DeviceManager


class SAHIPredictor:
    """
    SAHI-based predictor for high-resolution image inference.
    Optimized for 4K images on 8GB VRAM.
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "cuda:0",
        confidence_threshold: float = 0.25,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        postprocess_type: str = "NMS",
        postprocess_match_metric: str = "IOS",
        postprocess_match_threshold: float = 0.5
    ):
        """
        Initialize SAHI predictor.

        Args:
            model_path: Path to trained YOLO model
            device: Device to use
            confidence_threshold: Confidence threshold for detections
            slice_height: Height of each slice in pixels
            slice_width: Width of each slice in pixels
            overlap_height_ratio: Vertical overlap ratio
            overlap_width_ratio: Horizontal overlap ratio
            postprocess_type: Post-processing type (NMS, GREEDYNMM)
            postprocess_match_metric: Matching metric (IOU, IOS)
            postprocess_match_threshold: Matching threshold
        """
        self.model_path = Path(model_path)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.postprocess_type = postprocess_type
        self.postprocess_match_metric = postprocess_match_metric
        self.postprocess_match_threshold = postprocess_match_threshold

        # Initialize device manager
        self.device_manager = DeviceManager(device=device)

        # Initialize SAHI detection model
        self._init_model()

        logger.info(
            f"SAHI Predictor initialized:\n"
            f"  - Model: {self.model_path}\n"
            f"  - Device: {self.device}\n"
            f"  - Slice size: {slice_height}×{slice_width}\n"
            f"  - Overlap: {overlap_height_ratio:.0%} × {overlap_width_ratio:.0%}\n"
            f"  - Confidence: {confidence_threshold}"
        )

    def _init_model(self):
        """Initialize SAHI detection model."""
        try:
            self.detection_model = Yolov8DetectionModel(
                model_path=str(self.model_path),
                confidence_threshold=self.confidence_threshold,
                device=self.device,
                load_at_init=True
            )
            logger.info(f"Model loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict_image(
        self,
        image_path: Union[Path, str],
        visualize: bool = False,
        output_path: Optional[Path] = None
    ) -> Dict:
        """
        Run SAHI inference on a single image.

        Args:
            image_path: Path to input image
            visualize: Whether to create visualization
            output_path: Path to save visualization (if visualize=True)

        Returns:
            Dictionary with predictions and metadata
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Pre-inference cleanup
        self._cleanup_memory()

        # Get image dimensions
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]

        logger.info(f"Processing: {image_path.name} ({width}×{height})")

        # Calculate number of slices
        num_slices_h = int(np.ceil(height / (self.slice_height * (1 - self.overlap_height_ratio))))
        num_slices_w = int(np.ceil(width / (self.slice_width * (1 - self.overlap_width_ratio))))
        total_slices = num_slices_h * num_slices_w

        logger.info(f"Slicing: {num_slices_h}×{num_slices_w} = {total_slices} slices")

        # Run SAHI prediction
        try:
            result = get_sliced_prediction(
                str(image_path),
                self.detection_model,
                slice_height=self.slice_height,
                slice_width=self.slice_width,
                overlap_height_ratio=self.overlap_height_ratio,
                overlap_width_ratio=self.overlap_width_ratio,
                postprocess_type=self.postprocess_type,
                postprocess_match_metric=self.postprocess_match_metric,
                postprocess_match_threshold=self.postprocess_match_threshold,
                verbose=0
            )

            # Extract predictions
            predictions = []
            for obj in result.object_prediction_list:
                predictions.append({
                    'bbox': [obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy],
                    'class_id': obj.category.id,
                    'class_name': obj.category.name,
                    'confidence': obj.score.value
                })

            logger.info(f"Detected {len(predictions)} objects")

            # Visualize if requested
            if visualize:
                self._visualize_predictions(
                    image_path=image_path,
                    predictions=predictions,
                    output_path=output_path
                )

            # Post-inference cleanup
            self._cleanup_memory()

            return {
                'image_path': str(image_path),
                'image_size': {'width': width, 'height': height},
                'num_slices': total_slices,
                'num_detections': len(predictions),
                'predictions': predictions
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            self._cleanup_memory()
            raise

    def predict_batch(
        self,
        image_paths: List[Path],
        visualize: bool = False,
        output_dir: Optional[Path] = None
    ) -> List[Dict]:
        """
        Run SAHI inference on batch of images.

        Args:
            image_paths: List of image paths
            visualize: Whether to create visualizations
            output_dir: Directory to save visualizations

        Returns:
            List of prediction dictionaries
        """
        results = []

        for i, image_path in enumerate(image_paths):
            logger.info(f"[{i+1}/{len(image_paths)}] Processing: {image_path.name}")

            output_path = None
            if visualize and output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{image_path.stem}_predicted.jpg"

            try:
                result = self.predict_image(
                    image_path=image_path,
                    visualize=visualize,
                    output_path=output_path
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to process {image_path.name}: {e}")
                results.append({
                    'image_path': str(image_path),
                    'error': str(e)
                })

        logger.info(f"Batch complete: {len(results)}/{len(image_paths)} images processed")
        return results

    def _visualize_predictions(
        self,
        image_path: Path,
        predictions: List[Dict],
        output_path: Optional[Path] = None
    ):
        """
        Visualize predictions on image.

        Args:
            image_path: Path to input image
            predictions: List of prediction dictionaries
            output_path: Path to save visualization
        """
        # Load image
        image = cv2.imread(str(image_path))

        # Draw bounding boxes
        for pred in predictions:
            bbox = pred['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            conf = pred['confidence']
            class_name = pred['class_name']

            # Draw box
            color = self._get_color(pred['class_id'])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        # Save visualization
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), image)
            logger.info(f"Visualization saved: {output_path}")

    def _get_color(self, class_id: int) -> tuple:
        """Get color for class ID."""
        np.random.seed(class_id)
        return tuple(map(int, np.random.randint(0, 255, 3)))

    def _cleanup_memory(self):
        """Cleanup CUDA cache and garbage collect."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def get_vram_usage(self) -> Dict:
        """Get current VRAM usage."""
        return self.device_manager.get_memory_info()


def load_sahi_config(config_path: Path) -> Dict:
    """
    Load SAHI configuration from YAML.

    Args:
        config_path: Path to SAHI config YAML

    Returns:
        Configuration dictionary
    """
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
