"""
Memory optimizer for training on 8GB VRAM constraint.
Provides dynamic batch sizing, aggressive garbage collection, and VRAM monitoring.
"""
import gc
import torch
from typing import Dict, Optional, Callable
from loguru import logger
from src.core.device_manager import DeviceManager


class MemoryOptimizer:
    """
    Optimizes memory usage during training and inference.
    Critical for 8GB VRAM constraint with YOLO11m.
    """

    def __init__(
        self,
        device_manager: DeviceManager,
        alert_threshold_gb: float = 7.5,
        aggressive_cleanup: bool = True
    ):
        """
        Initialize memory optimizer.

        Args:
            device_manager: DeviceManager instance
            alert_threshold_gb: Alert when VRAM usage exceeds this
            aggressive_cleanup: Enable aggressive garbage collection
        """
        self.device_manager = device_manager
        self.alert_threshold_gb = alert_threshold_gb
        self.aggressive_cleanup = aggressive_cleanup

    def calculate_optimal_batch_size(
        self,
        img_size: int,
        num_classes: int,
        base_vram_gb: float = 8.0,
        reserved_vram_gb: float = 2.0
    ) -> int:
        """
        Calculate optimal batch size based on VRAM constraints.

        Memory model:
        - YOLO11m base: ~1.5GB
        - Per-image: ~50MB * (img_size/640)^2
        - Class head: ~0.01MB per class

        Args:
            img_size: Input image size
            num_classes: Number of detection classes
            base_vram_gb: Total VRAM available
            reserved_vram_gb: VRAM to reserve for OS/driver

        Returns:
            Optimal batch size (clamped to [1, 16])
        """
        available_vram = base_vram_gb - reserved_vram_gb

        # Model base memory
        model_base_memory = 1.5

        # Per-image memory (scales quadratically with img_size)
        per_image_memory = 0.05 * (img_size / 640) ** 2

        # Class head overhead
        class_overhead = 0.01 * num_classes / 1000

        # Calculate batch size
        batch_size = int(
            (available_vram - model_base_memory - class_overhead) / per_image_memory
        )

        # Clamp to reasonable range
        batch_size = max(1, min(batch_size, 16))

        logger.info(
            f"Calculated optimal batch size: {batch_size} "
            f"(img_size={img_size}, num_classes={num_classes}, "
            f"available_vram={available_vram:.2f}GB)"
        )

        return batch_size

    def calculate_gradient_accumulation(
        self,
        actual_batch_size: int,
        target_effective_batch: int = 16
    ) -> int:
        """
        Calculate gradient accumulation steps to simulate larger batch.

        Args:
            actual_batch_size: Actual batch size that fits in VRAM
            target_effective_batch: Target effective batch size

        Returns:
            Gradient accumulation steps
        """
        accumulate = max(1, target_effective_batch // actual_batch_size)
        logger.info(
            f"Gradient accumulation: {accumulate} steps "
            f"(actual_batch={actual_batch_size}, effective_batch={actual_batch_size * accumulate})"
        )
        return accumulate

    def cleanup_cuda_cache(self, aggressive: bool = None):
        """
        Clean CUDA cache and run garbage collection.

        Args:
            aggressive: Override aggressive cleanup setting
        """
        if aggressive is None:
            aggressive = self.aggressive_cleanup

        self.device_manager.cleanup_cuda_cache(aggressive=aggressive)

    def monitor_memory_usage(self, prefix: str = ""):
        """
        Monitor and log memory usage. Alert if exceeding threshold.

        Args:
            prefix: Prefix for log message
        """
        mem_info = self.device_manager.get_memory_info()

        # Alert if exceeding threshold
        if mem_info['allocated'] > self.alert_threshold_gb:
            logger.warning(
                f"{prefix} VRAM usage HIGH: {mem_info['allocated']:.2f} GB "
                f"(threshold: {self.alert_threshold_gb:.2f} GB)"
            )
        else:
            self.device_manager.monitor_memory(prefix=prefix)

    def create_cleanup_callback(self) -> Dict[str, Callable]:
        """
        Create training callbacks for memory management.

        Returns:
            Dictionary of callback functions for YOLO trainer
        """
        def on_train_batch_end(trainer):
            """Called after each training batch."""
            # Cleanup every 10 epochs
            if trainer.epoch % 10 == 0 and trainer.batch_idx == 0:
                self.cleanup_cuda_cache(aggressive=False)

        def on_val_start(trainer):
            """Called before validation."""
            self.cleanup_cuda_cache(aggressive=True)

        def on_val_end(trainer):
            """Called after validation."""
            self.cleanup_cuda_cache(aggressive=True)
            self.monitor_memory_usage(prefix=f"[Epoch {trainer.epoch}]")

        def on_train_epoch_end(trainer):
            """Called at end of each epoch."""
            if trainer.epoch % 5 == 0:
                self.monitor_memory_usage(prefix=f"[Epoch {trainer.epoch} End]")

        return {
            'on_train_batch_end': on_train_batch_end,
            'on_val_start': on_val_start,
            'on_val_end': on_val_end,
            'on_train_epoch_end': on_train_epoch_end
        }

    def optimize_training_config(
        self,
        config: Dict,
        num_classes: int
    ) -> Dict:
        """
        Optimize training configuration for VRAM constraints.

        Args:
            config: Training configuration dictionary
            num_classes: Number of detection classes

        Returns:
            Optimized configuration
        """
        # Calculate optimal batch size
        optimal_batch = self.calculate_optimal_batch_size(
            img_size=config.get('img_size', 640),
            num_classes=num_classes
        )

        # If calculated batch is smaller than config, update it
        if optimal_batch < config.get('batch_size', 8):
            logger.warning(
                f"Reducing batch_size from {config['batch_size']} to {optimal_batch} "
                f"due to VRAM constraints"
            )
            config['batch_size'] = optimal_batch

            # Increase gradient accumulation to maintain effective batch
            target_effective = 16
            config['accumulate'] = self.calculate_gradient_accumulation(
                actual_batch_size=optimal_batch,
                target_effective_batch=target_effective
            )

        # Ensure mixed precision is enabled
        if not config.get('amp', False):
            logger.warning("Enabling AMP (mixed precision) for VRAM optimization")
            config['amp'] = True

        return config

    def get_memory_stats(self) -> Dict:
        """
        Get comprehensive memory statistics.

        Returns:
            Dictionary with memory stats
        """
        mem_info = self.device_manager.get_memory_info()
        nvml_info = self.device_manager.get_nvml_memory_info()
        gpu_util = self.device_manager.get_gpu_utilization()

        return {
            'torch_allocated_gb': mem_info['allocated'],
            'torch_reserved_gb': mem_info['reserved'],
            'torch_free_gb': mem_info['free'],
            'nvml_used_gb': nvml_info.get('used', 0),
            'nvml_free_gb': nvml_info.get('free', 0),
            'gpu_utilization_pct': gpu_util,
            'alert': mem_info['allocated'] > self.alert_threshold_gb
        }


def setup_mixed_precision() -> bool:
    """
    Setup Automatic Mixed Precision (AMP) for training.
    Reduces VRAM usage by ~40% with minimal accuracy impact.

    Returns:
        True if AMP is available and enabled
    """
    if torch.cuda.is_available() and torch.cuda.is_amp_available():
        logger.info("AMP (Automatic Mixed Precision) enabled - ~40% VRAM reduction")
        return True
    else:
        logger.warning("AMP not available")
        return False


def estimate_vram_usage(
    img_size: int,
    batch_size: int,
    num_classes: int,
    model_size: str = "yolo11m"
) -> float:
    """
    Estimate VRAM usage for training configuration.

    Args:
        img_size: Input image size
        batch_size: Batch size
        num_classes: Number of classes
        model_size: Model size (yolo11n, yolo11s, yolo11m, yolo11l, yolo11x)

    Returns:
        Estimated VRAM usage in GB
    """
    # Base model memory (YOLO11m)
    model_base = {
        'yolo11n': 0.8,
        'yolo11s': 1.0,
        'yolo11m': 1.5,
        'yolo11l': 2.0,
        'yolo11x': 2.5
    }.get(model_size, 1.5)

    # Per-image memory (scales quadratically)
    per_image = 0.05 * (img_size / 640) ** 2

    # Class head overhead
    class_overhead = 0.01 * num_classes / 1000

    # Total estimate
    total = model_base + (per_image * batch_size) + class_overhead

    logger.info(
        f"Estimated VRAM: {total:.2f} GB "
        f"(model={model_base:.2f}, images={per_image * batch_size:.2f}, classes={class_overhead:.2f})"
    )

    return total
