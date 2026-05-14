"""
CUDA device management and memory monitoring.
Ensures optimal GPU utilization and prevents OOM errors.
"""
import gc
import torch
import psutil
from typing import Optional, Dict
from loguru import logger

try:
    import py3nvml.py3nvml as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logger.warning("py3nvml not available. GPU monitoring will be limited.")


class DeviceManager:
    """Manages CUDA device and memory."""

    def __init__(self, device: str = "cuda:0"):
        """
        Initialize device manager.

        Args:
            device: Device string (e.g., 'cuda:0', 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            self.device_id = int(device.split(':')[1]) if ':' in device else 0
            self.device_name = torch.cuda.get_device_name(self.device_id)
            self.total_vram_gb = torch.cuda.get_device_properties(self.device_id).total_memory / (1024**3)

            if NVML_AVAILABLE:
                nvml.nvmlInit()
                self.nvml_handle = nvml.nvmlDeviceGetHandleByIndex(self.device_id)
            else:
                self.nvml_handle = None

            logger.info(f"Using device: {self.device_name} (Total VRAM: {self.total_vram_gb:.2f} GB)")
        else:
            logger.warning("CUDA not available. Using CPU.")
            self.device_id = None
            self.device_name = "CPU"
            self.total_vram_gb = 0
            self.nvml_handle = None

    def get_memory_info(self) -> Dict[str, float]:
        """
        Get current CUDA memory usage.

        Returns:
            Dictionary with memory statistics in GB
        """
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "free": 0, "total": 0}

        allocated = torch.cuda.memory_allocated(self.device_id) / (1024**3)
        reserved = torch.cuda.memory_reserved(self.device_id) / (1024**3)
        free = self.total_vram_gb - allocated

        return {
            "allocated": allocated,
            "reserved": reserved,
            "free": free,
            "total": self.total_vram_gb
        }

    def get_nvml_memory_info(self) -> Dict[str, float]:
        """
        Get detailed memory info from NVML (more accurate).

        Returns:
            Dictionary with NVML memory statistics in GB
        """
        if not NVML_AVAILABLE or self.nvml_handle is None:
            return self.get_memory_info()

        try:
            mem_info = nvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
            return {
                "total": mem_info.total / (1024**3),
                "used": mem_info.used / (1024**3),
                "free": mem_info.free / (1024**3)
            }
        except Exception as e:
            logger.warning(f"NVML memory query failed: {e}")
            return self.get_memory_info()

    def get_gpu_utilization(self) -> float:
        """
        Get GPU utilization percentage.

        Returns:
            GPU utilization (0-100)
        """
        if not NVML_AVAILABLE or self.nvml_handle is None:
            return 0.0

        try:
            util = nvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
            return util.gpu
        except Exception as e:
            logger.warning(f"GPU utilization query failed: {e}")
            return 0.0

    def cleanup_cuda_cache(self, aggressive: bool = False):
        """
        Clean up CUDA cache to free memory.

        Args:
            aggressive: If True, also runs garbage collection
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            if aggressive:
                gc.collect()
                torch.cuda.empty_cache()

            mem_info = self.get_memory_info()
            logger.debug(f"CUDA cache cleaned. Free VRAM: {mem_info['free']:.2f} GB")

    def check_memory_available(self, required_gb: float, safety_margin_gb: float = 1.0) -> bool:
        """
        Check if enough VRAM is available.

        Args:
            required_gb: Required VRAM in GB
            safety_margin_gb: Additional safety margin

        Returns:
            True if enough memory available
        """
        mem_info = self.get_memory_info()
        available = mem_info['free'] - safety_margin_gb

        if available < required_gb:
            logger.warning(
                f"Insufficient VRAM: Required {required_gb:.2f} GB, "
                f"Available {available:.2f} GB (with {safety_margin_gb:.2f} GB margin)"
            )
            return False

        return True

    def monitor_memory(self, prefix: str = ""):
        """
        Log current memory usage.

        Args:
            prefix: Prefix for log message
        """
        if torch.cuda.is_available():
            mem_info = self.get_memory_info()
            nvml_info = self.get_nvml_memory_info()
            gpu_util = self.get_gpu_utilization()

            logger.info(
                f"{prefix} VRAM: {mem_info['allocated']:.2f}/{mem_info['total']:.2f} GB "
                f"(Reserved: {mem_info['reserved']:.2f} GB, Free: {mem_info['free']:.2f} GB) "
                f"| GPU Util: {gpu_util:.1f}%"
            )
        else:
            # Log system RAM usage
            ram = psutil.virtual_memory()
            logger.info(
                f"{prefix} RAM: {ram.used / (1024**3):.2f}/{ram.total / (1024**3):.2f} GB "
                f"({ram.percent:.1f}%)"
            )

    def __del__(self):
        """Cleanup NVML on destruction."""
        if NVML_AVAILABLE and self.nvml_handle is not None:
            try:
                nvml.nvmlShutdown()
            except:
                pass


def get_device(device_str: str = "cuda:0") -> torch.device:
    """
    Get torch device.

    Args:
        device_str: Device string

    Returns:
        torch.device instance
    """
    if torch.cuda.is_available():
        return torch.device(device_str)
    else:
        logger.warning("CUDA not available, falling back to CPU")
        return torch.device("cpu")
