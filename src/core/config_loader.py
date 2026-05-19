"""
YAML configuration loader with type-safe validation using Pydantic.
Ensures configurations are valid before training/inference.
"""
from pathlib import Path
from typing import Any, Dict
import yaml
from pydantic import BaseModel, Field, validator


class TrainingConfig(BaseModel):
    """Type-safe training configuration."""

    # Model
    model: str = Field(..., description="Path to YOLO model or pretrained weights")
    task: str = Field(default="detect", description="Task type")

    # Training
    epochs: int = Field(ge=1, le=1000, description="Number of epochs")
    patience: int = Field(ge=1, le=100, description="Early stopping patience")
    save_period: int = Field(ge=-1, description="Save checkpoint every N epochs")

    # Hardware
    img_size: int = Field(ge=320, le=1280, description="Input image size")
    batch_size: int = Field(ge=1, le=64, description="Batch size")
    workers: int = Field(ge=0, le=32, description="Number of data loading workers")
    accumulate: int = Field(ge=1, le=64, default=1, description="Gradient accumulation steps")
    amp: bool = Field(default=True, description="Automatic Mixed Precision")
    device: str = Field(default="cuda:0", description="Device to use")

    # Optimizer
    optimizer: str = Field(default="AdamW", description="Optimizer")
    lr0: float = Field(gt=0, le=1, description="Initial learning rate")
    lrf: float = Field(gt=0, le=1, description="Final learning rate")
    momentum: float = Field(ge=0, le=1, default=0.937)
    weight_decay: float = Field(ge=0, default=0.0005)
    warmup_epochs: int = Field(ge=0, default=3)
    warmup_momentum: float = Field(ge=0, le=1, default=0.8)
    warmup_bias_lr: float = Field(ge=0, le=1, default=0.1)
    cos_lr: bool = Field(default=True, description="Use cosine LR scheduler")

    # Augmentation (YOLO built-in)
    hsv_h: float = Field(ge=0, le=1, default=0.015, description="HSV hue augmentation")
    hsv_s: float = Field(ge=0, le=1, default=0.7, description="HSV saturation augmentation")
    hsv_v: float = Field(ge=0, le=1, default=0.4, description="HSV value augmentation")
    degrees: float = Field(ge=0, le=360, default=0.0, description="Rotation degrees")
    translate: float = Field(ge=0, le=0.5, default=0.1, description="Translation fraction")
    scale: float = Field(ge=0, le=1, default=0.5, description="Scale factor")
    shear: float = Field(ge=0, le=10, default=0.0, description="Shear degrees")
    perspective: float = Field(ge=0, le=0.001, default=0.0, description="Perspective distortion")
    flipud: float = Field(ge=0, le=1, default=0.0, description="Vertical flip probability")
    fliplr: float = Field(ge=0, le=1, default=0.5, description="Horizontal flip probability")
    mosaic: float = Field(ge=0, le=1, default=1.0, description="Mosaic augmentation")
    mixup: float = Field(ge=0, le=1, default=0.0, description="Mixup augmentation")
    copy_paste: float = Field(ge=0, le=1, default=0.0, description="Copy-paste augmentation")
    close_mosaic: int = Field(ge=0, le=50, default=10, description="Disable mosaic last N epochs")

    # Validation
    val: bool = Field(default=True)
    plots: bool = Field(default=True)
    save_json: bool = Field(default=False)

    # Project
    project: str = Field(default="models/pest_detection")
    name: str = Field(default="train")
    exist_ok: bool = Field(default=False)

    class Config:
        validate_assignment = True


class AugmentationConfig(BaseModel):
    """Type-safe augmentation configuration."""

    minority_percentile: int = Field(ge=1, le=50, default=10)
    target_samples_per_class: int = Field(ge=10, le=10000, default=100)
    severity: str = Field(default="medium", pattern="^(light|medium|heavy)$")
    preserve_aspect_ratio: bool = Field(default=True)
    interpolation: str = Field(default="bilinear")

    class Config:
        validate_assignment = True


class SAHIConfig(BaseModel):
    """Type-safe SAHI configuration."""

    slice_height: int = Field(ge=320, le=1280, default=640)
    slice_width: int = Field(ge=320, le=1280, default=640)
    overlap_height_ratio: float = Field(ge=0, le=0.9, default=0.2)
    overlap_width_ratio: float = Field(ge=0, le=0.9, default=0.2)
    postprocess_type: str = Field(default="NMS")
    postprocess_match_metric: str = Field(default="IOS")
    postprocess_match_threshold: float = Field(ge=0, le=1, default=0.5)
    confidence_threshold: float = Field(ge=0, le=1, default=0.25)
    iou_threshold: float = Field(ge=0, le=1, default=0.45)
    device: str = Field(default="cuda:0")
    verbose: int = Field(ge=0, le=2, default=1)

    class Config:
        validate_assignment = True


def load_yaml(config_path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML file

    Returns:
        Configuration dictionary
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def load_training_config(config_path: Path) -> TrainingConfig:
    """Load and validate training configuration."""
    config_dict = load_yaml(config_path)
    return TrainingConfig(**config_dict)


def load_augmentation_config(config_path: Path) -> AugmentationConfig:
    """Load and validate augmentation configuration."""
    config_dict = load_yaml(config_path)
    return AugmentationConfig(**config_dict)


def load_sahi_config(config_path: Path) -> SAHIConfig:
    """Load and validate SAHI configuration."""
    config_dict = load_yaml(config_path)
    return SAHIConfig(**config_dict)


def save_yaml(config: Dict[str, Any], save_path: Path):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save YAML
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
