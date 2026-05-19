#!/usr/bin/env python3
"""Test if config loader properly reads augmentation parameters."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.core.config_loader import load_training_config

def test_config():
    config_path = Path("config/pest_detection_fast.yaml")

    print(f"Loading config from: {config_path}")

    try:
        config = load_training_config(config_path)
        config_dict = config.model_dump()

        print("\n✓ Config loaded successfully!")
        print("\nAugmentation Parameters:")
        print(f"  hsv_h:      {config_dict['hsv_h']} (expected: 0.15)")
        print(f"  hsv_s:      {config_dict['hsv_s']} (expected: 0.8)")
        print(f"  hsv_v:      {config_dict['hsv_v']} (expected: 0.5)")
        print(f"  degrees:    {config_dict['degrees']} (expected: 45.0)")
        print(f"  translate:  {config_dict['translate']} (expected: 0.2)")
        print(f"  scale:      {config_dict['scale']} (expected: 0.7)")
        print(f"  shear:      {config_dict['shear']} (expected: 5.0)")
        print(f"  perspective:{config_dict['perspective']} (expected: 0.001)")
        print(f"  flipud:     {config_dict['flipud']} (expected: 0.3)")
        print(f"  fliplr:     {config_dict['fliplr']} (expected: 0.5)")
        print(f"  mosaic:     {config_dict['mosaic']} (expected: 1.0)")
        print(f"  mixup:      {config_dict['mixup']} (expected: 0.3)")
        print(f"  copy_paste: {config_dict['copy_paste']} (expected: 0.3)")

        # Validation
        errors = []
        if config_dict['hsv_h'] != 0.15:
            errors.append(f"hsv_h mismatch: {config_dict['hsv_h']} != 0.15")
        if config_dict['degrees'] != 45.0:
            errors.append(f"degrees mismatch: {config_dict['degrees']} != 45.0")
        if config_dict['mixup'] != 0.3:
            errors.append(f"mixup mismatch: {config_dict['mixup']} != 0.3")
        if config_dict['copy_paste'] != 0.3:
            errors.append(f"copy_paste mismatch: {config_dict['copy_paste']} != 0.3")

        if errors:
            print("\n✗ VALIDATION ERRORS:")
            for error in errors:
                print(f"  - {error}")
            return False
        else:
            print("\n✓ All augmentation parameters validated correctly!")
            return True

    except Exception as e:
        print(f"\n✗ Config loading FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config()
    sys.exit(0 if success else 1)
