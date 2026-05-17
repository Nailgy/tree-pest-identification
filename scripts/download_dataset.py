#!/usr/bin/env python3
"""Download Roboflow dataset via direct ZIP download."""
import argparse
from pathlib import Path
import sys
import zipfile
import requests
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
from src.core.logger import setup_logger


def download_roboflow_zip(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    output_dir: Path
) -> Path:
    """Download Roboflow dataset as ZIP and extract."""
    logger.info(f"Downloading {workspace}/{project} v{version}")

    # Construct download URL
    url = f"https://app.roboflow.com/api/project/{project}/dataset/{version}/download?api_key={api_key}"

    logger.info(f"Download URL: {url}")

    # Download ZIP
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "dataset.zip"

    logger.info(f"Downloading to: {zip_path}")

    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise ValueError(f"Download failed: {response.status_code} - {response.text}")

    # Save ZIP
    total_size = int(response.headers.get('content-length', 0))
    with open(zip_path, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

    logger.info(f"ZIP saved: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Check if it's actually a zip
    if zip_path.stat().st_size < 1000:
        # Too small, probably an error message
        with open(zip_path, 'r', errors='ignore') as f:
            content = f.read()
            logger.error(f"Response content: {content}")
            if 'error' in content.lower():
                raise ValueError(f"API error: {content}")
            raise ValueError(f"Downloaded file too small: {zip_path.stat().st_size} bytes")

    # Extract
    logger.info(f"Extracting to: {output_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    # Verify
    data_yaml = output_dir / "data.yaml"
    if data_yaml.exists():
        logger.info(f"✓ Dataset extracted successfully")
        logger.info(f"✓ data.yaml found at: {data_yaml}")
        return output_dir
    else:
        raise FileNotFoundError(f"data.yaml not found after extraction")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("data/raw/pests"))

    args = parser.parse_args()
    setup_logger(level="INFO")

    try:
        download_roboflow_zip(
            api_key=args.api_key,
            workspace=args.workspace,
            project=args.project,
            version=args.version,
            output_dir=args.output
        )
        logger.info(f"✓ Ready for training: {args.output}")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
