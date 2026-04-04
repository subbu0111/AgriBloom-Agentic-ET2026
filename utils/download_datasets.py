"""
Dataset Download and Ingestion Pipeline
Downloads datasets from Kaggle and creates training manifests

Supported Datasets:
- PlantVillage (Maize, Tomato, Potato)
- Rice Disease Dataset
- Wheat Plant Diseases
- Finger Millet (Ragi) Dataset
- Sugarcane Disease Dataset
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

# Kaggle dataset configurations
KAGGLE_DATASETS = {
    "plantvillage": {
        "kaggle_id": "emmarex/plantdisease",
        "description": "PlantVillage Dataset (Maize, Tomato, Potato)",
        "class_mappings": {
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("maize", "gray_leaf_spot"),
            "Corn_(maize)___Common_rust_": ("maize", "common_rust"),
            "Corn_(maize)___Northern_Leaf_Blight": ("maize", "blight"),
            "Corn_(maize)___healthy": ("maize", "healthy"),
            "Tomato___Bacterial_spot": ("tomato", "bacterial_spot"),
            "Tomato___Early_blight": ("tomato", "early_blight"),
            "Tomato___Late_blight": ("tomato", "late_blight"),
            "Tomato___Leaf_Mold": ("tomato", "leaf_mold"),
            "Tomato___Septoria_leaf_spot": ("tomato", "septoria_leaf_spot"),
            "Tomato___Spider_mites Two-spotted_spider_mite": ("tomato", "spider_mites"),
            "Tomato___Target_Spot": ("tomato", "target_spot"),
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("tomato", "yellow_leaf_curl"),
            "Tomato___Tomato_mosaic_virus": ("tomato", "mosaic_virus"),
            "Tomato___healthy": ("tomato", "healthy"),
            "Potato___Early_blight": ("potato", "early_blight"),
            "Potato___Late_blight": ("potato", "late_blight"),
            "Potato___healthy": ("potato", "healthy"),
        },
    },
    "rice": {
        "kaggle_id": "anshulm257/rice-disease-dataset",
        "description": "Rice Disease Dataset",
        "class_mappings": {
            "Bacterial_Leaf_Blight": ("rice", "bacterial_leaf_blight"),
            "Brown_Spot": ("rice", "brown_spot"),
            "Leaf_Blast": ("rice", "leaf_blast"),
            "Leaf_Scald": ("rice", "leaf_scald"),
            "Narrow_Brown_Spot": ("rice", "narrow_brown_spot"),
            "Healthy": ("rice", "healthy"),
        },
    },
    "wheat": {
        "kaggle_id": "kushagra3204/wheat-plant-diseases",
        "description": "Wheat Plant Diseases",
        "class_mappings": {
            "Brown_Rust": ("wheat", "brown_rust"),
            "Leaf_Rust": ("wheat", "leaf_rust"),
            "Septoria": ("wheat", "septoria"),
            "Yellow_Rust": ("wheat", "yellow_rust"),
            "Healthy": ("wheat", "healthy"),
        },
    },
    "ragi": {
        "kaggle_id": "prajwalbax/finger-millet-ragi-dataset",
        "description": "Finger Millet (Ragi) Dataset",
        "class_mappings": {
            "Blast": ("ragi", "blast"),
            "Brown_Spot": ("ragi", "brown_spot"),
            "Healthy": ("ragi", "healthy"),
        },
    },
    "sugarcane": {
        "kaggle_id": "prabhakaransoundar/sugarcane-disease-dataset",
        "description": "Sugarcane Disease Dataset",
        "class_mappings": {
            "Bacterial_Blight": ("sugarcane", "bacterial_blight"),
            "Red_Rot": ("sugarcane", "red_rot"),
            "Rust": ("sugarcane", "rust"),
            "Healthy": ("sugarcane", "healthy"),
        },
    },
}


def setup_kaggle_credentials() -> bool:
    """Check and setup Kaggle credentials."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    if kaggle_json.exists():
        logger.info("Kaggle credentials found")
        return True

    # Check environment variables
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        logger.info("Kaggle credentials found in environment")
        return True

    logger.error(
        "Kaggle credentials not found!\n"
        "Please follow these steps:\n"
        "1. Go to https://www.kaggle.com/settings\n"
        "2. Click 'Create New Token' under API section\n"
        "3. Save kaggle.json to ~/.kaggle/kaggle.json\n"
        "4. Run: chmod 600 ~/.kaggle/kaggle.json"
    )
    return False


def download_dataset(dataset_key: str, output_dir: Path) -> bool:
    """Download a dataset from Kaggle."""
    if dataset_key not in KAGGLE_DATASETS:
        logger.error(f"Unknown dataset: {dataset_key}")
        return False

    config = KAGGLE_DATASETS[dataset_key]
    kaggle_id = config["kaggle_id"]

    download_path = output_dir / dataset_key
    download_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {config['description']}...")

    try:
        result = subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", kaggle_id,
                "-p", str(download_path),
                "--unzip",
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            logger.error(f"Download failed: {result.stderr}")
            return False

        logger.info(f"Downloaded {dataset_key} to {download_path}")
        return True

    except subprocess.TimeoutExpired:
        logger.error("Download timed out")
        return False
    except FileNotFoundError:
        logger.error("Kaggle CLI not found. Install with: pip install kaggle")
        return False


def find_image_directories(base_path: Path) -> list[Path]:
    """Find directories containing images."""
    image_dirs = []

    for path in base_path.rglob("*"):
        if path.is_dir():
            # Check if directory contains images
            images = list(path.glob("*.jpg")) + list(path.glob("*.jpeg")) + list(path.glob("*.png"))
            if images:
                image_dirs.append(path)

    return image_dirs


def parse_class_from_path(
    image_path: Path,
    class_mappings: dict[str, tuple[str, str]],
) -> tuple[str, str] | None:
    """Parse crop and disease from image path."""
    # Try to match against known class names
    path_str = str(image_path).replace("\\", "/")

    for class_name, (crop, disease) in class_mappings.items():
        # Normalize for comparison
        normalized_class = class_name.lower().replace(" ", "_").replace("-", "_")

        if normalized_class in path_str.lower():
            return (crop, disease)

        # Also try parent directory name
        parent_name = image_path.parent.name.lower().replace(" ", "_").replace("-", "_")
        if normalized_class in parent_name or class_name.lower() in parent_name:
            return (crop, disease)

    return None


def validate_image(image_path: Path, min_size: int = 32) -> bool:
    """Validate that an image is readable and meets minimum requirements."""
    try:
        with Image.open(image_path) as img:
            if img.size[0] < min_size or img.size[1] < min_size:
                return False
            return True
    except Exception:
        return False


def ingest_dataset(
    dataset_key: str,
    data_dir: Path,
    manifest_dir: Path,
) -> list[dict[str, Any]]:
    """Ingest a downloaded dataset and return examples."""
    if dataset_key not in KAGGLE_DATASETS:
        logger.error(f"Unknown dataset: {dataset_key}")
        return []

    config = KAGGLE_DATASETS[dataset_key]
    class_mappings = config["class_mappings"]

    dataset_path = data_dir / dataset_key
    if not dataset_path.exists():
        logger.warning(f"Dataset directory not found: {dataset_path}")
        return []

    examples = []
    image_dirs = find_image_directories(dataset_path)

    logger.info(f"Found {len(image_dirs)} image directories in {dataset_key}")

    for img_dir in image_dirs:
        # Get all images
        images = (
            list(img_dir.glob("*.jpg")) +
            list(img_dir.glob("*.jpeg")) +
            list(img_dir.glob("*.png")) +
            list(img_dir.glob("*.JPG")) +
            list(img_dir.glob("*.JPEG")) +
            list(img_dir.glob("*.PNG"))
        )

        for img_path in images:
            # Parse class from path
            parsed = parse_class_from_path(img_path, class_mappings)
            if parsed is None:
                continue

            crop, disease = parsed

            # Validate image
            if not validate_image(img_path):
                continue

            examples.append({
                "image_path": str(img_path.absolute()),
                "crop": crop,
                "disease": disease,
                "label": f"{crop}_{disease}",
                "source": dataset_key,
            })

    logger.info(f"Ingested {len(examples)} valid images from {dataset_key}")
    return examples


def create_train_val_test_split(
    examples: list[dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split examples into train/val/test sets with stratification."""
    random.seed(seed)

    # Group by label for stratified splitting
    label_groups = {}
    for ex in examples:
        label = ex["label"]
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(ex)

    train_examples = []
    val_examples = []
    test_examples = []

    for label, group in label_groups.items():
        random.shuffle(group)

        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_examples.extend(group[:n_train])
        val_examples.extend(group[n_train:n_train + n_val])
        test_examples.extend(group[n_train + n_val:])

    # Shuffle each split
    random.shuffle(train_examples)
    random.shuffle(val_examples)
    random.shuffle(test_examples)

    return train_examples, val_examples, test_examples


def save_manifest(examples: list[dict[str, Any]], output_path: Path) -> None:
    """Save examples to CSV manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(examples)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(examples)} examples to {output_path}")


def print_dataset_stats(examples: list[dict[str, Any]], name: str) -> None:
    """Print dataset statistics."""
    df = pd.DataFrame(examples)

    print(f"\n{'=' * 60}")
    print(f"{name} Statistics")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(examples)}")

    print("\nBy Crop:")
    for crop, count in df["crop"].value_counts().items():
        print(f"  {crop}: {count}")

    print("\nBy Label (Top 15):")
    for label, count in df["label"].value_counts().head(15).items():
        print(f"  {label}: {count}")


def main() -> None:
    """Main entry point for dataset ingestion."""
    parser = argparse.ArgumentParser(
        description="Download and ingest crop disease datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(KAGGLE_DATASETS.keys()) + ["all"],
        default=["plantvillage"],
        help="Datasets to download and ingest",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory to store downloaded datasets",
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/manifests",
        help="Directory to save manifest CSVs",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, only create manifests from existing data",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manifest_dir = Path(args.manifest_dir)

    # Determine datasets to process
    if "all" in args.datasets:
        datasets = list(KAGGLE_DATASETS.keys())
    else:
        datasets = args.datasets

    # Check Kaggle credentials if downloading
    if not args.skip_download:
        if not setup_kaggle_credentials():
            print("\nTo skip download and use existing data, run with --skip-download")
            sys.exit(1)

        # Download datasets
        for dataset_key in datasets:
            download_dataset(dataset_key, data_dir)

    # Ingest all datasets
    all_examples = []
    for dataset_key in datasets:
        examples = ingest_dataset(dataset_key, data_dir, manifest_dir)
        all_examples.extend(examples)

    if not all_examples:
        logger.error("No examples found! Check that datasets are downloaded correctly.")
        sys.exit(1)

    # Print combined statistics
    print_dataset_stats(all_examples, "Combined Dataset")

    # Create train/val/test splits
    train_examples, val_examples, test_examples = create_train_val_test_split(
        examples=all_examples,
        train_ratio=args.train_ratio,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed,
    )

    # Save manifests
    save_manifest(train_examples, manifest_dir / "train.csv")
    save_manifest(val_examples, manifest_dir / "val.csv")
    save_manifest(test_examples, manifest_dir / "test.csv")

    # Save combined manifest
    save_manifest(all_examples, manifest_dir / "all.csv")

    # Save metadata
    metadata = {
        "datasets": datasets,
        "total_samples": len(all_examples),
        "train_samples": len(train_examples),
        "val_samples": len(val_examples),
        "test_samples": len(test_examples),
        "classes": sorted(set(ex["label"] for ex in all_examples)),
        "num_classes": len(set(ex["label"] for ex in all_examples)),
        "created_at": pd.Timestamp.now().isoformat(),
    }

    (manifest_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"\n{'=' * 60}")
    print("Dataset Ingestion Complete!")
    print(f"{'=' * 60}")
    print(f"Train samples: {len(train_examples)}")
    print(f"Val samples: {len(val_examples)}")
    print(f"Test samples: {len(test_examples)}")
    print(f"Total classes: {metadata['num_classes']}")
    print(f"\nManifests saved to: {manifest_dir}")
    print("\nNext steps:")
    print("  1. Review manifests in data/manifests/")
    print("  2. Run training: python utils/train_vision.py --train-csv data/manifests/train.csv --val-csv data/manifests/val.csv")


if __name__ == "__main__":
    main()
